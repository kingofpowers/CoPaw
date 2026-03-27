# -*- coding: utf-8 -*-
# pylint: disable=protected-access
# ChannelManager is the framework owner of BaseChannel and must call
# _is_native_payload and _consume_one_request as part of the contract.

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from typing import (
    Callable,
    List,
    Optional,
    Any,
    Dict,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from .base import BaseChannel, ContentType, ProcessHandler, TextContent
from .registry import get_channel_registry
from .router import AgentRouter, RouterConfig
from ...config import get_available_channels
from ...agents.command_handler import CommandHandler
from ...app.runner.daemon_commands import DAEMON_SUBCOMMANDS, DAEMON_SHORT_ALIASES

if TYPE_CHECKING:
    from ....config.config import Config

logger = logging.getLogger(__name__)

# Callback when user reply was sent: (channel, user_id, session_id)
OnLastDispatch = Optional[Callable[[str, str, str], None]]

# Default max size per channel queue
_CHANNEL_QUEUE_MAXSIZE = 1000

# Workers per channel: drain same-session from queue and process in parallel
_CONSUMER_WORKERS_PER_CHANNEL = 4


def _drain_same_key(
    q: asyncio.Queue,
    ch: BaseChannel,
    key: str,
    first_payload: Any,
) -> List[Any]:
    """Drain queue of payloads with same debounce key; return batch."""
    batch = [first_payload]
    put_back: List[Any] = []
    while True:
        try:
            p = q.get_nowait()
        except asyncio.QueueEmpty:
            break
        if ch.get_debounce_key(p) == key:
            batch.append(p)
        else:
            put_back.append(p)
    for p in put_back:
        q.put_nowait(p)
    return batch


async def _process_batch(ch: BaseChannel, batch: List[Any]) -> None:
    """Merge if needed and process one payload (native or request)."""
    try:
        if ch.channel == "dingtalk" and batch and ch._is_native_payload(batch[0]):
            first = batch[0] if isinstance(batch[0], dict) else {}
            logger.info(
                "manager _process_batch dingtalk: batch_len=%s first_has_sw=%s",
                len(batch),
                bool(first.get("session_webhook")),
            )
        if len(batch) > 1 and ch._is_native_payload(batch[0]):
            merged = ch.merge_native_items(batch)
            if ch.channel == "dingtalk" and isinstance(merged, dict):
                logger.info(
                    "manager _process_batch dingtalk merged: has_sw=%s",
                    bool(merged.get("session_webhook")),
                )
            await ch._consume_one_request(merged)
        elif len(batch) > 1:
            merged = ch.merge_requests(batch)
            if merged is not None:
                await ch._consume_one_request(merged)
            else:
                await ch.consume_one(batch[0])
        elif ch._is_native_payload(batch[0]):
            await ch._consume_one_request(batch[0])
        else:
            await ch.consume_one(batch[0])
    except asyncio.CancelledError:
        logger.info(f"Process batch cancelled for channel={ch.channel}")
        raise  # Re-raise to propagate cancellation


def _put_pending_merged(
    ch: BaseChannel,
    q: asyncio.Queue,
    pending: List[Any],
) -> None:
    """Merge pending items if multiple and put one or more on queue."""
    if not pending:
        return
    merged = None
    if len(pending) > 1 and ch._is_native_payload(pending[0]):
        merged = ch.merge_native_items(pending)
    elif len(pending) > 1:
        merged = ch.merge_requests(pending)
    if merged is not None:
        q.put_nowait(merged)
    else:
        for p in pending:
            q.put_nowait(p)


class ChannelManager:
    """Owns queues and consumer loops; channels define how to consume via
    consume_one(). Enqueue via enqueue(channel_id, payload) (thread-safe).
    """

    def __init__(
        self,
        channels: List[BaseChannel],
        router: Optional[AgentRouter] = None,
    ):
        self.channels = channels
        self.router = router
        self._lock = asyncio.Lock()
        self._queues: Dict[str, asyncio.Queue] = {}
        self._consumer_tasks: List[asyncio.Task[None]] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Session in progress: (channel_id, debounce_key) -> True while worker
        # is processing. New payloads for that key go to _pending, merged
        # when worker finishes.
        self._in_progress: Set[Tuple[str, str]] = set()
        self._pending: Dict[Tuple[str, str], List[Any]] = {}
        # Per-key lock: same session is claimed by one worker for drain so
        # [image1, text] are not split across workers (avoids no-text
        # debounce reordering and duplicate content in AgentRequest).
        self._key_locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        # Session tasks: (channel_id, session_key) -> Task for cancellation
        self._session_tasks: Dict[Tuple[str, str], asyncio.Task] = {}

    @classmethod
    def from_env(
        cls,
        process: ProcessHandler,
        on_last_dispatch: OnLastDispatch = None,
    ) -> "ChannelManager":
        """
        Create channels from env and inject unified process
        (AgentRequest -> Event stream).
        process is typically runner.stream_query, handled by AgentApp's
        process endpoint.
        on_last_dispatch: called when a user send+reply was sent.
        """
        available = get_available_channels()
        registry = get_channel_registry()
        channels: list[BaseChannel] = [
            ch_cls.from_env(process, on_reply_sent=on_last_dispatch)
            for key, ch_cls in registry.items()
            if key in available
        ]
        return cls(channels)

    @classmethod
    # pylint: disable=too-many-branches
    def from_config(
        cls,
        process: ProcessHandler,
        config: "Config",
        on_last_dispatch: OnLastDispatch = None,
        workspace_dir: Path | None = None,
    ) -> "ChannelManager":
        """Create channels from config (config.json or agent.json).

        Args:
            process: Process handler for agent communication
            config: Configuration object with channels
            on_last_dispatch: Callback for dispatch events
            workspace_dir: Agent workspace directory for channel state files
        """
        available = get_available_channels()
        ch = config.channels
        show_tool_details = getattr(config, "show_tool_details", True)
        extra = getattr(ch, "__pydantic_extra__", None) or {}

        channels: list[BaseChannel] = []
        for key, ch_cls in get_channel_registry().items():
            if key not in available:
                continue
            ch_cfg = getattr(ch, key, None)
            if ch_cfg is None and key in extra:
                ch_cfg = extra[key]
            if ch_cfg is None:
                continue
            if isinstance(ch_cfg, dict):
                from types import SimpleNamespace
                from ...config.config import BaseChannelConfig

                defaults = BaseChannelConfig().model_dump()
                defaults.update(ch_cfg)
                ch_cfg = SimpleNamespace(**defaults)

            # Check if channel is enabled
            # Handle both Pydantic objects (built-in)
            # and dicts (customchannels)
            if isinstance(ch_cfg, dict):
                enabled = ch_cfg.get("enabled", False)
            else:
                enabled = getattr(ch_cfg, "enabled", False)
            if not enabled:
                continue

            # Handle both Pydantic objects (built-in)
            # and dicts (custom channels)
            if isinstance(ch_cfg, dict):
                filter_tool_messages = ch_cfg.get(
                    "filter_tool_messages",
                    False,
                )
                filter_thinking = ch_cfg.get("filter_thinking", False)
            else:
                filter_tool_messages = getattr(
                    ch_cfg,
                    "filter_tool_messages",
                    False,
                )
                filter_thinking = getattr(
                    ch_cfg,
                    "filter_thinking",
                    False,
                )

            from_config_kwargs = {
                "process": process,
                "config": ch_cfg,
                "on_reply_sent": on_last_dispatch,
                "show_tool_details": show_tool_details,
                "filter_tool_messages": filter_tool_messages,
                "filter_thinking": filter_thinking,
                "workspace_dir": workspace_dir,
            }

            # Only pass kwargs that the channel's from_config accepts
            import inspect

            sig = inspect.signature(ch_cls.from_config)
            if any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            ):
                filtered_kwargs = from_config_kwargs
            else:
                filtered_kwargs = {
                    k: v
                    for k, v in from_config_kwargs.items()
                    if k in sig.parameters
                }

            try:
                channels.append(ch_cls.from_config(**filtered_kwargs))
            except Exception as e:
                logger.warning(
                    "Failed to initialize channel '%s', skipping: %s",
                    key,
                    e,
                )
                continue

        return cls(channels)

    def _make_enqueue_cb(self, channel_id: str) -> Callable[[Any], None]:
        """Return a callback that enqueues payload for the given channel."""

        def cb(payload: Any) -> None:
            self.enqueue(channel_id, payload)

        return cb

    def _enqueue_one(self, channel_id: str, payload: Any) -> None:
        """Run on event loop: enqueue or append to pending if session in
        progress.
        """
        q = self._queues.get(channel_id)
        if not q:
            logger.debug("enqueue: no queue for channel=%s", channel_id)
            return
        ch = next(
            (c for c in self.channels if c.channel == channel_id),
            None,
        )
        if not ch:
            q.put_nowait(payload)
            return
        key = ch.get_debounce_key(payload)
        if channel_id == "dingtalk" and isinstance(payload, dict):
            logger.info(
                "manager _enqueue_one dingtalk: key=%s in_progress=%s "
                "payload_has_sw=%s -> %s",
                key,
                (channel_id, key) in self._in_progress,
                bool(payload.get("session_webhook")),
                "pending"
                if (channel_id, key) in self._in_progress
                else "queue",
            )
        if (channel_id, key) in self._in_progress:
            self._pending.setdefault((channel_id, key), []).append(payload)
            return
        q.put_nowait(payload)

    def _extract_text_from_content_parts(self, parts: list) -> str | None:
        """Extract text from content_parts list.

        Args:
            parts: List of content parts (dicts or objects with 'type' and content fields)

        Returns:
            Concatenated text content or None
        """
        if not parts:
            return None

        texts = []
        for part in parts:
            # Handle dict format
            if isinstance(part, dict):
                part_type = part.get("type", "")
                if part_type == "text":
                    text = part.get("text", "")
                    if text:
                        texts.append(text)
                elif part_type in ("image", "file"):
                    continue
                else:
                    text = part.get("content") or part.get("text", "")
                    if text:
                        texts.append(text)

            # Handle object format (e.g., TextContent with attributes)
            elif hasattr(part, "type"):
                part_type = getattr(part, "type", "")
                if part_type == "text":
                    text = getattr(part, "text", None) or getattr(part, "content", None) or ""
                    if text:
                        texts.append(text)
                elif part_type in ("image", "file"):
                    continue
                else:
                    text = getattr(part, "text", None) or getattr(part, "content", None) or ""
                    if text:
                        texts.append(text)

            # Handle plain string
            elif isinstance(part, str):
                texts.append(part)

        return " ".join(texts) if texts else None

    def _extract_text_from_payload(self, payload: Any) -> str | None:
        """Extract text content from payload for command detection.

        Args:
            payload: Channel payload (dict or AgentRequest)

        Returns:
            Text content if found, None otherwise
        """
        if payload is None:
            return None

        # Handle dict payload (native format, e.g., OneBot)
        if isinstance(payload, dict):
            # Try content_parts first (OneBot format)
            content_parts = payload.get("content_parts")
            if isinstance(content_parts, list) and content_parts:
                text = self._extract_text_from_content_parts(content_parts)
                if text:
                    return text

            # Try content or text fields
            content = payload.get("content") or payload.get("text", "")
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                text = self._extract_text_from_content_parts(content)
                if text:
                    return text
            return None

        # Handle AgentRequest-like objects
        if hasattr(payload, "content"):
            content = payload.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text = self._extract_text_from_content_parts(content)
                if text:
                    return text

        # Handle content_parts attribute directly
        if hasattr(payload, "content_parts"):
            parts = payload.content_parts
            if isinstance(parts, list) and parts:
                text = self._extract_text_from_content_parts(parts)
                if text:
                    return text

        # Handle input attribute
        if hasattr(payload, "input"):
            inp = payload.input
            if isinstance(inp, str):
                return inp
            if isinstance(inp, list) and inp:
                last = inp[-1]
                if hasattr(last, "get_text_content"):
                    return last.get_text_content()
                if isinstance(last, dict):
                    return last.get("content") or last.get("text", "")

        return None

    def _is_slash_command(self, text: str | None) -> bool:
        """Check if text is a slash command that should bypass queue.

        Checks against:
        - CommandHandler.SYSTEM_COMMANDS (conversation commands)
        - DAEMON_SUBCOMMANDS (daemon commands)
        - DAEMON_SHORT_ALIASES (daemon shortcuts)

        Args:
            text: Text content to check

        Returns:
            True if this is a slash command
        """
        if not text:
            return False
        text = text.strip()
        if not text.startswith("/"):
            return False
        # Extract command name (first word after /)
        cmd = text[1:].split()[0] if len(text) > 1 else ""

        # Check against all known command sets
        all_commands = (
            CommandHandler.SYSTEM_COMMANDS
            | DAEMON_SUBCOMMANDS
            | set(DAEMON_SHORT_ALIASES.keys())
        )

        return cmd in all_commands

    def enqueue(self, channel_id: str, payload: Any) -> None:
        """Enqueue a payload for the channel. Thread-safe (e.g. from sync
        WebSocket or polling thread). If this session is already being
        processed, payload is held in pending and merged when the worker
        finishes. Call after start_all().

        If a router is configured, the payload will be routed to the
        appropriate agent pipeline based on routing rules.

        Slash commands bypass the queue and are processed immediately.
        """
        if not self._queues.get(channel_id):
            logger.debug("enqueue: no queue for channel=%s", channel_id)
            return
        if self._loop is None:
            logger.warning("enqueue: loop not set for channel=%s", channel_id)
            return

        # Check if this is a slash command that should bypass queue
        text = self._extract_text_from_payload(payload)
        if self._is_slash_command(text):
            logger.info(
                "Slash command bypassing queue: channel=%s, cmd=%s",
                channel_id,
                text.strip().split()[0] if text else "",
            )
            # Process directly without queuing
            self._loop.call_soon_threadsafe(
                self._process_slash_command,
                channel_id,
                payload,
            )
            return

        # Route payload to target agent if router is configured
        if self.router and self.router.config.enabled:
            target_agent = self.router.route(payload)
            if target_agent:
                # Attach routing info to payload
                payload = self._attach_routing_info(payload, target_agent)
                logger.debug(
                    "Routed payload: channel=%s, target_agent=%s",
                    channel_id,
                    target_agent,
                )

        self._loop.call_soon_threadsafe(
            self._enqueue_one,
            channel_id,
            payload,
        )

    def _attach_routing_info(self, payload: Any, target_agent: str) -> Any:
        """Attach routing information to payload.

        Args:
            payload: Original payload
            target_agent: Target agent_id from routing

        Returns:
            Payload with routing info attached
        """
        # If payload is a dict, add routing info directly
        if isinstance(payload, dict):
            payload = dict(payload)  # Make a copy
            payload["_router_target_agent"] = target_agent
            return payload

        # If payload has attributes, try to set routing info
        if hasattr(payload, "__dict__"):
            try:
                payload._router_target_agent = target_agent
                return payload
            except AttributeError:
                pass

        # Wrap in a routing envelope as last resort
        return {
            "_router_envelope": True,
            "_router_target_agent": target_agent,
            "_router_payload": payload,
        }

    def _process_slash_command(
        self,
        channel_id: str,
        payload: Any,
    ) -> None:
        """Process a slash command directly without queuing.

        This method is called from the event loop thread (via call_soon_threadsafe)
        to process slash commands immediately, bypassing the normal queue mechanism.

        Args:
            channel_id: Channel identifier
            payload: Command payload
        """
        # Create a task to process the command asynchronously
        asyncio.create_task(
            self._process_slash_command_async(channel_id, payload),
        )

    async def _process_slash_command_async(
        self,
        channel_id: str,
        payload: Any,
    ) -> None:
        """Async implementation of slash command processing.

        Args:
            channel_id: Channel identifier
            payload: Command payload
        """
        try:
            ch = await self.get_channel(channel_id)
            if not ch:
                logger.warning(
                    "Cannot process slash command: channel=%s not found",
                    channel_id,
                )
                return

            logger.debug(
                "Processing slash command directly: channel=%s",
                channel_id,
            )

            # Convert payload to AgentRequest
            request = ch._payload_to_request(payload)

            # Process the request directly (same as consume_one but without queue)
            await ch._consume_one_request(request)

            logger.debug(
                "Slash command processed successfully: channel=%s",
                channel_id,
            )

        except Exception:
            logger.exception(
                "Failed to process slash command: channel=%s",
                channel_id,
            )

    async def _consume_channel_loop(
        self,
        channel_id: str,
        worker_index: int,
    ) -> None:
        """
        Run one consumer worker: pop payload, drain queue of same session,
        mark session in progress, merge batch (native or requests), process
        once, then flush any pending for this session (merged) back to queue.
        Multiple workers per channel allow different sessions in parallel.
        """
        q = self._queues.get(channel_id)
        if not q:
            return
        while True:
            try:
                payload = await q.get()
                ch = await self.get_channel(channel_id)
                if not ch:
                    continue
                key = ch.get_debounce_key(payload)
                key_lock = self._key_locks.setdefault(
                    (channel_id, key),
                    asyncio.Lock(),
                )
                try:
                    async with key_lock:
                        self._in_progress.add((channel_id, key))
                        batch = _drain_same_key(q, ch, key, payload)
                    # Wrap processing in a Task for cancellation support
                    process_task = asyncio.create_task(
                        _process_batch(ch, batch),
                        name=f"process_{channel_id}_{key}",
                    )
                    self._session_tasks[(channel_id, key)] = process_task
                    try:
                        await process_task
                    finally:
                        self._session_tasks.pop((channel_id, key), None)
                finally:
                    self._in_progress.discard((channel_id, key))
                    pending = self._pending.pop((channel_id, key), [])
                    _put_pending_merged(ch, q, pending)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(
                    "channel consume_one failed: channel=%s worker=%s",
                    channel_id,
                    worker_index,
                )

    async def start_all(self) -> None:
        self._loop = asyncio.get_running_loop()
        async with self._lock:
            snapshot = list(self.channels)
        for ch in snapshot:
            if getattr(ch, "uses_manager_queue", True):
                self._queues[ch.channel] = asyncio.Queue(
                    maxsize=_CHANNEL_QUEUE_MAXSIZE,
                )
                ch.set_enqueue(self._make_enqueue_cb(ch.channel))
        for ch in snapshot:
            if ch.channel in self._queues:
                for w in range(_CONSUMER_WORKERS_PER_CHANNEL):
                    task = asyncio.create_task(
                        self._consume_channel_loop(ch.channel, w),
                        name=f"channel_consumer_{ch.channel}_{w}",
                    )
                    self._consumer_tasks.append(task)
        logger.debug(
            "starting channels=%s queues=%s",
            [g.channel for g in snapshot],
            list(self._queues.keys()),
        )
        for g in snapshot:
            try:
                await g.start()
            except Exception:
                logger.exception(f"failed to start channels={g.channel}")

    async def stop_all(self) -> None:
        self._in_progress.clear()
        self._pending.clear()
        for task in self._consumer_tasks:
            task.cancel()
        if self._consumer_tasks:
            _, pending = await asyncio.wait(
                self._consumer_tasks,
                timeout=5.0,
                return_when=asyncio.ALL_COMPLETED,
            )
            if pending:
                logger.warning(
                    "stop_all: %s consumer task(s) still pending after 5s",
                    len(pending),
                )
        self._consumer_tasks.clear()
        self._queues.clear()
        self._session_tasks.clear()
        async with self._lock:
            snapshot = list(self.channels)
        for ch in snapshot:
            ch.set_enqueue(None)

        async def _stop(ch):
            try:
                await ch.stop()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(f"failed to stop channels={ch.channel}")

        await asyncio.gather(*[_stop(g) for g in reversed(snapshot)])

    async def get_channel(self, channel: str) -> Optional[BaseChannel]:
        async with self._lock:
            for ch in self.channels:
                if ch.channel == channel:
                    return ch
            return None

    async def replace_channel(
        self,
        new_channel: BaseChannel,
    ) -> None:
        """Replace a single channel by name.

        Flow: ensure queue+enqueue for new channel → start new (outside lock)
        → swap + stop old (inside lock). Lock only guards the swap+stop.

        Args:
            new_channel: New channel instance to replace with
        """
        new_channel_name = new_channel.channel
        # 1) Ensure queue and enqueue callback before start() so the channel
        #    (e.g. DingTalk) registers its handler with a valid callback.
        if new_channel_name not in self._queues:
            if getattr(new_channel, "uses_manager_queue", True):
                self._queues[new_channel_name] = asyncio.Queue(
                    maxsize=_CHANNEL_QUEUE_MAXSIZE,
                )
                for w in range(_CONSUMER_WORKERS_PER_CHANNEL):
                    task = asyncio.create_task(
                        self._consume_channel_loop(new_channel_name, w),
                        name=f"channel_consumer_{new_channel_name}_{w}",
                    )
                    self._consumer_tasks.append(task)
        new_channel.set_enqueue(self._make_enqueue_cb(new_channel_name))

        # 2) Start new channel outside lock (may be slow, e.g. DingTalk stream)
        logger.info(f"Pre-starting new channel: {new_channel_name}")
        try:
            await new_channel.start()
        except Exception:
            logger.exception(
                f"Failed to start new channel: {new_channel_name}",
            )
            try:
                await new_channel.stop()
            except Exception:
                pass
            raise

        # 3) Swap + stop old inside lock
        async with self._lock:
            old_channel = None
            for i, ch in enumerate(self.channels):
                if ch.channel == new_channel_name:
                    old_channel = ch
                    self.channels[i] = new_channel
                    break

            if old_channel is None:
                logger.info(f"Adding new channel: {new_channel_name}")
                self.channels.append(new_channel)
            else:
                logger.info(f"Stopping old channel: {old_channel.channel}")
                try:
                    await old_channel.stop()
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception(
                        f"Failed to stop old channel: {old_channel.channel}",
                    )

    async def send_event(
        self,
        *,
        channel: str,
        user_id: str,
        session_id: str,
        event: Any,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        ch = await self.get_channel(channel)
        if not ch:
            raise KeyError(f"channel not found: {channel}")
        merged_meta = dict(meta or {})
        merged_meta["session_id"] = session_id
        merged_meta["user_id"] = user_id
        bot_prefix = getattr(ch, "bot_prefix", None) or getattr(
            ch,
            "_bot_prefix",
            None,
        )
        if bot_prefix and "bot_prefix" not in merged_meta:
            merged_meta["bot_prefix"] = bot_prefix
        await ch.send_event(
            user_id=user_id,
            session_id=session_id,
            event=event,
            meta=merged_meta,
        )

    async def send_text(
        self,
        *,
        channel: str,
        user_id: str,
        session_id: str,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send plain text to a specific channel
        (used for scheduled jobs like task_type='text').
        """
        ch = await self.get_channel(channel.lower())
        if not ch:
            raise KeyError(f"channel not found: {channel}")

        # Convert (user_id, session_id) into the channel-specific target handle
        to_handle = ch.to_handle_from_target(
            user_id=user_id,
            session_id=session_id,
        )
        ch_name = getattr(ch, "channel", channel)
        logger.info(
            "channel send_text: channel=%s user_id=%s session_id=%s "
            "to_handle=%s",
            ch_name,
            (user_id or "")[:40],
            (session_id or "")[:40],
            (to_handle or "")[:60],
        )

        # Keep the same behavior as the agent pipeline:
        # if the channel has a fixed bot prefix, merge it into meta so
        # send_content_parts can use it.
        merged_meta = dict(meta or {})
        bot_prefix = getattr(ch, "bot_prefix", None) or getattr(
            ch,
            "_bot_prefix",
            None,
        )
        if bot_prefix and "bot_prefix" not in merged_meta:
            merged_meta["bot_prefix"] = bot_prefix
        merged_meta["session_id"] = session_id
        merged_meta["user_id"] = user_id

        # Send as content parts (single text part; use TextContent so channel
        # getattr(p, "type") / getattr(p, "text") work)
        await ch.send_content_parts(
            to_handle,
            [TextContent(type=ContentType.TEXT, text=text)],
            merged_meta,
        )

    def get_session_status(
        self,
        channel_id: str,
        session_key: str,
    ) -> dict:
        """Get session processing status and pending count.

        Args:
            channel_id: Channel identifier
            session_key: Session debounce key (usually "user_id:session_id")

        Returns:
            dict with:
                - is_processing: bool, whether session is being processed
                - pending_count: int, number of pending messages
                - queue_size: int, total queue size for the channel
        """
        key = (channel_id, session_key)
        is_processing = key in self._in_progress
        pending_count = len(self._pending.get(key, []))
        queue_size = 0
        if channel_id in self._queues:
            queue_size = self._queues[channel_id].qsize()

        return {
            "is_processing": is_processing,
            "pending_count": pending_count,
            "queue_size": queue_size,
        }

    async def skip_session(
        self,
        channel_id: str,
        session_key: str,
    ) -> dict:
        """Skip/clear pending messages for a session and cancel running task.

        Args:
            channel_id: Channel identifier
            session_key: Session debounce key

        Returns:
            dict with:
                - cleared_count: int, number of pending messages cleared
                - was_processing: bool, whether session was being processed
                - cancelled: bool, whether running task was cancelled
        """
        key = (channel_id, session_key)
        was_processing = key in self._in_progress

        # Clear pending messages for this session
        cleared = self._pending.pop(key, [])
        cleared_count = len(cleared)

        # Cancel running task if exists
        cancelled = False
        task = self._session_tasks.get(key)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            cancelled = True

        return {
            "cleared_count": cleared_count,
            "was_processing": was_processing,
            "cancelled": cancelled,
        }
