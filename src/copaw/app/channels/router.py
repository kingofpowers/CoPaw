# -*- coding: utf-8 -*-
"""Agent Router: route messages to different agent pipelines based on rules.

This module provides routing functionality to forward messages from the
channel manager entry point to different agent processing pipelines.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Pattern
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Type alias for routing decision result
RouteTarget = Optional[str]  # Target agent_id or None for default


@dataclass
class RouteRule:
    """A single routing rule.

    Attributes:
        name: Rule name for logging/debugging
        field: Field to match (e.g., 'session_id', 'user_id', 'content')
        pattern: Regex pattern to match against the field value
        target_agent: Target agent_id to route to when matched
        priority: Rule priority (higher = evaluated first)
    """
    name: str
    field: str
    pattern: Pattern[str]
    target_agent: str
    priority: int = 0


@dataclass
class RouterConfig:
    """Configuration for the agent router.

    Attributes:
        enabled: Whether routing is enabled
        default_agent: Default agent_id when no rule matches
        rules: List of routing rules
    """
    enabled: bool = False
    default_agent: Optional[str] = None
    rules: List[RouteRule] = field(default_factory=list)


class AgentRouter:
    """Routes incoming messages to appropriate agent pipelines.

    The router evaluates routing rules against message payload and determines
    which agent should handle the message. This enables multi-agent scenarios
    where different agents handle different types of conversations or users.
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        """Initialize router with configuration.

        Args:
            config: Router configuration. If None, routing is disabled.
        """
        self.config = config or RouterConfig()
        self._rules_sorted: List[RouteRule] = []
        self._compile_rules()

    def _compile_rules(self) -> None:
        """Sort rules by priority (descending)."""
        self._rules_sorted = sorted(
            self.config.rules,
            key=lambda r: r.priority,
            reverse=True,
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentRouter":
        """Create router from dictionary configuration.

        Example config:
            {
                "enabled": true,
                "default_agent": "default",
                "rules": [
                    {
                        "name": "support_team",
                        "field": "session_id",
                        "pattern": "^support-.*",
                        "target_agent": "support_agent",
                        "priority": 10
                    },
                    {
                        "name": "admin_users",
                        "field": "user_id",
                        "pattern": "^(admin|root)$",
                        "target_agent": "admin_agent",
                        "priority": 100
                    }
                ]
            }
        """
        enabled = config_dict.get("enabled", False)
        default_agent = config_dict.get("default_agent")

        rules = []
        for rule_dict in config_dict.get("rules", []):
            try:
                pattern = re.compile(rule_dict["pattern"])
                rule = RouteRule(
                    name=rule_dict.get("name", "unnamed"),
                    field=rule_dict["field"],
                    pattern=pattern,
                    target_agent=rule_dict["target_agent"],
                    priority=rule_dict.get("priority", 0),
                )
                rules.append(rule)
            except (KeyError, re.error) as e:
                logger.warning("Invalid routing rule: %s - %s", rule_dict, e)
                continue

        config = RouterConfig(
            enabled=enabled,
            default_agent=default_agent,
            rules=rules,
        )
        return cls(config)

    def route(self, payload: Any) -> RouteTarget:
        """Determine target agent for the given payload.

        Args:
            payload: Message payload (dict or object with attributes)

        Returns:
            Target agent_id or None if routing disabled/no match
        """
        if not self.config.enabled:
            return None

        # Convert payload to dict if it's an object
        data = self._normalize_payload(payload)

        # Evaluate rules in priority order
        for rule in self._rules_sorted:
            value = self._extract_field(data, rule.field)
            if value is not None and rule.pattern.search(str(value)):
                logger.debug(
                    "Route matched: rule=%s, field=%s, target=%s",
                    rule.name,
                    rule.field,
                    rule.target_agent,
                )
                return rule.target_agent

        # Return default if no rule matched
        return self.config.default_agent

    def _normalize_payload(self, payload: Any) -> Dict[str, Any]:
        """Convert payload to dictionary for field extraction."""
        if isinstance(payload, dict):
            return payload
        # Try to extract common attributes from object
        result = {}
        for attr in ["session_id", "user_id", "content", "text", "agent_id"]:
            if hasattr(payload, attr):
                result[attr] = getattr(payload, attr)
        return result

    def _extract_field(self, data: Dict[str, Any], field: str) -> Optional[str]:
        """Extract field value from data dictionary.

        Supports nested fields using dot notation (e.g., 'user.id').
        """
        parts = field.split(".")
        current: Any = data

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

            if current is None:
                return None

        return str(current) if current is not None else None

    def add_rule(self, rule: RouteRule) -> None:
        """Add a new routing rule at runtime."""
        self.config.rules.append(rule)
        self._compile_rules()
        logger.info("Added routing rule: %s -> %s", rule.name, rule.target_agent)

    def remove_rule(self, name: str) -> bool:
        """Remove a routing rule by name."""
        original_len = len(self.config.rules)
        self.config.rules = [r for r in self.config.rules if r.name != name]
        if len(self.config.rules) < original_len:
            self._compile_rules()
            logger.info("Removed routing rule: %s", name)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            "enabled": self.config.enabled,
            "default_agent": self.config.default_agent,
            "rule_count": len(self.config.rules),
            "rules": [
                {
                    "name": r.name,
                    "field": r.field,
                    "pattern": r.pattern.pattern,
                    "target_agent": r.target_agent,
                    "priority": r.priority,
                }
                for r in self._rules_sorted
            ],
        }
