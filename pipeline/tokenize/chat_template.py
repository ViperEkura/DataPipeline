"""
Chat template module with Jinja2 rendering support.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Optional

from jinja2 import Template


# Message type for chat messages
type MessageType = Dict[str, Any]


@dataclass
class ChatTemplate:
    """A chat template with Jinja2 rendering support.

    Attributes:
        name: Unique identifier for the template.
        template_str: Jinja2 template string.
        description: Optional description.
        default_variables: Optional dictionary of default variable values
            that will be passed to the template if not overridden during rendering.
        special_tokens: Optional dictionary mapping token names to their string values.
            These tokens are automatically added to the template variables.
    """

    name: str
    template_str: str
    description: str = ""
    default_variables: Dict[str, Any] = field(default_factory=dict)
    special_tokens: Dict[str, str] = field(default_factory=dict)

    @cached_property
    def _compiled(self) -> Template:
        return Template(self.template_str)

    @classmethod
    def from_string(
        cls,
        template_str: str,
        description: str = "",
        default_variables: Optional[Dict[str, Any]] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ) -> "ChatTemplate":
        """Create a ChatTemplate instance directly from a template string."""
        return cls(
            name="",  # empty name for ad-hoc templates
            template_str=template_str,
            description=description,
            default_variables=default_variables or {},
            special_tokens=special_tokens or {},
        )

    def render(
        self,
        messages: List[MessageType],
        system_prompt: Optional[str] = None,
        add_generation_prompt: bool = True,
        **extra_variables: Any,
    ) -> str:
        """Render the template with given messages and variables.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            system_prompt: Optional system prompt string.
            add_generation_prompt: Whether to add generation prompt after messages.
            **extra_variables: Additional variables to pass to the template.
                These override default_variables and special_tokens.

        Returns:
            Rendered prompt string.
        """
        # Merge default variables, special tokens, and extra variables
        variables = {
            **self.default_variables,
            **self.special_tokens,
            **extra_variables,
        }
        variables["messages"] = messages
        variables["add_generation_prompt"] = add_generation_prompt
        if system_prompt is not None:
            variables["system_prompt"] = system_prompt

        return self._compiled.render(**variables)


# Default ChatML template
DEFAULT_CHATML_TEMPLATE = """{% for message in messages %}{{ bos_token }}{{ message['role'] }}
{{ message['content'] }}{{ eos_token }}{% endfor %}{% if add_generation_prompt %}{{ bos_token }}assistant
{% endif %}"""


# Pre-built template registry
TEMPLATE_REGISTRY: Dict[str, ChatTemplate] = {}


def register_chat_template(name: str, template: ChatTemplate) -> None:
    """Register a chat template in the global registry.
    
    Args:
        name: Name to register the template under
        template: ChatTemplate instance
    """
    TEMPLATE_REGISTRY[name] = template


def get_chat_template(name: str) -> ChatTemplate:
    """Get a registered chat template.
    
    Args:
        name: Template name
        
    Returns:
        ChatTemplate instance
        
    Raises:
        KeyError: If template not found
    """
    if name not in TEMPLATE_REGISTRY:
        raise KeyError(f"Chat template '{name}' not found. Available: {list(TEMPLATE_REGISTRY.keys())}")
    return TEMPLATE_REGISTRY[name]
