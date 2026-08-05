"""
Tokenizer module with BPE implementation and auto-loading support.
"""

from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tokenizers import Tokenizer
from tokenizers import decoders, processors, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from jinja2 import Template


DEFAULT_SPECIAL_TOKENS = {
    "bos_token": "<｜begin▁of▁sentence｜>",
    "eos_token": "<｜end▁of▁sentence｜>",
    "pad_token": "<｜▁pad▁｜>",
}

CONTROL_TOKENS = [
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜▁pad▁｜>",
]

SPECIAL_TOKENS = ["<｜im▁start｜>", "<｜im▁end｜>"]


def train_bpe_tokenizer(
    files: List[str],
    vocab_size: int,
    min_freq: int = 2,
    reserved_token_size: int = 100,
    max_token_length: int = 18,
) -> Tokenizer:

    reserved_tokens = [
        f"<｜reserve{i:02d}｜>"
        for i in range(reserved_token_size - len(SPECIAL_TOKENS))
    ]
    detail_vocab_size = vocab_size - (len(reserved_tokens) + len(SPECIAL_TOKENS))

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    min_size = len(alphabet) + len(CONTROL_TOKENS)
    assert detail_vocab_size > min_size

    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Strip()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = BpeTrainer(
        vocab_size=detail_vocab_size,
        min_frequency=min_freq,
        limit_alphabet=detail_vocab_size // 6,
        max_token_length=max_token_length,
        special_tokens=CONTROL_TOKENS + SPECIAL_TOKENS,
        initial_alphabet=alphabet,
        show_progress=True,
    )

    tokenizer.train(files=files, trainer=trainer)
    tokenizer.add_special_tokens(CONTROL_TOKENS + SPECIAL_TOKENS + reserved_tokens)

    return tokenizer


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
    default_variables: Dict[str, Any] = None
    special_tokens: Dict[str, str] = None

    def __post_init__(self):
        if self.default_variables is None:
            self.default_variables = {}
        if self.special_tokens is None:
            self.special_tokens = {}

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
            name="",  # empty name for ad‑hoc templates
            template_str=template_str,
            description=description,
            default_variables=default_variables,
            special_tokens=special_tokens,
        )

    def render(
        self,
        messages: List[MessageType],
        system_prompt: Optional[str] = None,
        **extra_variables: Any,
    ) -> str:
        """Render the template with given messages and variables.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            system_prompt: Optional system prompt string.
            **extra_variables: Additional variables to pass to the template.
                These override default_variables and special_tokens.

        Returns:
            Rendered prompt string.
        """
        # Merge default variables, special tokens, and extra variables
        variables = {**self.default_variables, **self.special_tokens, **extra_variables}
        variables["messages"] = messages
        if system_prompt is not None:
            variables["system_prompt"] = system_prompt

        return self._compiled.render(**variables)



class AutoTokenizer:
    """Base tokenizer class with automatic loading support"""

    TOKENIZER_CLASSES = {}  # Registry for auto-loading

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        special_token_map: Optional[Dict[str, str]] = None,
        chat_template: Optional[str] = None,
    ):
        self._tokenizer: Tokenizer = None
        self._chat_template: Optional[ChatTemplate] = None
        self._special_token_map: Optional[Dict] = special_token_map or {}

        if chat_template:
            self.set_chat_template(chat_template)

        if path:
            self.load(path)

    def load(self, path: Union[str, Path]):
        """Load tokenizer from directory."""
        path = Path(path)
        tokenizer_file = path / "tokenizer.json"
        config_file = path / "tokenizer_config.json"
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))

        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            if "special_tokens" in config:
                self._special_token_map.update(config["special_tokens"])

            # Load chat template from config
            if "chat_template" in config:
                self.set_chat_template(config["chat_template"])

    @classmethod
    def from_pretrained(cls, path: Union[str, Path], **kwargs) -> "AutoTokenizer":
        """Load tokenizer from pretrained directory."""
        instance = cls(path)
        return instance

    def save_pretrained(self, save_path: str):
        """
        Save tokenizer to pretrained directory.

        Args:
            save_path: Path to save the tokenizer
        """

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save tokenizer
        self._tokenizer.save(str(save_path / "tokenizer.json"))

        # Save tokenizer config
        config = {}
        if self._special_token_map is not None:
            config["special_tokens"] = self._special_token_map
        if self._chat_template is not None:
            config["chat_template"] = self._chat_template.template_str

        with open(save_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def register_tokenizer(cls, name: str, tokenizer_class: type):
        """
        Register a new tokenizer class.

        Args:
            name: Name to register the tokenizer class under
            tokenizer_class: The tokenizer class to register
        """
        cls.TOKENIZER_CLASSES[name] = tokenizer_class

    def encode(
        self,
        tokens: Union[str, List[str]],
        out_ids: bool = True,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> List:
        """Encode text to tokens or token IDs."""
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer not initialized. Load or create a tokenizer first."
            )

        single = isinstance(tokens, str)
        if single:
            tokens = [tokens]
        encoded_list = self._tokenizer.encode_batch(
            tokens,
            is_pretokenized=is_pretokenized,
            add_special_tokens=add_special_tokens,
        )
        result = [
            encoded.ids if out_ids else encoded.tokens for encoded in encoded_list
        ]
        return result[0] if single else result

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer not initialized. Load or create a tokenizer first."
            )

        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert a token string to its integer ID."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized.")
        return self._tokenizer.token_to_id(token)

    def __len__(self) -> int:
        if self._tokenizer is None:
            return 0
        return self._tokenizer.get_vocab_size()

    def __getattr__(self, key: str):
        """
        Dynamically intercept special token attribute access.
        Supports three forms:
          - tokenizer.bos_token   → returns string
          - tokenizer.bos_token_id → returns corresponding integer ID
          - tokenizer.stop_ids → returns list of corresponding integer IDs for all special tokens
        """
        # Handle stop_ids - return IDs for all special tokens
        if key == "stop_ids":
            stop_ids = []

            if self._tokenizer is None:
                return stop_ids

            for val in self._special_token_map.values():
                token_id = self._tokenizer.token_to_id(val)
                if token_id is not None:
                    stop_ids.append(token_id)

            return stop_ids

        # Handle _id suffix (e.g., bos_token_id -> bos_token)
        if key.endswith("_id"):
            base_attr = key[:-3]  # Remove "_id"
            token_str = self._special_token_map.get(base_attr)
            if token_str is None:
                return None
            if self._tokenizer is None:
                raise RuntimeError("Tokenizer not loaded, cannot convert token to id.")
            return self._tokenizer.token_to_id(token_str)

        # Handle regular string attributes
        if key in self._special_token_map:
            return self._special_token_map.get(key)

        # Other attributes trigger default AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    @property
    def vocab_size(self) -> int:
        return len(self)

    def set_chat_template(self, template: Union[str, ChatTemplate]):
        """
        Set the chat template for the tokenizer.

        Args:
            template: Either a template name (str) registered in the global registry,
                      or a ChatTemplate instance, or a Jinja2 template string.

        Raises:
            KeyError: If template name is not registered.
        """
        if isinstance(template, str):
            self._chat_template = ChatTemplate.from_string(
                template, special_tokens=self._special_token_map
            )
        elif isinstance(template, ChatTemplate):
            self._chat_template = template
        else:
            raise ValueError("Invalid template type, must be str or ChatTemplate.")

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> Union[str, List[int]]:
        """
        Apply the chat template to messages and optionally tokenize the result.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            system_prompt: Optional system prompt string (auto-converted to first message).
            tokenize: Whether to return token IDs (True) or raw string (False).
            add_generation_prompt: Whether to add the generation prompt (default: True).
            **kwargs: Additional variables to pass to the template.

        Returns:
            Either the rendered string or list of token IDs.

        Raises:
            RuntimeError: If chat template is not set.
        """
        if self._chat_template is None:
            raise RuntimeError(
                "Chat template not set. Use set_chat_template() to set a template first."
            )

        # Auto-convert system_prompt to first message if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + list(messages)

        # Render the template
        rendered = self._chat_template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

        if tokenize:
            return self.encode(rendered)

        return rendered
