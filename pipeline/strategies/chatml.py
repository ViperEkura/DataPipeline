"""ChatML format strategy."""

from typing import Dict, List, Tuple

from pipeline.tokenize import AutoTokenizer
from pipeline.strategies.base import PromptStrategy
from pipeline.strategies.factory import StrategyFactory

DEFAULT_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


@StrategyFactory.register("chatml")
class ChatMLStrategy(PromptStrategy):
    """ChatML format strategy.

    Renders messages using the tokenizer's jinja chat_template from
    ``tokenizer_config.json``.  Falls back to DEFAULT_CHATML_TEMPLATE
    when no template is configured.

    The strategy does **not** hard-code any special tokens – all
    formatting is driven by the jinja template.
    """

    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)
        if tokenizer._chat_template is None:
            tokenizer.set_chat_template(DEFAULT_CHATML_TEMPLATE)

    @property
    def name(self) -> str:
        return "chatml"

    def format_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[List[int], List[int]]:
        """Render a single-turn messages conversation.

        Returns ``(prompt_tokens, response_tokens)`` where
        *prompt_tokens* contains everything up to (and including) the
        last assistant start marker, and *response_tokens* is the
        assistant content plus the closing markers.
        """
        last_asst = max(
            i for i, m in enumerate(messages) if m["role"] == "assistant"
        )

        prompt = self.tokenizer.apply_chat_template(
            messages[:last_asst],
            add_generation_prompt=True,
            tokenize=True,
        )
        full = self.tokenizer.apply_chat_template(
            messages[: last_asst + 1],
            add_generation_prompt=False,
            tokenize=True,
        )
        return prompt, full[len(prompt) :]

    def assemble_prompt(self, query_tokens: List[int]) -> List[int]:
        text = self.tokenizer.decode(query_tokens)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
        )

    def assemble_response(self, response_tokens: List[int]) -> List[int]:
        text = self.tokenizer.decode(response_tokens)
        full = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": text}],
            add_generation_prompt=False,
            tokenize=True,
        )
        opening = self.tokenizer.apply_chat_template(
            [], add_generation_prompt=True, tokenize=True
        )
        return full[len(opening) :]
