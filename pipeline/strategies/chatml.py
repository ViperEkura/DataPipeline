"""ChatML format strategy."""

from typing import List

from pipeline.tokenize import AutoTokenizer
from pipeline.strategies.base import PromptStrategy
from pipeline.strategies.factory import StrategyFactory


@StrategyFactory.register("chatml")
class ChatMLStrategy(PromptStrategy):
    """ChatML format strategy."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        user_start: str = "<｜im▁start｜>user",
        user_end: str = "<｜im▁end｜>",
        assistant_start: str = "<｜im▁start｜>assistant",
        assistant_end: str = "<｜im▁end｜>",
    ):
        super().__init__(tokenizer)
        nl_id = tokenizer.encode("a\nb", add_special_tokens=False)[1]

        self._user_start_ids = self._encode_format(user_start) + [nl_id]
        self._user_end_ids = self._encode_format(user_end) + [nl_id]
        self._assistant_start_ids = self._encode_format(assistant_start) + [nl_id]
        self._assistant_end_ids = self._encode_format(assistant_end) + [nl_id]

    @property
    def name(self) -> str:
        return "chatml"

    def assemble_prompt(self, query_tokens: List[int]) -> List[int]:
        return (
            self._user_start_ids
            + query_tokens
            + self._user_end_ids
            + self._assistant_start_ids
        )

    def assemble_response(self, response_tokens: List[int]) -> List[int]:
        return response_tokens + self._assistant_end_ids
