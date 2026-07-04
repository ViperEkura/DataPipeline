"""Tests for strategy module."""

import pytest
from pipeline.strategies import (
    PromptStrategy,
    ChatMLStrategy,
    AlpacaStrategy,
    StrategyFactory,
)


class DummyTokenizer:
    def __init__(self):
        self._special_token_map = {}
        self._chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(t) for t in tokens)

    def token_to_id(self, token: str):
        return ord(token)

    def set_chat_template(self, template):
        self._chat_template = template

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        text = ""
        for m in messages:
            text += f"<｜im▁start｜>{m['role']}\n{m['content']}<｜im▁end｜>\n"
        if add_generation_prompt:
            text += "<｜im▁start｜>assistant\n"
        return self.encode(text) if tokenize else text


class DummyStrategy(PromptStrategy):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    @property
    def name(self) -> str:
        return "dummy"

    def assemble_prompt(self, query_tokens):
        prefix = self._encode_format("Q: ")
        return prefix + query_tokens

    def assemble_response(self, response_tokens):
        suffix = self._encode_format("<｜end▁of▁sentence｜>")
        return response_tokens + suffix


def _decode(tokens):
    return "".join(chr(t) for t in tokens)


class TestChatMLStrategy:
    def test_name(self):
        assert ChatMLStrategy(DummyTokenizer()).name == "chatml"

    def test_assemble_prompt(self):
        tk = DummyTokenizer()
        strategy = ChatMLStrategy(tk)
        query_tokens = tk.encode("hello")
        prompt = strategy.assemble_prompt(query_tokens)
        text = _decode(prompt)
        assert "<｜im▁start｜>user" in text
        assert "hello" in text
        assert "<｜im▁start｜>assistant" in text

    def test_assemble_response(self):
        tk = DummyTokenizer()
        strategy = ChatMLStrategy(tk)
        response_tokens = tk.encode("world")
        response = strategy.assemble_response(response_tokens)
        text = _decode(response)
        assert "world" in text
        assert "<｜im▁end｜>" in text

    def test_prompt_ends_with_assistant_start(self):
        tk = DummyTokenizer()
        strategy = ChatMLStrategy(tk)
        prompt = strategy.assemble_prompt(tk.encode("hi"))
        assistant_start = tk.encode("<｜im▁start｜>assistant\n")
        assert prompt[-len(assistant_start):] == assistant_start


class TestAlpacaStrategy:
    def test_name(self):
        assert AlpacaStrategy(DummyTokenizer()).name == "alpaca"

    def test_assemble_prompt(self):
        tk = DummyTokenizer()
        strategy = AlpacaStrategy(tk)
        query_tokens = tk.encode("hello")
        prompt = strategy.assemble_prompt(query_tokens)
        text = _decode(prompt)
        assert "### Instruction:" in text
        assert "hello" in text
        assert "### Response:" in text

    def test_assemble_response(self):
        tk = DummyTokenizer()
        strategy = AlpacaStrategy(tk)
        response_tokens = tk.encode("world")
        response = strategy.assemble_response(response_tokens)
        text = _decode(response)
        assert "world" in text
        assert "<｜end▁of▁sentence｜>" in text


class TestStrategyFactory:
    def test_create_chatml(self):
        tk = DummyTokenizer()
        assert isinstance(StrategyFactory.create("chatml", tk), ChatMLStrategy)

    def test_create_alpaca(self):
        tk = DummyTokenizer()
        assert isinstance(StrategyFactory.create("alpaca", tk), AlpacaStrategy)

    def test_create_invalid_raises_error(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            StrategyFactory.create("invalid_strategy", DummyTokenizer())

    def test_register_and_create(self):
        StrategyFactory.register("dummy")(DummyStrategy)
        tk = DummyTokenizer()
        strategy = StrategyFactory.create("dummy", tk)
        assert isinstance(strategy, DummyStrategy)
        assert strategy.name == "dummy"

    def test_available_strategies(self):
        strategies = StrategyFactory.available_types()
        assert "chatml" in strategies
        assert "alpaca" in strategies
