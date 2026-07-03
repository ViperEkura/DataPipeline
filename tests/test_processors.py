"""Tests for pipeline.processors module."""

import pytest
import torch

from pipeline.processors import (
    BaseProcessor,
    PreTrainProcessor,
    SFTProcessor,
    DPOProcessor,
    ProcessorFactory,
)


class DummyTokenizer:
    im_end = "<|im_end|>"

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(c) for c in text]

    def token_to_id(self, token: str):
        return ord(token)

    def apply_chat_template(
        self, messages, add_generation_prompt=True, tokenize=True
    ):
        text = ""
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text) if tokenize else text


class TestBaseProcessor:
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseProcessor()


class TestPreTrainProcessor:
    def test_output_keys(self):
        assert PreTrainProcessor(DummyTokenizer()).output_keys == ["sequence"]

    def test_process_returns_tensor(self):
        processor = PreTrainProcessor(DummyTokenizer())
        result = processor.process({"text": "hello world"})
        assert "sequence" in result
        assert isinstance(result["sequence"], torch.Tensor)
        assert result["sequence"].dtype == torch.int32

    def test_process_adds_eos(self):
        result = PreTrainProcessor(DummyTokenizer()).process({"text": "a"})
        assert len(result["sequence"]) > 0


class TestSFTProcessor:
    def test_output_keys(self):
        keys = SFTProcessor(DummyTokenizer()).output_keys
        assert "sequence" in keys
        assert "loss_mask" in keys
        assert "position_ids" in keys

    def test_process_returns_all_keys(self):
        result = SFTProcessor(DummyTokenizer()).process(
            {"query": "hello", "response": "world"}
        )
        for key in ["sequence", "loss_mask", "position_ids"]:
            assert key in result
            assert isinstance(result[key], torch.Tensor)

    def test_loss_mask_correct_length(self):
        result = SFTProcessor(DummyTokenizer()).process(
            {"query": "hi", "response": "bye"}
        )
        assert len(result["sequence"]) == len(result["loss_mask"])

    def test_loss_mask_is_bool(self):
        result = SFTProcessor(DummyTokenizer()).process(
            {"query": "ab", "response": "cd"}
        )
        assert result["loss_mask"].dtype == torch.bool

    def test_position_ids_start_from_zero(self):
        result = SFTProcessor(DummyTokenizer()).process(
            {"query": "abc", "response": "de"}
        )
        seq_len = len(result["sequence"])
        expected = torch.arange(seq_len, dtype=torch.int32)
        assert torch.equal(result["position_ids"], expected)

    def test_messages_single_turn(self):
        result = SFTProcessor(DummyTokenizer()).process({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "bye"},
            ]
        })
        for key in ["sequence", "loss_mask", "position_ids"]:
            assert key in result
        assert len(result["sequence"]) == len(result["loss_mask"])

    def test_messages_loss_on_last_assistant_only(self):
        result = SFTProcessor(DummyTokenizer()).process({
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ]
        })
        mask = result["loss_mask"]
        first_true = mask.tolist().index(True)
        assert not mask[:first_true].any()
        assert mask[-1].item() is True

    def test_messages_with_system_prompt(self):
        result = SFTProcessor(DummyTokenizer()).process({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        })
        assert "sequence" in result

    def test_messages_empty_raises(self):
        with pytest.raises(ValueError, match="Messages list is empty"):
            SFTProcessor(DummyTokenizer()).process({"messages": []})

    def test_messages_last_not_assistant_raises(self):
        with pytest.raises(ValueError, match="Last message must"):
            SFTProcessor(DummyTokenizer()).process({
                "messages": [{"role": "user", "content": "hi"}]
            })

    def test_missing_fields_raises(self):
        with pytest.raises(KeyError):
            SFTProcessor(DummyTokenizer()).process({"foo": "bar"})

    def test_position_ids_start_from_zero(self):
        result = SFTProcessor(DummyTokenizer()).process(
            {"query": "hi", "response": "ok"}
        )
        pos_ids = result["position_ids"]
        assert pos_ids.dtype == torch.int32
        assert len(pos_ids) == len(result["sequence"])
        assert pos_ids[0].item() == 0
        assert (pos_ids == torch.arange(len(pos_ids))).all()


class TestDPOProcessor:
    def test_output_keys(self):
        keys = DPOProcessor(DummyTokenizer()).output_keys
        assert keys == ["chosen", "chosen_mask", "rejected", "rejected_mask"]

    def test_process_returns_all_keys(self):
        result = DPOProcessor(DummyTokenizer()).process(
            {"query": "hello", "chosen": "r1", "rejected": "r2"}
        )
        for key in ["chosen", "chosen_mask", "rejected", "rejected_mask"]:
            assert key in result
            assert isinstance(result[key], torch.Tensor)

    def test_masks_match_lengths(self):
        result = DPOProcessor(DummyTokenizer()).process(
            {"query": "test", "chosen": "yes", "rejected": "no"}
        )
        assert len(result["chosen"]) == len(result["chosen_mask"])
        assert len(result["rejected"]) == len(result["rejected_mask"])

    def test_masks_are_bool(self):
        result = DPOProcessor(DummyTokenizer()).process(
            {"query": "test", "chosen": "yes", "rejected": "no"}
        )
        assert result["chosen_mask"].dtype == torch.bool
        assert result["rejected_mask"].dtype == torch.bool


class TestProcessorFactory:
    def test_create_pre_train_processor(self):
        assert isinstance(
            ProcessorFactory.create("pt", DummyTokenizer()), PreTrainProcessor
        )

    def test_create_sft_processor(self):
        assert isinstance(
            ProcessorFactory.create("sft", DummyTokenizer()), SFTProcessor
        )

    def test_create_dpo_processor(self):
        assert isinstance(
            ProcessorFactory.create("dpo", DummyTokenizer()), DPOProcessor
        )

    def test_create_invalid_processor_raises_error(self):
        with pytest.raises(ValueError, match="Unknown processor type"):
            ProcessorFactory.create("invalid", DummyTokenizer())

    def test_register_and_create_custom_processor(self):
        class CustomProcessor(BaseProcessor):
            def __init__(self, tokenizer=None):
                self._tokenizer = tokenizer

            @property
            def output_keys(self):
                return ["custom"]

            def process(self, input_dict):
                return {"custom": torch.tensor([1, 2, 3])}

        ProcessorFactory.register("custom")(CustomProcessor)
        assert isinstance(
            ProcessorFactory.create("custom", DummyTokenizer()), CustomProcessor
        )
