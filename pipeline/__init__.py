"""DataPipeline: A flexible data processing pipeline for LLM training.

Architecture:
    - Pipeline: Composable stage-based processing
    - Processors: Data transformation (pretrain, sft, dpo)
    - Strategies: Prompt format abstraction (ChatML, Alpaca)
    - I/O: File scanning and HDF5 storage

Usage::

    from pipeline import Pipeline, ProcessorFactory, FileScanner, HDF5Handler
    from pipeline.pipeline import TransformStage
    from pipeline.io import export_dataset, cache_jsonl

    # Create pipeline
    pipeline = Pipeline()
    pipeline.add_stages(
        TransformStage("normalize", normalizer.normalize),
        TransformStage("tokenize", tokenizer.encode),
    )

    # Process data
    results = pipeline.run(texts)
    HDF5Handler.save("./output", "data", {"tokens": results})
"""

# Core modules
from pipeline.pipeline import Pipeline, PipelineConfig, Stage, TransformStage
from pipeline.tokenize import AutoTokenizer, ChatTemplate, train_bpe_tokenizer
from pipeline.text import TextNormalizer
from pipeline.packing import (
    GreedyPacker,
    FfDPacker,
    BfdPacker,
    BasePacker,
    pack_tensors,
)

# I/O module
from pipeline.io import FileScanner, HDF5Handler, export_dataset, cache_jsonl

# Processors
from pipeline.processors import (
    ProcessorFactory,
    BaseProcessor,
    ProcessorSchema,
    ProcessorConfig,
    PreTrainProcessor,
    SFTProcessor,
    DPOProcessor,
)

# Strategies
from pipeline.strategies import (
    PromptStrategy,
    ChatMLStrategy,
    AlpacaStrategy,
    StrategyFactory,
)

# Utilities (lazy initialization)
from pipeline import utils

# Expose setup_logging for explicit use
setup_logging = utils.setup_logging

__all__ = [
    # Pipeline
    "Pipeline",
    "PipelineConfig",
    "Stage",
    "TransformStage",
    # Tokenizer
    "AutoTokenizer",
    "ChatTemplate",
    "train_bpe_tokenizer",
    # Text processing
    "TextNormalizer",
    "GreedyPacker",
    "FfDPacker",
    "BfdPacker",
    "BasePacker",
    "pack_tensors",
    # I/O
    "FileScanner",
    "HDF5Handler",
    "export_dataset",
    "cache_jsonl",
    # Processors
    "ProcessorFactory",
    "BaseProcessor",
    "ProcessorSchema",
    "ProcessorConfig",
    "PreTrainProcessor",
    "SFTProcessor",
    "DPOProcessor",
    # Strategies
    "PromptStrategy",
    "ChatMLStrategy",
    "AlpacaStrategy",
    "StrategyFactory",
    # Utils
    "setup_logging",
]
