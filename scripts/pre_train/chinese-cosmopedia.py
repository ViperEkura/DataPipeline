from datasets import load_dataset
from pipeline import export_dataset

if __name__ == "__main__":
    dataset = load_dataset(
        "opencsg/chinese-cosmopedia",
    )
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="chinese-cosmopedia-pretrain",
        chunk_size=1_000_000,
    )
