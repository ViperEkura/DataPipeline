from datasets import load_dataset
from pipeline import export_dataset

if __name__ == "__main__":
    dataset = load_dataset("Blaze7451/enwiki_structured_content")
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="english-wiki-pretrain",
    )
