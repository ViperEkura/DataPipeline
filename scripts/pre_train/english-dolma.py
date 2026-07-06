from datasets import load_dataset
from pipeline import export_dataset

if __name__ == "__main__":
    dataset = load_dataset("emozilla/dolma-v1_7-30B")
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="english-dolma-30b-pretrain",
        max_chunks=18,
    )
