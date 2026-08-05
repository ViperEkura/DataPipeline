from datasets import load_dataset
from pipeline import export_dataset


def process_func(input_dict: dict):
    return {"text": input_dict["content"]}


if __name__ == "__main__":
    dataset = load_dataset(
        "openbmb/Ultra-FineWeb-L3",
        "Ultra-FineWeb-L3-en-QA-Synthetic",
        split="train",
    )
    export_dataset(
        dataset=dataset,
        output_dir="./dataset",
        output_prefix="ultra-fineweb-l3-en-qa-synthetic-pretrain",
        process_func=process_func,
    )
