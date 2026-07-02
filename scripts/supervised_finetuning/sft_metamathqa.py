from datasets import load_dataset
from pipeline import export_dataset


def process_func(sample: dict) -> dict:
    return {"query": sample["query"], "response": sample["response"]}


if __name__ == "__main__":
    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    export_dataset(
        dataset=dataset,
        output_dir="./dataset",
        output_prefix="MetaMathQA",
        process_func=process_func,
        chunk_size=1_000_000,
    )
