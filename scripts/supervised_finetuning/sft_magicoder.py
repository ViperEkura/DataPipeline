from datasets import load_dataset
from pipeline import export_dataset


def process_func(input_dict: dict):
    return {"messages": [
        {"role": "user", "content": input_dict["instruction"]},
        {"role": "assistant", "content": input_dict["response"]},
    ]}


if __name__ == "__main__":
    dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="Magicoder-Evol-Instruct-110K",
        process_func=process_func,
    )
