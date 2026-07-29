from datasets import load_dataset
from pipeline import export_dataset


def process_func(input_dict: dict):
    instruction = input_dict["instruction"]
    inp = input_dict.get("input", "")
    if inp:
        content = instruction + "\n" + inp
    else:
        content = instruction
    return {"messages": [
        {"role": "user", "content": content},
        {"role": "assistant", "content": input_dict["output"]},
    ]}


if __name__ == "__main__":
    dataset = load_dataset("llm-wizard/alpaca-gpt4-data-zh")
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="alpaca-gpt4-data-zh",
        process_func=process_func,
    )
