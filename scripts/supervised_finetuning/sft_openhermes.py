from datasets import load_dataset
from pipeline import export_dataset


ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}


def process_func(input_dict: dict):
    conversations = input_dict["conversations"]

    system_msgs = []
    idx = 0
    if conversations and conversations[0]["from"] == "system":
        system_msgs.append({
            "role": "system",
            "content": conversations[0]["value"],
        })
        idx = 1

    examples = []
    for i in range(idx, len(conversations) - 1, 2):
        user_msg = conversations[i]
        assistant_msg = conversations[i + 1]
        messages = system_msgs + [
            {"role": ROLE_MAP[user_msg["from"]], "content": user_msg["value"]},
            {"role": ROLE_MAP[assistant_msg["from"]], "content": assistant_msg["value"]},
        ]
        examples.append({"messages": messages})
    return examples


if __name__ == "__main__":
    dataset = load_dataset("teknium/OpenHermes-2.5")
    export_dataset(
        dataset=dataset["train"],
        output_dir="./dataset",
        output_prefix="OpenHermes-2.5",
        process_func=process_func,
    )
