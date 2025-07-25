"""
File to train encoding semantic-ids and recommendation
"""

import json

from datasets import Dataset

from dudu.utils import TrainerFactory

with open(
    "data/prompts/all_beauty_recommend_prompts.jsonl",
    "r",
    encoding="utf-8",
) as f:  # noqa
    train_dataset = []
    for line in f:
        try:
            json_object = json.loads(line.strip())
            train_dataset.append(json_object)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {line.strip()} - {e}")

    train_dataset = Dataset.from_list(train_dataset)

train_fn = TrainerFactory.get("grpo_trainer")

train_fn(
    train_dataset=train_dataset,
    reward_fns=["format_reward", "semantic_reward", "next_product_reward"],
)  # noqa
