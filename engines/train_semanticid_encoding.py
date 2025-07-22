import json

from datasets import Dataset

from engines.utils import TrainerFactory

# Load data
with open("data/data/prompts/all_beauty_prompts.jsonl", "rb") as file:
    train_dataset = json.loads(file)
    train_dataset = Dataset.from_list(train_dataset)

train_fn = TrainerFactory.get("grpo_trainer")
train_fn(
    train_dataset=train_dataset, reward_fns=["format_reward", "semantic_reward"]
)  # noqa
