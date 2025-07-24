"""
File to train encoding semantic-ids and recommendation
"""

import json

from datasets import Dataset

from engines.reward_functions import SEMANTIC_ID_SIZE
from engines.utils import TrainerFactory

SYSTEM_PROMPT = f"""
You are a helpful product recommendation assistant.
Respond in the following format:
<recommend>
...
</recommend>
<explain>
...
</explain>
Between <recommend> and </recommend> are all {SEMANTIC_ID_SIZE} digits ranging from 0 to 256.
"""

with open(
    "data/data/prompts/all_beauty_prompts.jsonl", "r", encoding="utf-8"
) as f:  # noqa
    train_dataset = []
    for line in f:
        try:
            json_object = json.loads(line.strip())
            json_object["messages"][0] = {"role": "system", "content": SYSTEM_PROMPT}
            json_object["messages"][1] = {
                "content": json_object["messages"][1]["content"][0]["text"],
                "role": "user",
            }
            train_dataset.append(
                {
                    "prompt": json_object["messages"][:-1],
                    "answer": json_object["messages"][-1]["content"][0]["text"],
                }
            )

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line: {line.strip()} - {e}")

    train_dataset = Dataset.from_list(train_dataset)

train_fn = TrainerFactory.get("grpo_trainer")

train_fn(
    train_dataset=train_dataset,
    reward_fns=["format_reward", "semantic_reward"],
)  # noqa
