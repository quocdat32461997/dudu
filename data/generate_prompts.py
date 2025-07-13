import pickle
import random
from collections import defaultdict
from datetime import datetime
from functools import partial

from datasets import load_dataset

# def create_conversation(content):
#     # TO-DO: add prompt
#     return {
#         "messages": [
#             {"role": "system"},
#             {"role": "user", "content": content["user_content"]},
#             {"role": "assistant", "content": content["assistant_content"]},
#         ]
#     }


def generate_prompts(
    data_path: str,
    review_data_name: str,
    meta_data_name: str,
    path_to_save_prompt: str,
    sample_count: int = -1,
):
    """
    Using gemma-4b-it, should follow this format: https://huggingface.co/unsloth/gemma-3-4b-it # noqa
    """

    # Extract raw data
    review_partition = defaultdict(partial(defaultdict, list))
    metadata_partition = defaultdict(partial(defaultdict, list))

    reviews = load_dataset(
        data_path,  # "McAuley-Lab/Amazon-Reviews-2023",
        review_data_name,  # "5core_timestamp_Automotive",
        trust_remote_code=True,
    )
    metadata = load_dataset(
        data_path,  # "McAuley-Lab/Amazon-Reviews-2023",
        meta_data_name,  # "raw_meta_All_Beauty",
        split="full",
        trust_remote_code=True,
    )

    # Parse meta data
    for product_id, title, features, description, brand, categories in zip(
        metadata["parent_asin"],
        metadata["title"],
        metadata["description"],
        metadata["brand"],
        metadata["categories"],
    ):
        metadata_partition["product_id"]["title"] = title
        metadata_partition["product_id"]["features"] = features
        metadata_partition["product_id"]["title"] = title
        metadata_partition["product_id"]["description"] = description
        metadata_partition["product_id"]["brand"] = brand
        metadata_partition["categories"]["categories"] = categories

    prompts = []
    for user_id, product_id, history, timestamp in zip(
        reviews["user_id"][:sample_count],
        reviews["product_id"][:sample_count],
        reviews["history"][:sample_count],
        reviews["timestamp"][:sample_count],
    ):
        review_partition[user_id]["product_id"].append(product_id)
        review_partition[user_id]["history"].append(history)
        review_partition[user_id]["timestamp"].append(
            datetime.fromtimestamp(int(timestamp) / 1000)
        )

    # Save to pickle file
    with open(path_to_save_prompt, "wb") as file:
        pickle.dump(prompts, file)
