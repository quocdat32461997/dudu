from collections import defaultdict
from datetime import datetime
from functools import partial

from datasets import load_dataset


def preprocess(
    data_path: str,
    review_data_name: str,
    meta_data_name: str,
    sample_count: int = -1,
):
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
