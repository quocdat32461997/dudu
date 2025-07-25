import json
import os
import random
from collections import defaultdict
from datetime import datetime
from functools import partial

from dudu.prompts import _SEMANTIC_ID_SAMPLE, SEMANTIC_ID_SIZE, SYSTEM_PROMPT

random.seed(22)


def create_conversation(content):
    return {
        "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    semantic_id_size=SEMANTIC_ID_SIZE,
                    semantic_id_sample=_SEMANTIC_ID_SAMPLE,
                ),
            },
            {
                "role": "user",
                "content": content["user_content"],
            },
        ],
        "answer": content["assistant_content"],
    }


def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def generate_prompts(
    review_file: str = "data/source/All_Beauty.jsonl",
    meta_file: str = "data/source/meta_All_Beauty.jsonl",
    path_to_save_prompt: str = "data/prompts/all_beauty_prompts.jsonl",
    sample_count: int = -1,
):
    """
    Load Amazon Reviews 2023 dataset from local JSONL files
    """

    # Create prompts directory if it doesn't exist
    os.makedirs(os.path.dirname(path_to_save_prompt), exist_ok=True)

    # Initialize prompts list
    analysis_prompts = []

    # Extract raw data
    review_partition = defaultdict(partial(defaultdict, list))
    metadata_partition = {}

    print("Loading reviews dataset from local JSONL file...")
    reviews_data = load_jsonl(review_file)

    print("Loading metadata dataset from local JSONL file...")
    metadata_data = load_jsonl(meta_file)

    print("Processing metadata...")
    # Parse metadata from JSONL
    for i, item in enumerate(metadata_data):
        product_id = item.get("parent_asin")
        if product_id:
            metadata_partition[product_id] = {
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "brand": item.get("brand", ""),
                "categories": item.get("categories", []),
            }

        if i % 10000 == 0:
            print(f"Processed {i} metadata records...")

    print(f"Total metadata records processed: {len(metadata_partition)}")

    # Generate quantization prompts for product features and categories
    path_to_save = "data/prompts/all_beauty_analysis_prompts.jsonl"
    print("Generating product analysis prompts...")
    for product_id, data in metadata_partition.items():
        title = data.get("title", "")
        description = data.get("description", "")
        brand = data.get("brand", "")
        categories = data.get("categories", [])
        if title:
            category_text = (
                ", ".join(random.sample(categories, 2))
                if isinstance(categories, list) and len(categories) > 1
                else str(categories)
            )
            description_text = description if description else "NA"
            brand_text = brand if brand else "Unknown"

            # system_content = "You are a helpful product recommendation assistant."
            user_content = f"Analyze this product: {title}. Brand: {brand_text}. Categories: {category_text}. Description: {description_text}"  # noqa
            assistant_content = f"This product belongs to {category_text} category/brand. Key features include: {description_text}"  # noqa
            prompt_data = {
                # "system_content": system_content,
                "user_content": user_content,
                "assistant_content": assistant_content,
            }
            analysis_prompts.append(create_conversation(prompt_data))

    print(f"Generated {len(analysis_prompts)} product analysis prompts")
    # Save to JSONL file

    print(f"Saving prompts to {path_to_save}...")
    with open(path_to_save, "w", encoding="utf-8") as file:
        for prompt in analysis_prompts:
            file.write(json.dumps(prompt) + "\n")

    # Parse transaction history
    print("Processing transaction history...")

    # Initialize prompts list
    recommend_prompts = []

    # Apply sample count if specified
    if sample_count != -1:
        reviews_data = reviews_data[:sample_count]

    total_reviews = len(reviews_data)
    print(f"Total review records: {total_reviews}")

    for i, item in enumerate(reviews_data):
        user_id = item.get("user_id")
        product_id = item.get("parent_asin")
        timestamp = item.get("timestamp")

        if user_id and product_id and timestamp:
            review_partition[user_id]["product_id"].append(product_id)
            review_partition[user_id]["timestamp"].append(
                datetime.fromtimestamp(int(timestamp) / 1000)
            )

        if i % 10000 == 0:
            print(f"Processed {i} review records...")

    print(f"Total users with purchase history: {len(review_partition)}")

    # Generate transaction-related prompts based on purchase history
    print("Generating purchase prediction prompts...")
    path_to_save = "data/prompts/all_beauty_recommend_prompts.jsonl"
    prediction_prompts = 0

    # Group purchases by user and sort by timestamp to create purchase sequences
    for user_id, user_data in review_partition.items():
        product_ids = user_data["product_id"]
        timestamps = user_data["timestamp"]

        if len(product_ids) >= 2:  # Need at least 2 products for history
            # Sort by timestamp to get chronological order
            sorted_items = sorted(
                zip(product_ids, timestamps), key=lambda x: x[1]
            )  # noqa
            sorted_product_ids = [item[0] for item in sorted_items]

            # Get product details from metadata
            purchased_products = []
            next_product_id = sorted_product_ids[
                -1
            ]  # Last product as prediction target
            history_products = sorted_product_ids[:-1]  # Previous products as history

            # Build purchase history text
            for hist_id in history_products[-3:]:  # Use last 3 products as history
                if hist_id in metadata_partition:
                    product_title = metadata_partition[hist_id].get(
                        "title", f"Product_{hist_id}"
                    )
                    purchased_products.append(product_title)

            if purchased_products and next_product_id in metadata_partition:
                next_product = metadata_partition[next_product_id]
                next_title = next_product.get(
                    "title",
                    f"Product_{next_product_id}",
                )
                next_categories = next_product.get("categories", [])
                next_description = next_product.get("description", "")

                history_text = ", ".join(purchased_products)
                category_text = (
                    ", ".join(random.sample(categories, 2))
                    if isinstance(next_categories, list) and len(categories) > 1
                    else str(next_categories)
                )
                description_text = (
                    next_description if next_description else "NA"
                )  # noqa

                # system_content = "You are a helpful product recommendation assistant that predicts customer purchases based on purchase history."
                user_content = f"The customer purchased {history_text} in timely order. What product will they likely purchase next?"
                assistant_content = f"The customer will purchase {next_title}."  # that belongs to {category_text} category or brand. The product has following features: {description_text}"

                prompt_data = {
                    # "system_content": system_content,
                    "user_content": user_content,
                    "assistant_content": assistant_content,
                }
                recommend_prompts.append(create_conversation(prompt_data))
                prediction_prompts += 1

    print(f"Generated {len(recommend_prompts)} purchase prediction prompts")
    print(f"Total prompts generated: {len(recommend_prompts)}")

    # Save to JSONL file
    print(f"Saving prompts to {path_to_save}...")
    with open(path_to_save, "w", encoding="utf-8") as file:
        for prompt in recommend_prompts:
            file.write(json.dumps(prompt) + "\n")

    print(
        f"Successfully generated {len(recommend_prompts)} prompts and saved to {path_to_save_prompt}"  # noqa
    )

    # Save comprehensive list
    print(
        f"Generated {len(recommend_prompts) + len(analysis_prompts)} purchase prompts"  # noqas
    )
    print(
        f"Total prompts generated: {len(recommend_prompts) + len(analysis_prompts)}"  # noqa
    )  # noqa

    # Save to JSONL file
    print(f"Saving prompts to {path_to_save_prompt}...")
    with open(path_to_save_prompt, "w", encoding="utf-8") as file:
        for prompt in recommend_prompts + analysis_prompts:
            file.write(json.dumps(prompt) + "\n")

    print(
        f"Successfully generated {len(recommend_prompts) + len(analysis_prompts)} prompts and saved to {path_to_save_prompt}"
    )


if __name__ == "__main__":
    # Test with smaller sample first
    generate_prompts()
