import pickle
import random
from collections import defaultdict
from datetime import datetime
from functools import partial

from datasets import load_dataset

"""
Look at this https://huggingface.co/docs/trl/grpo_trainer#example-1-reward-longer-completions for better understanding.
history: A, B, C, D
text: A_product_name, B_product_name, C_product_name

prompt_1: The customer purchased A_product_name, B_product_name, C_product_name in the timely order. A_product_name belongs to category Beauty and has following features or categories. 
label: <explain>The customer will purchase F_product_name that belongs to category or brand. The product has following features</explain>

prompt_2: The customer purchased A_product_name, B_product_name, C_product_name in the timely order. A_product_name belongs to category Beauty and has following features or categories. 
label: <explain>The customer will purchase F_product_name that belongs to category or brand. The product has following features</explain>
"""
def create_conversation(content):
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": content["system_content"]}]},
            {"role": "user", "content": [{"type": "text", "text": content["user_content"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": content["assistant_content"]}]},
        ]
    }


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
        review_data_name,  # "5core_timestamp_Automotive", replace Automative w/ All_Beauty
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

    # Generate quantization prompts for product features and categories
    for product_id, data in metadata_partition.items():
        if product_id != "product_id" and product_id != "categories":
            title = data.get("title", "")
            features = data.get("features", "")
            brand = data.get("brand", "")
            categories = data.get("categories", [])
            
            if title and categories:
                category_text = ", ".join(categories) if isinstance(categories, list) else str(categories)
                features_text = features if features else "NA"
                
                system_content = "You are a helpful product recommendation assistant."
                user_content = f"Analyze this product: {title}. Brand: {brand}. Categories: {category_text}. Features: {features_text}"
                assistant_content = f"<explain>This product belongs to {category_text} category/brand. Key features include: {features_text}</explain>"
                
                prompt_data = {
                    "system_content": system_content,
                    "user_content": user_content,
                    "assistant_content": assistant_content
                }
                prompts.append(create_conversation(prompt_data))

    # Parse transaction history
    prompts = []
    for user_id, product_id, history, timestamp in zip(
        reviews["user_id"][:sample_count],
        reviews["product_id"][:sample_count],
        reviews["history"][:sample_count],
        reviews["timestamp"][:sample_count],
    ):
        review_partition[user_id]["product_id"].append(product_id)  # next product
        review_partition[user_id]["history"].append(history)  # past purchase history
        review_partition[user_id]["timestamp"].append(
            datetime.fromtimestamp(int(timestamp) / 1000)
        )

    # Generate transaction-related prompts based on purchase history
    for user_id, user_data in review_partition.items():
        product_ids = user_data["product_id"]
        
        if len(product_ids) >= 2:  # Need at least 2 products for history
            # Get product details from metadata
            purchased_products = []
            next_product_id = product_ids[-1]  # Last product as prediction target
            history_products = product_ids[:-1]  # Previous products as history
            
            # Build purchase history text
            for hist_id in history_products[-3:]:  # Use last 3 products as history
                if hist_id in metadata_partition:
                    product_title = metadata_partition[hist_id].get("title", f"Product_{hist_id}")
                    purchased_products.append(product_title)
            
            if purchased_products and next_product_id in metadata_partition:
                next_product = metadata_partition[next_product_id]
                next_title = next_product.get("title", f"Product_{next_product_id}")
                next_categories = next_product.get("categories", [])
                next_features = next_product.get("features", "")
                
                history_text = ", ".join(purchased_products)
                category_text = ", ".join(next_categories) if isinstance(next_categories, list) else str(next_categories)
                features_text = next_features if next_features else "NA"
                
                system_content = "You are a helpful product recommendation assistant that predicts customer purchases based on purchase history."
                user_content = f"The customer purchased {history_text} in timely order. What product will they likely purchase next?"
                assistant_content = f"<explain>The customer will purchase {next_title} that belongs to {category_text} category or brand. The product has following features: {features_text}</explain>"
                
                prompt_data = {
                    "system_content": system_content,
                    "user_content": user_content,
                    "assistant_content": assistant_content
                }
                prompts.append(create_conversation(prompt_data))

    # Save to pickle file
    with open(path_to_save_prompt, "wb") as file:
        pickle.dump(prompts, file)
