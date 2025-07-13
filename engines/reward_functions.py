import re
from typing import List

from engines.utils import RewardFunctionFactory

SEMANTIC_ID_SIZE = 6
ADDITIONAL_SEMANTIC_PATTERN = "<recommend>\d{0," + str(SEMANTIC_ID_SIZE) + "}"
SEMANTIC_PATTERN = f"({ADDITIONAL_SEMANTIC_PATTERN}</recommend>)"
SEMANTIC_SERIES_PATTERN = f"{ADDITIONAL_SEMANTIC_PATTERN}+" + "</recommend>"
EXPLAIN_PATTERN = "<explain>.*</explain>"


def extract_answer(text: str, start_text, end_text) -> str:
    answer = text.split(start_text)[-1]
    answer = answer.split(end_text)[0]
    return answer.strip()


@RewardFunctionFactory.register("format_reward")
def explain_format_reward(completions: List[str], tasks: List[str], **kwargs):
    rewards = []
    TASK = "format_reward"
    FORMAT_PATTERN = f"{SEMANTIC_PATTERN}\n{EXPLAIN_PATTERN}"
    for completion, task in zip(completions, tasks):
        if task != TASK:
            rewards.append(None)

        match = re.match(FORMAT_PATTERN, completion[0]["content"])
        rewards.append(1 if match else 0)

    return rewards


@RewardFunctionFactory.register("semantic_reward")
def semantic_reward(completions: List[str], **kwarsg):  # tasks: List[str],
    rewards = []
    # TASK = "semantic_reward"

    # Exact recommend with all digits and desited size
    # for completion, task in zip(completions, tasks):
    for completion in completions:
        # if task != TASK:
        #     rewards.append(None)

        match = re.match(SEMANTIC_PATTERN, completion[0]["content"])  # noqa

        if match:
            match = extract_answer(match.group(), "<recommend>", "</recommend>")  # noqa
            digit_count = sum([1 if x.isdigit() else -1 for x in match])  # noqa
            rewards.append(
                digit_count
                if digit_count <= SEMANTIC_ID_SIZE
                else (digit_count - SEMANTIC_ID_SIZE)
            )  # noqa
        else:
            rewards.append(0)

    return rewards


@RewardFunctionFactory.register("explain_reward")
def explain_reward(
    completions: List[str],
    labels: List[str],
    tasks: List[str],
    **kwargs,
):
    TASK = "explain_reward"
    rewards = []
    for completion, task, label in zip(completions, tasks, labels):
        if task != TASK:
            rewards.append(None)

        # Please explain futher why pick the next product.
        match = re.match(EXPLAIN_PATTERN, completion[0]["content"])
        if match:
            rewards.append(1 if match.group() == label else 0)
        else:
            0

    return rewards


@RewardFunctionFactory.register("similar_product")
def similar_product(completionts, expected_products, **kwargs):
    pass
