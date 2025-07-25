import random

from dudu.reward_functions import SEMANTIC_ID_SIZE

_SEMANTIC_ID_SAMPLE = ",".join(
    [str(random.randint(0, 256)) for _ in range(SEMANTIC_ID_SIZE)]
)
SYSTEM_PROMPT = """
Respond in the following format:
<recommend>
...
</recommend>
<explain>
...
</explain>
Between <recommend> and </recommend> are all {semantic_id_size} digits ranging from 0 to 256 and separated by ','.
For example, <recommend>{semantic_id_sample}</recommend>.
"""
