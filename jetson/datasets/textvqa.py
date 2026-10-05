#!/usr/bin/env python3
"""TextVQA (validation): statistics and examples. Usage: jetson/datasets/run.sh textvqa [--run DIR] [--examples ID ...]"""

from collections import Counter

from common import Dataset, main

DATASET = Dataset(
    task="textvqa_val",
    title="TextVQA",
    references=lambda d: d["answers"],  # 10 human answers
    reference=lambda d: Counter(d["answers"]).most_common(1)[0][0],  # the most common one (ties: first listed)
    image_id=lambda d: d["image_id"],
    scoring="VQA accuracy: min(1, human answers matching the model's / 3), averaged over the 10 leave-one-out subsets",
)

if __name__ == "__main__":
    main(DATASET)
