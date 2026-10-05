#!/usr/bin/env python3
"""GQA (testdev balanced): statistics and examples. Usage: jetson/datasets/run.sh gqa [--run DIR] [--examples ID ...]"""

from common import Dataset, main

DATASET = Dataset(
    task="gqa",
    title="GQA",
    references=lambda d: [d["answer"]],
    image_id=lambda d: d["imageId"],
    scoring="exact match",
)

if __name__ == "__main__":
    main(DATASET)
