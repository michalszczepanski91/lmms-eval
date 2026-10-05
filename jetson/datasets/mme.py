#!/usr/bin/env python3
"""MME: statistics and examples. Usage: jetson/datasets/run.sh mme [--run DIR] [--examples ID ...]"""

from common import Dataset, main

DATASET = Dataset(
    task="mme",
    title="MME",
    references=lambda d: [d["answer"]],  # "Yes" or "No"
    image_id=lambda d: d["question_id"],  # image file, e.g. code_reasoning/0020.png (2 questions each)
    scoring='exact match on "Yes"/"No" (the answer\'s first word); perception and cognition scores sum accuracy and accuracy+ per category',
)

if __name__ == "__main__":
    main(DATASET)
