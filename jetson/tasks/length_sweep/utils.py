import re


def process_results(doc, results):
    """1 if the answer is the requested count (starts 1, 2, 3 on separate lines); only a sanity check of the prompt."""
    numbers = re.findall(r"\d+", results[0])[:3]
    return {"counts": float(numbers == ["1", "2", "3"])}
