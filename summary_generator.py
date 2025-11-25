# summary_generator.py
def generate_summary(boxes):
    count = len(boxes)
    if count == 0:
        return "No obvious damage detected. Surface appears normal."
    return f"{count} potential damage spot(s) detected. Recommend manual inspection for confirmation."
