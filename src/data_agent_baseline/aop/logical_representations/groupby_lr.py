GROUPBY_SUM_LR = {
    "name": "GROUPBY_SUM_LR",
    "description": "Group records by a field and sum another field",
    "example_questions": [
        "Total sales grouped by region",
        "Sum of scores per patient",
    ],
    "operators": ["GroupBy", "Sum"],
    "output_type": "dict[str, float]",
}
