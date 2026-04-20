COUNT_LR = {
    "name": "COUNT_LR",
    "description": "Count the total number of records",
    "example_questions": [
        "How many records are there?",
        "Count all patients",
    ],
    "operators": ["Count"],
    "output_type": "int",
}

COUNT_DISTINCT_LR = {
    "name": "COUNT_DISTINCT_LR",
    "description": "Count distinct values of a field across records",
    "example_questions": [
        "How many unique patient IDs are there?",
        "Count distinct diagnoses",
    ],
    "operators": ["Count"],
    "output_type": "int",
}
