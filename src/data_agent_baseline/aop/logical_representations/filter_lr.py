RECORD_FILTER_LR = {
    "name": "RECORD_FILTER_LR",
    "description": "Filter records from a table where a field equals a value",
    "example_questions": [
        "Records from Examination where Thrombosis equals 2",
        "Filter patients where SEX is F",
    ],
    "operators": ["RecordScan"],
    "output_type": "list[dict]",
}

RECORD_FILTER_NUMERIC_LR = {
    "name": "RECORD_FILTER_NUMERIC_LR",
    "description": "Filter records where a numeric field satisfies a comparison",
    "example_questions": [
        "Records from Patient where age is greater than 70",
    ],
    "operators": ["RecordScan"],
    "output_type": "list[dict]",
}
