class ExtractOp:
    """Project a list of records down to specified columns (case-insensitive name matching)."""

    def __init__(self, records: list[dict], columns: list[str]) -> None:
        self.records = records
        self.columns = columns

    def execute(self) -> list[dict]:
        result = []
        for record in self.records:
            row: dict = {}
            for col in self.columns:
                actual = next((k for k in record if k.lower() == col.lower()), None)
                row[col] = record[actual] if actual is not None else None
            result.append(row)
        return result
