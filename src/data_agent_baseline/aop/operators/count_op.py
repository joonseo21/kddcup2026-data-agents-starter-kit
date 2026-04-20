class CountOp:
    """Count records, or count distinct values of a field when field is given."""

    def __init__(self, records: list[dict], field: str | None = None) -> None:
        self.records = records
        self.field = field

    def execute(self) -> int:
        if self.field is None:
            return len(self.records)

        seen: set = set()
        for record in self.records:
            actual = next((k for k in record if k.lower() == self.field.lower()), None)
            if actual is not None:
                seen.add(record[actual])
        return len(seen)
