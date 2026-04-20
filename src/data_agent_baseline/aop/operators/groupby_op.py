class GroupByOp:
    """Group records by the value of a field. Missing field values go to '__unknown__'."""

    def __init__(self, records: list[dict], field: str) -> None:
        self.records = records
        self.field = field

    def execute(self) -> dict[str, list[dict]]:
        groups: dict[str, list[dict]] = {}
        for record in self.records:
            actual = next((k for k in record if k.lower() == self.field.lower()), None)
            key = str(record[actual]) if actual is not None else "__unknown__"
            groups.setdefault(key, []).append(record)
        return groups
