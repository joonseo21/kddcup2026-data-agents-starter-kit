class SumOp:
    """Sum a numeric field across each group. Non-numeric values are treated as 0."""

    def __init__(self, groups: dict[str, list[dict]], field: str) -> None:
        self.groups = groups
        self.field = field

    def execute(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for group_key, records in self.groups.items():
            total = 0.0
            for record in records:
                actual = next((k for k in record if k.lower() == self.field.lower()), None)
                if actual is not None:
                    try:
                        total += float(record[actual])
                    except (TypeError, ValueError):
                        pass
            result[group_key] = total
        return result
