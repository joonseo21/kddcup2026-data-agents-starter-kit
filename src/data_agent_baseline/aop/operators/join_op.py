class JoinOp:
    """Inner join two record lists on matching key fields (case-insensitive key lookup)."""

    def __init__(
        self,
        left: list[dict],
        right: list[dict],
        left_key: str,
        right_key: str,
    ) -> None:
        self.left = left
        self.right = right
        self.left_key = left_key
        self.right_key = right_key

    def execute(self) -> list[dict]:
        def find_value(record: dict, key: str) -> object:
            actual = next((k for k in record if k.lower() == key.lower()), None)
            return record[actual] if actual is not None else None

        # Index right records by their join-key value for O(n) lookup.
        right_index: dict[object, list[dict]] = {}
        for r in self.right:
            k = find_value(r, self.right_key)
            right_index.setdefault(k, []).append(r)

        result: list[dict] = []
        for l in self.left:
            lval = find_value(l, self.left_key)
            for r in right_index.get(lval, []):
                merged = {**l, **r}
                result.append(merged)

        return result
