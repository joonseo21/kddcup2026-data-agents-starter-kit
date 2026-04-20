import json
import re
from typing import Callable


class RecordScanOp:
    """
    Filter records using a natural-language condition parsed by an LLM.

    The LLM parses the condition into (field, op, value) on execute(),
    not at construction time, so llm_fn can be injected by callers.
    """

    def __init__(self, records: list[dict], condition: str) -> None:
        self.records = records
        self.condition = condition

    def execute(self, llm_fn: Callable[[str], str]) -> list[dict]:
        # Empty condition means no filtering — return all records as-is.
        if not self.condition:
            return list(self.records)

        prompt = (
            f'Parse the filter condition into JSON.\n'
            f'Condition: "{self.condition}"\n'
            f'Return ONLY: {{"field": "...", "op": "= | != | > | < | >= | <=", "value": "..."}}'
        )
        raw = llm_fn(prompt)

        try:
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            parsed = json.loads(cleaned)
            field = parsed["field"]
            op = parsed["op"]
            value = parsed["value"]
        except (json.JSONDecodeError, KeyError):
            return []

        return [r for r in self.records if self._match(r, field, op, value)]

    def _match(self, record: dict, field: str, op: str, value: str) -> bool:
        record_key = next(
            (k for k in record if k.lower() == field.lower()), None
        )
        if record_key is None:
            return False

        record_val = record[record_key]

        # Try numeric comparison first; fall back to string comparison.
        try:
            rec_num = float(record_val)
            cmp_num = float(value)
            if op == "=":
                return rec_num == cmp_num
            if op == "!=":
                return rec_num != cmp_num
            if op == ">":
                return rec_num > cmp_num
            if op == "<":
                return rec_num < cmp_num
            if op == ">=":
                return rec_num >= cmp_num
            if op == "<=":
                return rec_num <= cmp_num
        except (TypeError, ValueError):
            pass

        rec_str = str(record_val).lower()
        cmp_str = str(value).lower()
        if op == "=":
            return rec_str == cmp_str
        if op == "!=":
            return rec_str != cmp_str
        if op == ">":
            return rec_str > cmp_str
        if op == "<":
            return rec_str < cmp_str
        if op == ">=":
            return rec_str >= cmp_str
        if op == "<=":
            return rec_str <= cmp_str
        return False
