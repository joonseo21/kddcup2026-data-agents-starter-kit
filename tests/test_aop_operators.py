"""
AOP operator unit tests.

Each operator is tested independently. RecordScanOp injects a mock llm_fn
so no real LLM API calls are made.
"""

import json

import pytest

from data_agent_baseline.aop.operators.record_scan import RecordScanOp
from data_agent_baseline.aop.operators.join_op import JoinOp
from data_agent_baseline.aop.operators.extract_op import ExtractOp
from data_agent_baseline.aop.operators.count_op import CountOp
from data_agent_baseline.aop.operators.groupby_op import GroupByOp
from data_agent_baseline.aop.operators.sum_op import SumOp


class TestRecordScanOp:
    @pytest.fixture
    def make_llm(self):
        """Return a factory that creates a mock llm_fn returning fixed field/op/value."""
        def _make(field, op, value):
            def llm_fn(prompt):
                return json.dumps({"field": field, "op": op, "value": value})
            return llm_fn
        return _make

    @pytest.fixture
    def sample_records(self):
        return [
            {"ID": "1", "SEX": "F", "Thrombosis": "2"},
            {"ID": "2", "SEX": "M", "Thrombosis": "0"},
            {"ID": "3", "SEX": "F", "Thrombosis": "2"},
        ]

    def test_equality_filter_string(self, make_llm, sample_records):
        op = RecordScanOp(sample_records, "SEX is F")
        result = op.execute(make_llm("SEX", "=", "F"))
        assert len(result) == 2
        assert all(r["SEX"] == "F" for r in result)

    def test_equality_filter_numeric(self, make_llm, sample_records):
        op = RecordScanOp(sample_records, "Thrombosis equals 2")
        result = op.execute(make_llm("Thrombosis", "=", "2"))
        assert len(result) == 2

    def test_greater_than_filter(self, make_llm):
        records = [{"age": "65"}, {"age": "72"}, {"age": "80"}]
        op = RecordScanOp(records, "age greater than 70")
        result = op.execute(make_llm("age", ">", "70"))
        assert len(result) == 2
        assert all(int(r["age"]) > 70 for r in result)

    def test_field_name_case_insensitive(self, make_llm):
        records = [{"thrombosis": "2"}, {"thrombosis": "0"}]
        op = RecordScanOp(records, "Thrombosis equals 2")
        # LLM uses mixed-case field name; _match should still find it.
        result = op.execute(make_llm("Thrombosis", "=", "2"))
        assert len(result) == 1

    def test_no_matching_records_returns_empty(self, make_llm, sample_records):
        op = RecordScanOp(sample_records, "SEX is X")
        result = op.execute(make_llm("SEX", "=", "X"))
        assert result == []

    def test_invalid_llm_response_returns_empty(self):
        records = [{"ID": "1"}]
        op = RecordScanOp(records, "anything")
        result = op.execute(lambda prompt: "NOT VALID JSON {{{{")
        assert result == []


class TestJoinOp:
    def test_basic_inner_join(self):
        left = [{"ID": "1", "name": "Alice"}, {"ID": "2", "name": "Bob"}]
        right = [{"ID": "1", "score": "90"}, {"ID": "3", "score": "70"}]
        result = JoinOp(left, right, "ID", "ID").execute()
        assert len(result) == 1
        assert result[0]["name"] == "Alice"
        assert result[0]["score"] == "90"

    def test_no_matching_keys_returns_empty(self):
        left = [{"ID": "1"}]
        right = [{"ID": "2"}]
        assert JoinOp(left, right, "ID", "ID").execute() == []

    def test_join_key_case_insensitive(self):
        left = [{"id": "1", "val": "a"}]
        right = [{"ID": "1", "extra": "b"}]
        result = JoinOp(left, right, "id", "ID").execute()
        assert len(result) == 1

    def test_right_fields_override_left_on_collision(self):
        left = [{"ID": "1", "name": "left_name"}]
        right = [{"ID": "1", "name": "right_name"}]
        result = JoinOp(left, right, "ID", "ID").execute()
        assert result[0]["name"] == "right_name"


class TestExtractOp:
    def test_extract_specified_columns(self):
        records = [{"ID": "1", "SEX": "F", "AGE": "30"}]
        result = ExtractOp(records, ["ID", "SEX"]).execute()
        assert result == [{"ID": "1", "SEX": "F"}]

    def test_missing_column_returns_none(self):
        records = [{"ID": "1"}]
        result = ExtractOp(records, ["ID", "MISSING"]).execute()
        assert result[0]["MISSING"] is None

    def test_column_name_case_insensitive(self):
        records = [{"id": "1", "sex": "F"}]
        result = ExtractOp(records, ["ID", "SEX"]).execute()
        assert result == [{"ID": "1", "SEX": "F"}]


class TestCountOp:
    def test_count_all_records(self):
        records = [{"ID": "1"}, {"ID": "2"}, {"ID": "3"}]
        assert CountOp(records).execute() == 3

    def test_count_distinct_field(self):
        records = [{"SEX": "F"}, {"SEX": "M"}, {"SEX": "F"}]
        assert CountOp(records, field="SEX").execute() == 2

    def test_count_empty_list(self):
        assert CountOp([]).execute() == 0


class TestGroupByOp:
    def test_group_by_field(self):
        records = [
            {"SEX": "F", "ID": "1"},
            {"SEX": "M", "ID": "2"},
            {"SEX": "F", "ID": "3"},
        ]
        groups = GroupByOp(records, "SEX").execute()
        assert set(groups.keys()) == {"F", "M"}
        assert len(groups["F"]) == 2
        assert len(groups["M"]) == 1

    def test_missing_field_goes_to_unknown(self):
        records = [{"SEX": "F"}, {"AGE": "30"}]  # second record has no SEX
        groups = GroupByOp(records, "SEX").execute()
        assert "__unknown__" in groups
        assert len(groups["__unknown__"]) == 1


class TestSumOp:
    def test_sum_numeric_field_per_group(self):
        groups = {
            "F": [{"score": "10"}, {"score": "20"}],
            "M": [{"score": "30"}],
        }
        result = SumOp(groups, "score").execute()
        assert result["F"] == 30.0
        assert result["M"] == 30.0

    def test_non_numeric_values_treated_as_zero(self):
        groups = {
            "A": [{"score": "10"}, {"score": "N/A"}, {"score": "5"}],
        }
        result = SumOp(groups, "score").execute()
        assert result["A"] == 15.0
