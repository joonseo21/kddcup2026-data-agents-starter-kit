"""
DagExecutor integration tests.

The Task-11-style pipeline (RecordScan -> Join -> Extract) is verified
end-to-end using mock data and a mock llm_fn — no real LLM API calls.
"""

import json

import pytest

from data_agent_baseline.aop.dag_executor import DagExecutor, DagNode


class TestDagExecutorSingleNode:
    def test_count_node_no_children(self):
        records = [{"ID": "1"}, {"ID": "2"}, {"ID": "3"}]
        node = DagNode(
            op_type="Count",
            params={"records": records},
            children=[
                DagNode(op_type="Extract", params={"columns": ["ID"]}, children=[
                    DagNode(op_type="RecordScan", params={"records": records, "condition": "always true"}, children=[])
                ])
            ],
        )
        # Wrap: RecordScan needs an llm_fn that passes everything through.
        executor = DagExecutor(llm_fn=lambda p: json.dumps({"field": "ID", "op": "!=", "value": "__never__"}))
        result = executor.execute(node)
        assert result == 3

    def test_extract_node_with_inline_records(self):
        records = [{"ID": "1", "SEX": "F", "AGE": "30"}]
        node = DagNode(
            op_type="Extract",
            params={"columns": ["ID", "SEX"]},
            children=[
                DagNode(op_type="Count", params={}, children=[
                    # Fake child that provides data via a no-op chain:
                    # We'll embed records directly into a leaf RecordScan.
                ])
            ],
        )
        # Simpler: test Extract with a RecordScan leaf directly.
        leaf = DagNode(op_type="RecordScan", params={"records": records, "condition": "pass all"}, children=[])
        extract_node = DagNode(op_type="Extract", params={"columns": ["ID", "SEX"]}, children=[leaf])

        # LLM always returns a condition that matches everything ("ID != __never__").
        llm_fn = lambda p: json.dumps({"field": "ID", "op": "!=", "value": "__never__"})
        executor = DagExecutor(llm_fn=llm_fn)
        result = executor.execute(extract_node)
        assert result == [{"ID": "1", "SEX": "F"}]


class TestDagExecutorTask11Pipeline:
    """
    Task 11: Filter Examination where Thrombosis=2, join with Patient on ID,
    then extract ID / SEX / Diagnosis columns.
    """

    @pytest.fixture
    def examination_records(self):
        return [
            {"ID": "100", "Thrombosis": "2", "Diagnosis": "SLE"},
            {"ID": "101", "Thrombosis": "0", "Diagnosis": "PSS"},
            {"ID": "102", "Thrombosis": "2", "Diagnosis": "APS"},
        ]

    @pytest.fixture
    def patient_records(self):
        return [
            {"ID": "100", "SEX": "F", "AGE": "45"},
            {"ID": "101", "SEX": "M", "AGE": "30"},
            {"ID": "102", "SEX": "F", "AGE": "60"},
        ]

    @pytest.fixture
    def mock_llm(self):
        """LLM that parses 'Thrombosis equals 2' into the expected triple."""
        def llm_fn(prompt):
            return json.dumps({"field": "Thrombosis", "op": "=", "value": "2"})
        return llm_fn

    def _build_pipeline(self, examination_records, patient_records):
        scan_node = DagNode(
            op_type="RecordScan",
            params={"records": examination_records, "condition": "Thrombosis equals 2"},
        )
        patient_node = DagNode(
            op_type="RecordScan",
            params={"records": patient_records, "condition": "all patients"},
        )
        join_node = DagNode(
            op_type="Join",
            params={"left_key": "ID", "right_key": "ID"},
            children=[scan_node, patient_node],
        )
        extract_node = DagNode(
            op_type="Extract",
            params={"columns": ["ID", "SEX", "Diagnosis"]},
            children=[join_node],
        )
        return extract_node

    def test_full_pipeline_returns_correct_records(
        self, examination_records, patient_records, mock_llm
    ):
        # patient_node also needs an llm_fn but should pass all patients through.
        # We give a condition that always matches (field != __never__).
        def combined_llm(prompt):
            if "Thrombosis" in prompt:
                return json.dumps({"field": "Thrombosis", "op": "=", "value": "2"})
            # For patient scan: pass everything by matching ID != __never__
            return json.dumps({"field": "ID", "op": "!=", "value": "__never__"})

        pipeline = self._build_pipeline(examination_records, patient_records)
        executor = DagExecutor(llm_fn=combined_llm)
        result = executor.execute(pipeline)

        assert len(result) == 2
        ids = {r["ID"] for r in result}
        assert ids == {"100", "102"}

    def test_pipeline_filters_correctly(
        self, examination_records, patient_records
    ):
        def combined_llm(prompt):
            if "Thrombosis" in prompt:
                return json.dumps({"field": "Thrombosis", "op": "=", "value": "2"})
            return json.dumps({"field": "ID", "op": "!=", "value": "__never__"})

        pipeline = self._build_pipeline(examination_records, patient_records)
        executor = DagExecutor(llm_fn=combined_llm)
        result = executor.execute(pipeline)

        # Only IDs 100 and 102 had Thrombosis=2; ID 101 must be excluded.
        result_ids = {r["ID"] for r in result}
        assert "101" not in result_ids

        # Each row must have exactly the extracted columns.
        for row in result:
            assert set(row.keys()) == {"ID", "SEX", "Diagnosis"}
