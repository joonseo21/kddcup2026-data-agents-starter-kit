from dataclasses import dataclass, field
from typing import Any, Callable

from .operators.record_scan import RecordScanOp
from .operators.join_op import JoinOp
from .operators.extract_op import ExtractOp
from .operators.count_op import CountOp
from .operators.groupby_op import GroupByOp
from .operators.sum_op import SumOp
from .operators.sqlite_filter import SqliteFilterOp


@dataclass
class DagNode:
    op_type: str  # "RecordScan" | "SqliteFilter" | "Join" | "Extract" | "Count" | "GroupBy" | "Sum"
    params: dict
    children: list["DagNode"] = field(default_factory=list)


class DagExecutor:
    """
    Execute a DAG of operator nodes in postorder (children before parent).

    llm_fn is required only for RecordScan nodes; pass None if the DAG
    contains no RecordScan nodes.
    """

    def __init__(self, llm_fn: Callable[[str], str] | None = None) -> None:
        self.llm_fn = llm_fn

    def execute(self, node: DagNode) -> Any:
        child_results = [self.execute(child) for child in node.children]

        match node.op_type:
            case "RecordScan":
                records = node.params["records"]
                condition = node.params["condition"]
                return RecordScanOp(records, condition).execute(self.llm_fn)

            case "SqliteFilter":
                return SqliteFilterOp(
                    db_path=node.params["db_path"],
                    table=node.params["table"],
                    condition=node.params["condition"],
                ).execute(self.llm_fn)

            case "Join":
                left = child_results[0]
                right = child_results[1]
                return JoinOp(
                    left,
                    right,
                    node.params["left_key"],
                    node.params["right_key"],
                ).execute()

            case "Extract":
                records = child_results[0]
                return ExtractOp(records, node.params["columns"]).execute()

            case "Count":
                records = child_results[0]
                return CountOp(records, node.params.get("field")).execute()

            case "GroupBy":
                records = child_results[0]
                return GroupByOp(records, node.params["field"]).execute()

            case "Sum":
                groups = child_results[0]
                return SumOp(groups, node.params["field"]).execute()

            case _:
                raise ValueError(f"Unknown op_type: {node.op_type!r}")
