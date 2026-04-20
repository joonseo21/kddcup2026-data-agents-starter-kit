# Unify LR 확장 플랜 — KDD Cup 2026 데이터 대응

## Context

KDD Cup 2026 DataAgent-Bench의 태스크들은 CSV/JSON/SQLite/Markdown으로 구성된
구조화된 테이블 데이터에 대한 질의를 요구한다.
현재 Unify는 비정형 문서(텍스트 청크) 단위로 동작하도록 설계되어 있어,
레코드 단위 필터링·조인·집계 등 테이블 연산을 표현할 LR이 없다.

목표: 최소한의 구조 변경으로 레코드 단위 파이프라인을 추가하고,
Task 11(Filter+Join+Extract)을 기준으로 동작을 검증한다.

---

## 핵심 분석

### KDD Cup 태스크 유형별 필요 연산

| 난이도 | 예시 태스크 | 필요 연산 |
|--------|------------|-----------|
| Easy | Task 11: 혈전증 환자 목록 | RecordFilter + Join + Extract |
| Easy | Task 38: 고객 거래 내역 | RecordFilter + Join + Extract |
| Medium | Task 163: 행사별 지출 합계 | Join + GroupBy + Sum |
| Medium | Task 214: 카드 세트 수 | RecordFilter + CountDistinct |
| Extreme | Task 418: 비정상 환자 수 | RecordFilter + CountDistinct |

### 현재 scanOP의 한계

- 텍스트 문서 전체를 하나의 단위로 처리 (파일 단위)
- field mapping이 "views", "picks"로 하드코딩
- 문자열 값 비교 불가 (숫자만)
- 다중 레코드 파일에서 레코드별 필터링 불가

---

## 변경 목록

### 1. `main/chunk/LoadChunks.py` — 구조 데이터 병렬 저장

**변경 내용**: `load_process_data_chunks()`가 `all_records_data`를 추가로 반환

```python
# 반환 타입 변경
# 기존: (all_file_data, all_chunks, all_ids, all_embeds, all_chunk_locs)
# 변경: (all_file_data, all_records_data, all_chunks, all_ids, all_embeds, all_chunk_locs)

all_records_data = {}   # {filename: [dict, dict, ...]}

# JSON 처리 — 구조 보존
elif file.endswith(".json"):
    records = data.get("records", data) if isinstance(data, dict) else data
    if isinstance(records, list):
        all_records_data[file] = records
    # 텍스트 변환은 기존 방식 유지 (임베딩용)

# SQLite 처리 — 신규 추가
elif file.endswith(".db"):
    import sqlite3
    conn = sqlite3.connect(os.path.join(doc_path, file))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    lines = []
    for table in tables:
        rows = cursor.execute(f"SELECT * FROM {table}").fetchall()
        table_records = [dict(row) for row in rows]
        all_records_data[f"{file}::{table}"] = table_records
        lines.extend(", ".join(f"{k}: {v}" for k, v in r.items()) for r in table_records)
    all_data = "\n".join(lines)
    conn.close()
```

---

### 2. `main/unify.py` + `main/API.py` — 반환값 처리

```python
# unify.py line ~354
all_file_data, all_records_data, all_chunks, ... = load_process_data_chunks(...)

# planManager 초기화에 all_records_data 추가
PM = planManager(
    query, final_plan, client, chatModel, final_BQ_list,
    all_file_data, all_records_data,
    parsed_result, partial_question_list, embedModel, index
)
```

---

### 3. `main/operators/recordScanOP.py` — 신규 파일

레코드 리스트(list of dict)를 받아 필드 조건으로 필터링.
LLM을 사용해 조건 문자열을 (field, operator, value) 트리플로 파싱.

```python
class recordScanOP:
    """
    입력:
        records: list[dict]  — all_records_data[table_name]
        condition: str       — 자연어 조건 ("Thrombosis = 2", "SEX is F", "age < 70")
    출력:
        list[dict]           — 조건을 만족하는 레코드들
    """

    PARSE_PROMPT = """Parse the following filter condition into a JSON object.
Condition: "{condition}"

Return ONLY a JSON object:
{{"field": "<field name>", "op": "<= | != | > | < | >= | =>", "value": "<value as string>"}}

Examples:
- "Thrombosis = 2"       → {{"field": "Thrombosis", "op": "=", "value": "2"}}
- "SEX is F"             → {{"field": "SEX", "op": "=", "value": "F"}}
- "age less than 70"     → {{"field": "age", "op": "<", "value": "70"}}
- "approved equals true" → {{"field": "approved", "op": "=", "value": "true"}}"""

    def __init__(self, records, condition):
        self.records = records if isinstance(records, list) else list(records.values())
        self.condition = condition

    def _parse_condition(self, client, chatModel):
        prompt = self.PARSE_PROMPT.format(condition=self.condition)
        response = chatModel.create_completion(client, messages=[{"role": "user", "content": prompt}])
        import json
        return json.loads(response)

    def _match(self, record, field, op, value):
        rec_field = next((k for k in record if k.lower() == field.lower()), None)
        if rec_field is None:
            return False
        rec_val = record[rec_field]
        try:
            rv, v = float(str(rec_val)), float(value)
            if op == "=":  return rv == v
            if op == "!=": return rv != v
            if op == ">":  return rv > v
            if op == "<":  return rv < v
            if op == ">=": return rv >= v
            if op == "<=": return rv <= v
        except (ValueError, TypeError):
            if op in ("=", "=="):  return str(rec_val).strip().lower() == value.strip().lower()
            if op == "!=":         return str(rec_val).strip().lower() != value.strip().lower()
        return False

    def execute(self, client, chatModel, ctxManager):
        parsed = self._parse_condition(client, chatModel)
        field, op, value = parsed["field"], parsed["op"], parsed["value"]
        result = [r for r in self.records if self._match(r, field, op, value)]
        print(f"RecordScan: {field} {op} {value} → {len(result)} records")
        return result, ctxManager
```

---

### 4. `main/operators/operatorMap.py` — RecordScan 등록

```python
from .recordScanOP import recordScanOP

OP_MAP = {
    ...
    "RecordScan": recordScanOP,
}
```

---

### 5. `main/PlanManager.py` — get_operator_instance 수정

**`__init__`**: `all_records_data` 파라미터 추가

```python
def __init__(self, ..., all_file_data, all_records_data, ...):
    self.all_records_data = all_records_data
```

**`get_operator_instance`**: RecordScan / Join 분기 추가

```python
if operator_type == "RecordScan":
    table_param = mapping.get(operator["Parameter"][0], operator["Parameter"][0])
    condition_param = mapping.get(operator["Parameter"][1], operator["Parameter"][1])
    records = self.all_records_data.get(table_param, [])
    return recordScanOP(records, condition_param)

elif operator_type == "Join":
    t1 = mapping.get(operator["Parameter"][0], operator["Parameter"][0])
    t2 = mapping.get(operator["Parameter"][1], operator["Parameter"][1])
    a1 = mapping.get(operator["Parameter"][2], operator["Parameter"][2])
    a2 = mapping.get(operator["Parameter"][3], operator["Parameter"][3])
    ds1 = self.all_records_data.get(t1, [])
    ds2 = self.all_records_data.get(t2, [])
    return joinOP(ds1, ds2, a1, a2)
```

---

### 6. `main/semanticParse/logicalRepresentations/filter.py` — 신규 LR 추가

```python
RECORD_FILTER_LR = {
    "Question": "Records from [Table] where [Field] equals [Value]",
    "IDQuestion": "Records from [Table1] where [Field1] equals [Value1]",
    "Plan": [{"RecordScan": []}],
    "IDPlan": [{
        "Operator": "RecordScan",
        "Parameter": ["[Table1]", "[Field1] equals [Value1]"],
        "Followup Plan": []
    }],
    "Return": "records"
}

RECORD_FILTER_NUMERIC_LR = {
    "Question": "Records from [Table] where [Field] is greater than [Value]",
    "IDQuestion": "Records from [Table1] where [Field1] is greater than [Value1]",
    "Plan": [{"RecordScan": []}],
    "IDPlan": [{
        "Operator": "RecordScan",
        "Parameter": ["[Table1]", "[Field1] greater than [Value1]"],
        "Followup Plan": []
    }],
    "Return": "records"
}
```

---

### 7. `main/semanticParse/logicalRepresentations/join.py` — JOIN_SIMPLE_LR 추가

```python
JOIN_SIMPLE_LR = {
    "Question": "Join [Table] and [Table] on [Field]",
    "IDQuestion": "Join [Table1] and [Table2] on [Field1]",
    "Plan": [{"Join": []}],
    "IDPlan": [{
        "Operator": "Join",
        "Parameter": ["[Table1]", "[Table2]", "[Field1]", "[Field1]"],
        "Followup Plan": []
    }],
    "Return": "records"
}
```

---

### 8. `main/semanticParse/basicQuestions.py` — 신규 LR 등록

```python
from .logicalRepresentations import (
    ...,
    RECORD_FILTER_LR, RECORD_FILTER_NUMERIC_LR,
    JOIN_SIMPLE_LR,
)

BasicQuestions = [
    ...,
    RECORD_FILTER_LR,
    RECORD_FILTER_NUMERIC_LR,
    JOIN_SIMPLE_LR,
]
```

---

### 9. `main/knowledge_base/BQReductionKnowledgeBase.json` — KDD Cup 스타일 예시 추가

```json
{
    "id": "example_22",
    "query": {
        "original": "Filter Examination records where Thrombosis equals 2",
        "parsed": "Records from [Table] where [Field] equals [Value]"
    },
    "BQ": { "question": "Records from [Table] where [Field] equals [Value]", "return": "records" },
    "judgment": {
        "fully_solved": true, "partially_solved": false,
        "transformed_original_query": "records", "sub_problems": "No",
        "transformed_parsed_query": "[records]"
    },
    "explanation": "Direct field equality filter on structured records."
},
{
    "id": "example_23",
    "query": {
        "original": "Join Patient and Examination tables on ID field",
        "parsed": "Join [Table] and [Table] on [Field]"
    },
    "BQ": { "question": "Join [Table] and [Table] on [Field]", "return": "records" },
    "judgment": {
        "fully_solved": true, "partially_solved": false,
        "transformed_original_query": "records", "sub_problems": "No",
        "transformed_parsed_query": "[records]"
    },
    "explanation": "Simple FK join between two tables on a shared field."
},
{
    "id": "example_24",
    "query": {
        "original": "For patients with severe degree of thrombosis, list their ID, sex and disease",
        "parsed": "For [Entity] with [Condition], list their [Attribute] and [Attribute] and [Attribute]"
    },
    "BQ": { "question": "Records from [Table] where [Field] equals [Value]", "return": "records" },
    "judgment": {
        "fully_solved": false, "partially_solved": true,
        "transformed_original_query": "sub_problems",
        "sub_problems": [
            "Records from Examination where Thrombosis equals 2",
            "Join filtered records with Patient on ID",
            "List ID, sex and disease from joined records"
        ],
        "transformed_parsed_query": "sub_problems"
    },
    "explanation": "Multi-step: filter by field value, join tables, then extract attributes."
}
```

---

## 전체 파이프라인 (Task 11 기준)

```
질문: "For patients with severe degree of thrombosis, list their ID, sex and disease"
                ↓
[semantic_parse]
  Entities: ["patients", "Examination", "Patient"]
  Conditions: ["severe degree of thrombosis" → Thrombosis=2]
  Attributes: ["ID", "SEX", "Diagnosis"]
                ↓
[recursive_plan_generation]
  BQ 1: RECORD_FILTER_LR
        → RecordScan("Examination.json", "Thrombosis equals 2")
        → 해당 환자 Examination 레코드 리스트
  BQ 2: JOIN_SIMPLE_LR
        → Join("Examination.json", "Patient.json", "ID")
        → 조인된 레코드 리스트
  BQ 3: EXTRACT_LR
        → Extract(["ID", "SEX", "Diagnosis"])
        → 최종 결과 리스트
                ↓
[execute_with_plan — postorder]
  RecordScan → joinOP → extractOP
```

---

## 입출력 형태 요약

| Operator | 입력 | 출력 |
|----------|------|------|
| RecordScan(table, condition) | `all_records_data[table]`: list[dict], condition: str | list[dict] |
| Join(t1, t2, f1, f2) | list[dict] × list[dict] | list[dict] |
| Extract(attrs) | list[dict] | list[dict] (지정 컬럼만) |
| Count(records) | list[dict] | int |
| GroupBy(records, field) | list[dict] | dict[str, list[dict]] |
| Sum(grouped, field) | dict[str, list[dict]] | dict[str, float] |

---

## 검증 방법

```bash
# Task 11로 검증 (Unify 기준)
cd ~/Unify/main
bash myrun.sh 2>&1 | tee output.txt
grep "Extract result\|final result" output.txt
# 기대 결과: ID/SEX/Diagnosis 3개 레코드
```

---

## 작업 순서

1. `LoadChunks.py` — SQLite 지원 + `all_records_data` 반환 추가
2. `recordScanOP.py` — 신규 파일 생성
3. `operatorMap.py` — RecordScan 등록
4. `PlanManager.py` — `all_records_data` 전달 + `get_operator_instance` 분기
5. `unify.py` / `API.py` — `all_records_data` 연결
6. `filter.py` + `join.py` — 신규 LR 정의
7. `__init__.py` + `basicQuestions.py` — 등록
8. `BQReductionKnowledgeBase.json` — 예시 3개 추가
