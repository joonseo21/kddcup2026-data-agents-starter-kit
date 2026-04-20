# KDD Cup 2026 DataAgent-Bench — 프로젝트 가이드

## 1. 대회 규칙 요약 (dataagent.top/rules)

### 평가 방식

```
점수 = Recall - λ × (추가 열 수 / 예측 열 수)
```

- 열 이름은 무시하고 **데이터 내용**만 비교
- 행 순서 무관
- 숫자는 소수점 2자리 자동 정규화
- 완전 일치가 아닌 **부분 점수** 존재 → 불필요한 컬럼을 넣으면 감점

### 제출 방식

- Docker 이미지로 제출: `<team_id>_v<N>.tar.gz` (최대 10GB)
- 출력 경로: `/output/task_<id>/prediction.csv`
- 일일 1회, Phase 1 총 30회 제한

### 실행 환경 (중요)

| 항목 | 내용 |
|------|------|
| CPU | 16코어 |
| RAM | 64GB |
| 시간 제한 | 전체 태스크 합산 **12시간** |
| 네트워크 | **외부 인터넷 완전 차단** |
| 허용 LLM | **Qwen3.5-35B-A3B 단 하나** (조직자 제공) |
| 보조 모델 | 임베딩 등 소형 모델 허용 (로컬 실행만) |

### 코드 필수 사항

API 정보를 절대 하드코딩 금지. 환경변수로 읽어야 함:

```python
MODEL_API_URL  # LLM 엔드포인트
MODEL_API_KEY  # API 키
MODEL_NAME     # 모델 이름
```

---

## 2. 데이터셋 구조

### 태스크 폴더 구조

```
data/public/input/task_<id>/
├── task.json               # task_id, difficulty, question
└── context/
    ├── knowledge.md        # 필수 — 데이터 스키마 설명
    ├── db/                 # SQLite (선택)
    ├── csv/                # CSV (선택)
    ├── json/               # JSON (선택)
    └── doc/                # Markdown 문서 (선택)
```

### 난이도별 데이터 특성

| 난이도 | 데이터 조합 | 비고 |
|--------|------------|------|
| Easy | CSV/JSON + knowledge.md | 단순 필터/조인 |
| Medium | 위 + db/ | 집계, 다중 테이블 |
| Hard | 위 + doc/ (10K~128K 토큰) | 문서 이해 필요 |
| Extreme | 위 + 128K 초과 | 장문 문서 처리 |

### 공개 데이터 통계 (50개 샘플 기준)

- SQLite: 12% / CSV: 17% / JSON: 37% / Markdown: 27%
- 전체 데이터셋에는 **이미지, PDF 포함** 예상
- 모든 태스크에 knowledge.md 포함 (필수)

---

## 3. 베이스라인 코드 구조

```
src/data_agent_baseline/
├── cli.py          # typer CLI: run-task, run-benchmark
├── config.py       # YAML 설정 → AppConfig dataclass
├── agents/
│   ├── model.py    # OpenAIModelAdapter, ModelAdapter Protocol
│   ├── react.py    # ReActAgent — Thought→Action→Observation 루프
│   ├── prompt.py   # 시스템 프롬프트, 메시지 빌더
│   └── runtime.py  # AgentRunResult, StepRecord
├── benchmark/
│   ├── dataset.py  # DABenchPublicDataset
│   └── schema.py   # PublicTask, AnswerTable
├── run/
│   └── runner.py   # run_benchmark(), run_single_task(), 타임아웃/멀티프로세스
├── tools/
│   ├── registry.py    # ToolRegistry — 도구 명세 + 실행
│   ├── filesystem.py  # list_context, read_csv, read_json, read_doc
│   ├── sqlite.py      # execute_read_only_sql, inspect_sqlite_schema
│   └── python_exec.py # execute_python_code (subprocess 격리)
└── aop/               # AOP 이식 작업 폴더 (진행 중)
    ├── parser.py
    └── semantic_parse_prompt.py
```

### 실행 명령어

```bash
uv sync                                                    # 의존성 설치
uv run pytest                                              # 전체 테스트
uv run dabench run-task task_11 --config configs/...yaml   # 단일 태스크
uv run dabench run-benchmark --config configs/...yaml      # 전체 벤치마크
```

---

## 4. AOP 이식 계획

### 핵심 아이디어

현재 베이스라인(ReAct)은 LLM이 매 스텝마다 즉흥적으로 도구를 선택하는 선형 구조.
AOP 논문의 DAG 구조를 적용해 계획 기반 병렬 실행으로 전환.

```
[현재] Task → ReAct 루프 (즉흥, 선형) → answer

[목표] Task → SemanticParser → DAG Plan → 병렬 실행 → answer
```

### 태스크 유형 분류

| 유형 | 설명 | 처리 방법 |
|------|------|----------|
| SQL형 | DB/CSV/JSON 집계·필터 | execute_context_sql / execute_python |
| 임베딩형 | PDF·이미지 시맨틱 검색 | 로컬 임베딩 모델 |
| 혼합형 | 이미지에서 ID 추출 → DB 조회 | ScanOp → SqlOp 순서 DAG |

### 연산자 입출력 형태

| Operator | 입력 | 출력 |
|----------|------|------|
| RecordScanOp(table, condition) | list[dict] | list[dict] |
| JoinOp(t1, t2, field) | list[dict] × list[dict] | list[dict] |
| ExtractOp(attrs) | list[dict] | list[dict] |
| CountOp() | list[dict] | int |
| GroupByOp(field) | list[dict] | dict[str, list[dict]] |
| SumOp(field) | dict[str, list[dict]] | dict[str, float] |

### DAG 실행 원리

```
질문: "혈전증 2등급 환자들의 ID, 성별, 진단명은?"
      ↓
RecordScanOp(Examination, "Thrombosis=2")  → 필터된 레코드
      ↓
JoinOp(Examination, Patient, "ID")         → 조인 결과
      ↓
ExtractOp(["ID", "SEX", "Diagnosis"])       → 최종 컬럼 추출
      ↓
answer
```

각 연산자의 출력이 다음 연산자의 입력으로 흐름 (postorder traversal).

### 목표 폴더 구조

```
src/data_agent_baseline/
├── aop/
│   ├── parser.py                  ← 있음
│   ├── semantic_parse_prompt.py   ← 있음
│   ├── embedding.py               ← 신규: 로컬 임베딩 인터페이스
│   ├── data_loader.py             ← 신규: context → all_records_data
│   ├── basic_questions.py         ← 신규: BQMatcher
│   ├── planner.py                 ← 신규: SemanticPlanner
│   ├── dag_executor.py            ← 신규: DAGExecutor
│   ├── logical_representations/   ← 신규: LR 정의
│   │   ├── filter.py
│   │   ├── join.py
│   │   ├── extract.py
│   │   ├── count.py
│   │   ├── groupby.py
│   │   └── sum.py
│   ├── prompts/                   ← Unify에서 이식
│   │   ├── check_bq_prerequisite.py
│   │   └── apply_bq_reduce.py
│   └── operators/                 ← 신규: KDD 도구 래핑
│       ├── record_scan_op.py
│       ├── join_op.py
│       ├── extract_op.py
│       ├── count_op.py
│       ├── groupby_op.py
│       ├── sum_op.py
│       └── operator_map.py
└── agents/
    └── aop_agent.py               ← 신규: AOPAgent
```

### 구현 순서 (의존성 순)

```
1. config.py 수정          환경변수 우선 읽기 (MODEL_API_URL 등)
2. aop/embedding.py        로컬 임베딩 인터페이스 + MockEmbedder
3. aop/logical_representations/  LR 정의 (의존성 없음)
4. aop/data_loader.py      context 폴더 파싱
5. aop/operators/          연산자 구현
6. aop/dag_executor.py     postorder traversal 실행기
7. aop/basic_questions.py  BQMatcher
8. aop/planner.py          SemanticPlanner
9. agents/aop_agent.py     AOPAgent (ReActAgent와 동일 인터페이스)
10. runner.py 수정          agent_type: aop | react 선택
```

---

## 5. 설계 결정 사항

### Gemini Embedding 2 사용 불가

평가 환경에서 외부 인터넷이 차단되므로 Gemini API 호출 불가.
로컬에서 실행 가능한 `sentence-transformers` 사용.

```python
# aop/embedding.py
class EmbeddingModel(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class SentenceTransformerEmbedder:       # 실제 사용
    def embed(self, texts): ...

class MockEmbedder:                       # 테스트 전용
    def embed(self, texts): ...
```

### LR을 명시적으로 정의하는 이유

임베딩은 "어떤 데이터를 찾을지" (retrieval)만 담당.
COUNT, SUM, GROUPBY 같은 연산 로직은 연산자로 명시적으로 정의해야 함.
LR(Logical Representation)은 "어떤 연산자를 어떤 순서로 실행할지"를 정의하는 템플릿.

### ReActAgent 유지 이유

AOPAgent가 판단 불가한 태스크나 실패 시 fallback으로 사용.
config에서 `agent_type: aop | react`로 선택 가능하게 유지.

### 부분 점수 전략

평가가 Recall 기반이므로:
- 정답 컬럼을 최대한 포함 (Recall 향상)
- 불필요한 컬럼은 제거 (패널티 방지)
- ExtractOp에서 컬럼 선택을 정확히 하는 것이 중요

---

## 6. 테스트 전략

### 원칙

모든 모듈은 LLM API 호출 없이 테스트 가능해야 함.
MockChatModel로 LLM 응답을 스크립트로 재생.

### 공통 fixture (tests/conftest.py)

```python
class MockChatModel:
    def __init__(self, responses: list[str])
    def create_completion(self, client, messages, **kwargs) -> str

@pytest.fixture
def json_task(tmp_path) -> PublicTask:
    # Examination.json + Patient.json 임시 생성

@pytest.fixture
def sqlite_task(tmp_path) -> PublicTask:
    # patients.db 임시 생성
```

### 테스트 폴더 구조

```
tests/
├── conftest.py
├── aop/
│   ├── test_data_loader.py
│   ├── test_basic_questions.py
│   ├── test_planner.py
│   ├── test_dag_executor.py
│   └── operators/
│       ├── test_record_scan_op.py
│       ├── test_join_op.py
│       ├── test_extract_op.py
│       └── test_count_op.py
└── agents/
    └── test_aop_agent.py
```

---

## 7. 참고 자료

- 대회 사이트: https://dataagent.top
- 규칙: https://dataagent.top/rules
- 원본 스타터킷: https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit
- AOP 논문: CIDR'25 — Automated and Interactive LLM Pipeline Orchestration
- Unify 구현체: ~/Unify (로컬, AOP 논문 원본 구현)
