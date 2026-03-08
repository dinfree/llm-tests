# LLM 성능 테스트 도구

OpenAI 호환 엔드포인트에서 여러 LLM 모델 성능을 비교하기 위한 CLI 벤치마크 도구입니다.

## 주요 기능

- 시작 시 `GET /v1/models`로 모델 목록을 조회합니다.
- 선택한 1개 이상 모델에 대해 `Single test`, `Multi test`를 실행합니다.
- 결과를 `results.db`(SQLite)에 저장합니다.
- 차트 이미지를 `result/`에 저장합니다.
- 프로그램 시작 시 `result/`의 기존 PNG를 `prev_result/`로 이동합니다.

## 프로젝트 구조

- `main.py`: 벤치마크 실행 및 메뉴
- `results.db`: 성능 측정 이력 DB
- `result/`: 최신 차트 이미지
- `prev_result/`: 이전 차트 이미지 보관
- `.env`: 실행 환경 설정

## 요구사항

- Python 3.10 이상
- OpenAI 호환 API 서버 (`/v1/models`, `/v1/chat/completions`)
- `requirements.txt`에 정의된 패키지

패키지 설치:

```bash
pip install -r requirements.txt
```

## 환경 변수 설정

`.env`에 아래 값을 설정합니다.

- `BASE_URL` (필수): API 기본 URL (예: `http://host:port/v1`)
- `API_KEY` (필수): API 키 (`OPENAI_API_KEY` 또는 `VLLM_API_KEY` 대체 가능)
- `BASE_PROMPT` (선택): 요청에 사용할 기본 프롬프트
- `MAX_TOKENS` (선택, 기본값 `512`): 최대 생성 토큰 수
- `TEMPERATURE` (선택, 기본값 `0.0`): 생성 온도
- `REQUEST_TIMEOUT` (선택, 기본값 `180`): 요청 타임아웃(초)

## 실행 방법

```bash
python3 main.py
```

## 비전 모델 간단 테스트

동일 환경에서 이미지 입력이 가능한 모델을 빠르게 점검하려면 아래 스크립트를 사용합니다.

- `vision_test.py`: 비전 요청 1회를 보내는 간단 테스트
- `sample_image.png`: 샘플 입력 이미지

주의: 이번 스크립트는 요청 사항에 맞춰 `BASE_URL`을 코드 내부에 하드코딩했습니다.

```bash
python3 vision_test.py
```

다른 이미지로 테스트:

```bash
python3 vision_test.py ./sample_image.png "What is in this image?"
```

기본 동작:

1. `GET /v1/models`로 첫 번째 모델을 선택
2. 로컬 이미지 파일을 base64 data URL로 변환
3. `POST /v1/chat/completions`로 텍스트+이미지 멀티모달 요청

프로그램 시작 시 동작:

1. DB 스키마를 초기화하고 필요한 컬럼을 자동 반영합니다.
2. `result/`의 기존 PNG를 `prev_result/`로 이동합니다.
3. 서버에서 사용 가능한 모델 목록을 가져옵니다.

메뉴:

1. `Single test`
2. `Multi test`
3. `View results`
4. `Exit`

모델 선택 입력:

- 전체 실행: `all`
- 일부 모델 실행: 인덱스 콤마 구분 입력 (예: `1,3,5`)

동시성 입력(`Multi test`):

- 양의 정수를 콤마로 입력 (예: `10,100,200`)

## 산출물

### 1) 콘솔 출력

- Single test: 모델별 토큰/지연/처리량 출력
- Multi test: 동시성별 요약 테이블 출력

### 2) DB (`results.db`)

테이블: `test_results`

주요 컬럼:

- `run_id`: 한 번의 배치 실행을 식별하는 ID
- `test_type`: `single` 또는 `multi`
- `model_name`: 서버 모델 ID
- `concurrency`: 동시 요청 수 (`single`은 1)
- `total_requests`: 총 요청 수
- `success_requests`: 성공 요청 수
- `error_requests`: 실패 요청 수
- `success_rate`: 성공률(%)
- `total_tokens`: 생성된 출력 토큰 수
- `total_duration`: 전체 소요 시간(초)
- `ttft`: 평균 TTFT (`single`은 단일 요청 TTFT)
- `ttft_p95`: TTFT 95 퍼센타일
- `tps`: 토큰 처리량(tokens/sec)
- `notes`: 오류 샘플 메시지

### 3) 차트 (`result/`)

- Single test: `<model>_single_1.png`
- Multi test: `<model>_multi_<conc1>_<conc2>_...png`

모델명은 파일 저장 시 안전한 문자열로 변환됩니다.

## 성능 지표 정의

- `Total Tokens`
	- 스트리밍 응답에서 관측된 생성 토큰 수

- `Total Duration (s)`
	- 요청 시작부터 스트림 종료까지의 경과 시간

- `TTFT (First Token Latency)`
	- 요청 시작 후 첫 토큰이 도착할 때까지 걸린 시간

- `TTFT p95`
	- 한 동시성 구간에서 성공 요청들의 TTFT 95 퍼센타일
	- 평균보다 꼬리 지연(Tail latency) 파악에 유리

- `TPS (Throughput, tokens/s)`
	- `total_tokens / total_duration`
	- `multi`에서는 해당 동시성의 가장 느린 성공 요청 시간을 기준으로 계산

- `Success Rate (%)`
	- `(success_requests / total_requests) * 100`

## 지표 해석 권장 순서

1. `Success Rate`로 안정성 확인
2. `TTFT p95`로 체감 지연 리스크 확인
3. `TPS`로 처리량 비교
4. 동일 조건 비교는 `run_id` 기준으로 수행

## 재현 가능한 벤치마크 팁

- 모델 간 비교 시 동일 프롬프트, 동일 `MAX_TOKENS`를 사용하세요.
- 서버 부하/하드웨어 조건을 가능한 고정하세요.
- 각 조건을 반복 실행하고 `run_id` 단위로 비교하세요.
- 성능 테스트 목적이면 `TEMPERATURE=0.0`을 권장합니다.

## 샘플 실행 화면

### 1) 시작 및 메뉴

```text
============================================================
LLM Performance Test Tool
============================================================
Fetching model list from server ...
Server  : http://127.0.0.1:8000/v1
Models  : 3 found
============================================================

Menu
	1. Single test
	2. Multi test
	3. View results
	4. Exit
Select [1-4]:
```

### 2) Single test 예시

```text
Select [1-4]: 1

Available models
------------------------------------------------------------
	 1. Qwen/Qwen3-8B
	 2. Qwen/Qwen3-30B-A3B
	 3. Meta-Llama-3.1-8B-Instruct
------------------------------------------------------------
Select models (all or comma index, example: 1,3): 1,2

Starting single tests | run_id=a1b2c3d4e5 | models=2

[Single] model=Qwen/Qwen3-8B
... (streaming output) ...

================================================================================
Single Test Results
================================================================================
Model               : Qwen/Qwen3-8B
Total Tokens        : 287
Total Duration (s)  : 3.21
TTFT (s)            : 0.412
Throughput TPS      : 89.41
================================================================================
```

### 3) Multi test 예시

```text
Select [1-4]: 2

Available models
------------------------------------------------------------
	 1. Qwen/Qwen3-8B
	 2. Qwen/Qwen3-30B-A3B
	 3. Meta-Llama-3.1-8B-Instruct
------------------------------------------------------------
Select models (all or comma index, example: 1,3): 2
Enter concurrency levels (example: 10, 100, 200): 10,25,50

Starting multi tests | run_id=f6g7h8i9j0 | models=1 | concurrency=[10, 25, 50]

[Multi] model=Qwen/Qwen3-30B-A3B

==============================================================================================================
Multi Test Results - Model: Qwen/Qwen3-30B-A3B
==============================================================================================================
	Conc |   Tokens |   Dur(s) |  TTFT Avg |  TTFT p95 |        TPS |   Success |    Rate
--------------------------------------------------------------------------------------------------------------
		10 |     3120 |     9.42 |     0.588 |     0.901 |     331.21 |   10/10   |  100.0%
		25 |     7450 |    19.37 |     0.973 |     1.682 |     384.61 |   24/25   |   96.0%
		50 |    13820 |    41.15 |     1.844 |     3.105 |     335.84 |   45/50   |   90.0%
==============================================================================================================
```

### 4) 결과 조회 예시

```text
Select [1-4]: 3

================================================================================================================================================================
Test Results
================================================================================================================================================================
	ID |           Timestamp |     Run ID |   Type |                              Model | Conc |     Req |    Succ |   Rate |   Tokens |  Dur(s) |    TTFT |     P95 |      TPS
----------------------------------------------------------------------------------------------------------------------------------------------------------------
	42 | 2026-03-08T11:21:01 | a1b2c3d4e5 | single |                        Qwen/Qwen3-8B |    1 |       1 |       1 | 100.0% |      287 |    3.21 |   0.412 |   0.412 |    89.41
	41 | 2026-03-08T11:18:44 | f6g7h8i9j0 |  multi |                  Qwen/Qwen3-30B-A3B |   25 |      25 |      24 |  96.0% |     7450 |   19.37 |   0.973 |   1.682 |   384.61
================================================================================================================================================================
```

## 트러블슈팅

- `No models found on server`
	- `BASE_URL`이 올바른지, `/v1/models` 응답이 정상인지 확인하세요.

- `notes`에 `HTTP 4xx/5xx`가 저장됨
	- API 키, 모델명 유효성, 서버 상태를 확인하세요.

- TPS가 매우 낮거나 TTFT p95가 높음
	- 서버 포화 여부, 동시성 수준, 타임아웃 설정을 점검하세요.

