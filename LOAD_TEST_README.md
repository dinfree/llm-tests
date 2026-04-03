# LLM Proxy 부하 테스트 프로그램

LLM Proxy 서버의 성능 특성을 평가하기 위한 부하 테스트 도구입니다. 동시 사용자 수 증가에 따른 **TTFT(Time-To-First-Token)** 와 **TPS(Token Per Second)** 의 변화를 측정합니다.

## 기능

- **동시성 테스트**: 1명부터 200명까지 동시 사용자 시뮬레이션 (커스터마이징 가능)
- **Burst 패턴**: 모든 사용자가 동시에 한 번씩 요청 (현실적인 서버 부하 시뮬레이션)
- **상세 메트릭 수집**:
  - TTFT (첫 토큰까지의 시간)
  - TPS (초당 토큰 생성 속도)
  - 요청 성공률, 토큰 사용량, 표준편차
- **다중 출력 포맷**:
  - 콘솔: 컬럼 기반 실시간 테이블
  - 파일: 테스트 개요 + 상세 통계 + **아스키 차트**

## 설치 및 준비

### 환경 설정
프로젝트의 `.env` 파일에 다음 변수를 설정해야 합니다:

```bash
API_KEY=your-api-key
BASE_URL=http://localhost:8000/v1  # OpenAI 호환 API 엔드포인트
```

### 의존성
기존 `requirements.txt` 파일이 필요한 모든 의존성을 포함하고 있습니다:

```bash
pip install -r requirements.txt
```

## 사용 방법

### 기본 실행
기본 설정(동시성 1, 5, 10, 20, 50, 100, 150, 200)으로 전체 테스트를 실행합니다:

```bash
python load_test.py --all
```

### 빠른 테스트
소규모 동시성(1, 5, 10, 20)으로 빠른 테스트를 실행합니다:

```bash
python load_test.py --quick
```

### 최소 테스트
2개의 동시성 수준(1, 2)으로 프로그램 동작을 확인합니다:

```bash
python load_test.py --min
```

### 커스텀 동시성 수준
특정 동시성 수준만 테스트할 수 있습니다:

```bash
python load_test.py --levels 1,10,50,100
```

### 커스텀 프롬프트
테스트 프롬프트를 변경할 수 있습니다:

```bash
python load_test.py --quick --prompt "Tell me about yourself in 100 words"
```

### 도움말
사용 가능한 옵션을 확인합니다:

```bash
python load_test.py --help
```

## 결과 해석

### 콘솔 출력
각 동시성 수준별 즉시 결과:

```
       동시성 |      요청수 |     성공 |     실패 |     성공률 |    TTFT평균(ms) | ... |      TPS평균
---------------------------------------...
         1 |        1 |      1 |      0 |  100.0% |        152.63 | ... |      29.36
         5 |        5 |      5 |      0 |  100.0% |        284.16 | ... |      15.41
        10 |       10 |     10 |      0 |  100.0% |        313.79 | ... |      11.10
```

### 결과 파일
테스트 완료 후 `result/` 디렉토리에 타임스탬프 파일이 생성됩니다:

```
result/load_test_20260401_225404.txt
```

파일 내용:
1. **테스트 개요**: 시간, 소요 시간, 동시성 수준, 모델명
2. **메인 테이블**: 모든 메트릭을 컬럼 형식으로 정리
3. **아스키 차트**: TTFT와 TPS의 시각적 추이
4. **상세 통계**: 동시성별 최소/최대/표준편차 등

### 주요 메트릭

| 메트릭 | 설명 | 낮을수록 좋음 |
|--------|------|--------------|
| **TTFT (Time-To-First-Token)** | 요청 후 첫 번째 토큰이 반환되는 시간 | ✓ |
| **TPS (Token Per Second)** | 초당 생성되는 토큰 수 | ✗ (높을수록 좋음) |
| **성공률** | 성공한 요청의 비율 (%) | ✓ |
| **표준편차** | 응답 시간의 일관성 | ✓ (낮을수록 일관적) |

## 성능 분석 팁

### 1. TTFT 분석
- **증가 추세**: 동시성 증가에 따라 TTFT가 선형적으로 증가하는 것이 정상
- **급격한 증가**: 서버가 병목에 도달했을 수 있음
- **최대 TTFT**: 사용자 체감 지연 시간으로 중요

### 2. TPS 분석
- **감소 추세**: 동시성 증가로 처리량이 감소하는 것이 정상
- **일정한 TPS**: 서버 리소스가 효율적으로 활용 중
- **급격한 감소**: 병목(메모리, GPU, I/O)이 발생했을 수 있음

### 3. 성공률
- **100% 이하**: 높은 동시성에서 타임아웃 또는 오류 발생
- **개선 방안**: 타임아웃 설정 증가 (`load_test_config.py`에서 `REQUEST_TIMEOUT` 조정)

### 4. 표준편차
- **낮음 (< 0.01)**: 서버 응답이 일관적
- **높음**: 서버의 성능이 불안정하거나 리소스 경합 발생

## 고급 설정

### 타임아웃 조정
`load_test_config.py`에서 타임아웃 값을 수정:

```python
REQUEST_TIMEOUT = 120  # 개별 요청 타임아웃 (초)
TOTAL_TIMEOUT = 3600   # 전체 테스트 타임아웃 (초)
```

### 동시성 범위 변경
`load_test_config.py`에서 기본 수준 정의:

```python
CONCURRENCY_LEVELS = [1, 5, 10, 20, 50, 100, 150, 200]
```

### 테스트 프롬프트 변경
`load_test_config.py`에서 기본 프롬프트 변경:

```python
TEST_PROMPT = "당신의 테스트 프롬프트를 여기에 입력하세요."
```

## 예제

### 예제 1: 기본 부하 테스트
```bash
python load_test.py --all
```

### 예제 2: 영어로 테스트
```bash
python load_test.py --levels 1,5,10,20,50 --prompt "Hello, please respond in English"
```

### 예제 3: 긴 응답 성능 테스트
```bash
python load_test.py --quick --prompt "Write a comprehensive essay on artificial intelligence and its impact on society"
```

## 제한사항

- **텍스트 모델만 지원**: 현재 텍스트 API만 부하 테스트 가능 (Vision/Embedding은 `simple_model_test.py` 참고)
- **단일 프롬프트**: 각 동시성 수준에서 동일한 프롬프트로 테스트
- **로컬 실행**: 클라이언트 머신에서 실행되므로 네트워크 지연 포함

## 트러블슈팅

### 모든 요청이 실패하는 경우
1. `.env` 파일에 `API_KEY`와 `BASE_URL` 설정 확인
2. 프록시 서버가 실행 중인지 확인
3. 타임아웃 설정 증가: `--timeout` 옵션 추가 (현재 없음, 코드 수정 필요)

### 메모리 부족 오류
- 동시성 수준을 줄이세요: `--levels 1,5,10`
- `--min` 옵션으로 먼저 테스트

### 결과 파일이 없는 경우
- `result/` 디렉토리 생성 권한 확인
- 콘솔 출력에서 에러 메시지 확인

## 파일 구조

```
load_test.py              # 메인 부하 테스트 프로그램
load_test_config.py       # 설정 및 상수
load_test_utils.py        # 메트릭 수집, 통계, 출력 유틸리티
LOAD_TEST_README.md       # 이 파일
result/                   # 테스트 결과 파일 저장 디렉토리
  └─ load_test_*.txt      # 타임스탬프 결과 파일
```

## 기타 문서

- `simple_model_test.py`: 단일 모델 테스트 (Text, Vision, Embedding/RAG)
- `README.md`: 프로젝트 개요
- `PRD.md`: 제품 요구사항 문서

## 라이센스

이 프로젝트는 테스트 목적으로 만들어졌습니다.
