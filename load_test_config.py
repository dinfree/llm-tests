"""부하 테스트 설정 및 상수."""

from dataclasses import dataclass
from typing import Optional

# 동시성 수준: 1, 5, 10, 20, 50, 100, 150, 200
CONCURRENCY_LEVELS = [1, 5, 10, 20, 50, 100, 150, 200]

# 테스트용 간단한 프롬프트
TEST_PROMPT = "안녕하세요. 간단히 인사해주세요."

# 타임아웃 설정 (초)
REQUEST_TIMEOUT = 120
TOTAL_TIMEOUT = 3600  # 1시간

# 결과 파일 경로
RESULT_DIR = "result"


@dataclass
class LoadTestConfig:
    """부하 테스트 설정."""

    concurrency_levels: list[int] = None
    test_prompt: str = TEST_PROMPT
    # 서버 주소 조회로 런타임에 결정되므로 기본값 없이 시작한다.
    model_name: Optional[str] = None
    request_timeout: float = REQUEST_TIMEOUT
    total_timeout: float = TOTAL_TIMEOUT
    result_dir: str = RESULT_DIR

    def __post_init__(self):
        if self.concurrency_levels is None:
            self.concurrency_levels = CONCURRENCY_LEVELS
