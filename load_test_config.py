"""부하 테스트 설정 및 상수."""

from dataclasses import dataclass
from typing import Optional

# 동시성 수준: 1, 5, 10, 20, 50, 100, 150, 200
CONCURRENCY_LEVELS = [1, 5, 10, 20, 50, 100, 150, 200]

# 테스트용 간단한 프롬프트
TEST_PROMPT = "안녕하세요. 간단히 인사해주세요."

# 모델 이름
TEXT_MODEL_NAME = "text"

# 타임아웃 설정 (초)
REQUEST_TIMEOUT = 120
TOTAL_TIMEOUT = 3600  # 1시간

# 결과 파일 경로
RESULT_DIR = "result"
RESULT_FILENAME_PREFIX = "load_test"


@dataclass
class LoadTestConfig:
    """부하 테스트 설정."""

    concurrency_levels: list[int] = None
    test_prompt: str = TEST_PROMPT
    model_name: str = TEXT_MODEL_NAME
    request_timeout: float = REQUEST_TIMEOUT
    total_timeout: float = TOTAL_TIMEOUT
    result_dir: str = RESULT_DIR
    result_filename_prefix: str = RESULT_FILENAME_PREFIX

    def __post_init__(self):
        if self.concurrency_levels is None:
            self.concurrency_levels = CONCURRENCY_LEVELS
