"""LLM Proxy 부하 테스트 프로그램."""

import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from load_test_config import LoadTestConfig, CONCURRENCY_LEVELS
from load_test_utils import (
    ConcurrencyMetrics,
    RequestMetrics,
    ResultFormatter,
)

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent


class ConfigError(Exception):
    """필수 설정값(.env)이 누락되었을 때 발생하는 예외."""

    pass


def get_required_env(key: str) -> str:
    """비어 있지 않은 환경 변수를 반환하고, 누락 시 예외를 발생시킨다."""
    value = os.getenv(key, "").strip()
    if not value:
        raise ConfigError(f"환경 변수 {key} 가 설정되어 있지 않습니다.")
    return value


def create_chat_model(model_name: str) -> ChatOpenAI:
    """텍스트 테스트에 사용하는 Chat 모델 클라이언트를 생성한다."""
    api_key = get_required_env("API_KEY")
    base_url = get_required_env("BASE_URL")
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        model_kwargs={"stream_options": {"include_usage": True}},
    )


def _extract_text(chunk: Any) -> str:
    """스트리밍 청크에서 텍스트만 추출한다."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)

    return ""


def _extract_usage(chunk: Any) -> dict[str, int]:
    """스트리밍 청크에서 토큰 사용량 메타데이터를 추출한다."""
    usage = getattr(chunk, "usage_metadata", None)
    if isinstance(usage, dict):
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        if input_tokens or output_tokens:
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

    response_metadata = getattr(chunk, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage", {})
        if isinstance(token_usage, dict):
            input_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
            output_tokens = int(token_usage.get("completion_tokens", 0) or 0)
            if input_tokens or output_tokens:
                return {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }

    return {}


def _estimate_tokens(text: str) -> int:
    """usage 정보가 없을 때 사용할 간단한 토큰 추정치(단어 수 기반)."""
    if not text.strip():
        return 0
    return max(1, len(text.split()))


def _stream_request(
    llm: ChatOpenAI,
    prompt: str,
) -> tuple[Optional[float], list[str], dict[str, int]]:
    """동기적으로 스트리밍 요청을 수행한다."""
    first_token_time: Optional[float] = None
    output_parts: list[str] = []
    usage: dict[str, int] = {}
    start_time = time.perf_counter()

    for chunk in llm.stream([HumanMessage(content=prompt)]):
        chunk_text = _extract_text(chunk)

        if chunk_text and first_token_time is None:
            first_token_time = time.perf_counter()

        if chunk_text:
            output_parts.append(chunk_text)

        current_usage = _extract_usage(chunk)
        if current_usage:
            usage = current_usage

    return first_token_time, output_parts, usage


async def single_request(
    llm: ChatOpenAI,
    prompt: str,
    concurrency: int,
    request_id: int,
    timeout: float,
) -> RequestMetrics:
    """단일 요청을 실행하고 메트릭을 수집한다."""
    start_time = time.perf_counter()

    try:
        # 동기 작업을 스레드풀에서 실행
        first_token_time, output_parts, usage = await asyncio.wait_for(
            asyncio.to_thread(_stream_request, llm, prompt),
            timeout=timeout,
        )

        end_time = time.perf_counter()

        output_text = "".join(output_parts)
        output_tokens = usage.get("output_tokens", _estimate_tokens(output_text))
        input_tokens = usage.get("input_tokens", 0)

        if first_token_time is not None:
            ttft = first_token_time - start_time
            gen_duration = max(end_time - first_token_time, 1e-9)
        else:
            ttft = None
            gen_duration = max(end_time - start_time, 1e-9)

        tps = output_tokens / gen_duration if output_tokens else 0.0

        return RequestMetrics(
            concurrency=concurrency,
            success=True,
            ttft=ttft,
            tps=tps,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    except asyncio.TimeoutError:
        return RequestMetrics(
            concurrency=concurrency,
            success=False,
            error_message="요청 타임아웃",
        )
    except Exception as e:
        return RequestMetrics(
            concurrency=concurrency,
            success=False,
            error_message=str(e),
        )


async def run_concurrent_requests(
    llm: ChatOpenAI,
    prompt: str,
    concurrency: int,
    timeout: float,
) -> list[RequestMetrics]:
    """N개의 동시 요청을 실행한다."""
    tasks = [
        single_request(llm, prompt, concurrency, i, timeout)
        for i in range(concurrency)
    ]
    return await asyncio.gather(*tasks)


def print_usage() -> None:
    """사용 방법을 출력한다."""
    print(
        """
사용 방법: python load_test.py [옵션]

옵션:
  --all              전체 동시성 수준(1, 5, 10, 20, 50, 100, 150, 200) 테스트
  --quick            빠른 테스트(1, 5, 10, 20) 테스트
  --min              최소 테스트(1, 2)
  --levels N,N,...   사용자 정의 동시성 수준 (예: --levels 1,5,10)
  --prompt TEXT      테스트 프롬프트 (기본값: "안녕하세요. 간단히 인사해주세요.")
  --help             이 도움말 표시

예제:
  python load_test.py --all
  python load_test.py --quick
  python load_test.py --levels 1,10,50,100
  python load_test.py --all --prompt "Hello, please respond in English"
"""
    )


def parse_args() -> LoadTestConfig:
    """명령행 인자를 파싱한다."""
    config = LoadTestConfig()

    if len(sys.argv) == 1:
        # 기본값 사용
        return config

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg == "--help":
            print_usage()
            sys.exit(0)
        elif arg == "--all":
            config.concurrency_levels = CONCURRENCY_LEVELS
        elif arg == "--quick":
            config.concurrency_levels = [1, 5, 10, 20]
        elif arg == "--min":
            config.concurrency_levels = [1, 2]
        elif arg == "--levels":
            if i + 1 < len(sys.argv):
                i += 1
                try:
                    config.concurrency_levels = [
                        int(x.strip()) for x in sys.argv[i].split(",")
                    ]
                except ValueError:
                    print("오류: --levels 옵션은 쉼표로 구분된 숫자여야 합니다.")
                    sys.exit(1)
            else:
                print("오류: --levels 옵션에 값이 필요합니다.")
                sys.exit(1)
        elif arg == "--prompt":
            if i + 1 < len(sys.argv):
                i += 1
                config.test_prompt = sys.argv[i]
            else:
                print("오류: --prompt 옵션에 값이 필요합니다.")
                sys.exit(1)
        else:
            print(f"오류: 알 수 없는 옵션 '{arg}'")
            print_usage()
            sys.exit(1)

        i += 1

    return config


def run_load_test(config: LoadTestConfig) -> None:
    """부하 테스트를 실행한다."""
    print("\n" + "=" * 100)
    print("LLM Proxy 부하 테스트 시작")
    print("=" * 100 + "\n")

    try:
        llm = create_chat_model(config.model_name)
    except ConfigError as e:
        print(f"설정 오류: {e}")
        return

    formatter = ResultFormatter(config.result_dir, config.result_filename_prefix)
    all_metrics: list[ConcurrencyMetrics] = []
    test_start_time = time.perf_counter()

    for concurrency in config.concurrency_levels:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 동시성 {concurrency}명 테스트 시작...")

        level_metrics = ConcurrencyMetrics(concurrency=concurrency)

        try:
            # 동시 요청 실행
            request_results = asyncio.run(
                run_concurrent_requests(
                    llm,
                    config.test_prompt,
                    concurrency,
                    config.request_timeout,
                )
            )

            for result in request_results:
                level_metrics.add_request(result)

            all_metrics.append(level_metrics)

            # 진행 상황 출력
            ttft_stats = level_metrics.get_ttft_stats()
            tps_stats = level_metrics.get_tps_stats()
            success_rate = level_metrics.get_success_rate()

            print(
                f"  ✓ 완료: {level_metrics.successful_requests}/{level_metrics.total_requests} 성공 "
                f"({success_rate:.1f}%), "
                f"TTFT: {ttft_stats['avg']*1000:.1f}ms, "
                f"TPS: {tps_stats['avg']:.2f}\n"
            )

        except Exception as e:
            print(f"  ✗ 오류: {e}\n")

        # 총 타임아웃 체크
        elapsed = time.perf_counter() - test_start_time
        if elapsed > config.total_timeout:
            print(f"전체 타임아웃({config.total_timeout}초) 초과. 테스트 중단.")
            break

    # 결과 저장
    test_end_time = time.perf_counter()
    test_overview = (
        f"테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"테스트 소요 시간: {test_end_time - test_start_time:.2f}초\n"
        f"동시성 수준: {config.concurrency_levels}\n"
        f"테스트 프롬프트: {config.test_prompt}\n"
        f"모델: {config.model_name}"
    )

    result_path = formatter.save_results(all_metrics, test_overview)
    formatter.print_console_table(all_metrics)

    print(f"결과가 저장되었습니다: {result_path}\n")


def main() -> None:
    """메인 진입점."""
    config = parse_args()
    run_load_test(config)


if __name__ == "__main__":
    main()
