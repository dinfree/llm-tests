"""부하 테스트 유틸리티: 메트릭 수집, 통계, 출력."""

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class RequestMetrics:
    """개별 요청의 메트릭."""

    concurrency: int
    success: bool
    ttft: Optional[float] = None
    tps: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    error_message: Optional[str] = None


@dataclass
class ConcurrencyMetrics:
    """동시성 수준별 메트릭."""

    concurrency: int
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    ttft_values: list[float] = field(default_factory=list)
    tps_values: list[float] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_request(self, metrics: RequestMetrics) -> None:
        """개별 요청 메트릭을 추가한다."""
        self.total_requests += 1
        if metrics.success:
            self.successful_requests += 1
            if metrics.ttft is not None:
                self.ttft_values.append(metrics.ttft)
            if metrics.tps is not None:
                self.tps_values.append(metrics.tps)
            self.total_input_tokens += metrics.input_tokens
            self.total_output_tokens += metrics.output_tokens
        else:
            self.failed_requests += 1

    def get_ttft_stats(self) -> dict[str, float]:
        """TTFT 통계를 반환한다."""
        if not self.ttft_values:
            return {
                "count": 0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "stdev": 0.0,
            }
        return {
            "count": len(self.ttft_values),
            "avg": statistics.mean(self.ttft_values),
            "min": min(self.ttft_values),
            "max": max(self.ttft_values),
            "stdev": (
                statistics.stdev(self.ttft_values)
                if len(self.ttft_values) > 1
                else 0.0
            ),
        }

    def get_tps_stats(self) -> dict[str, float]:
        """TPS 통계를 반환한다."""
        if not self.tps_values:
            return {
                "count": 0,
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "stdev": 0.0,
            }
        return {
            "count": len(self.tps_values),
            "avg": statistics.mean(self.tps_values),
            "min": min(self.tps_values),
            "max": max(self.tps_values),
            "stdev": (
                statistics.stdev(self.tps_values)
                if len(self.tps_values) > 1
                else 0.0
            ),
        }

    def get_success_rate(self) -> float:
        """성공률을 반환한다 (0~100)."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


class ResultFormatter:
    """테스트 결과를 콘솔 및 파일에 출력한다."""

    def __init__(self, result_dir: str, filename_prefix: str):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_result_filepath(self) -> Path:
        """결과 파일 경로를 반환한다."""
        return self.result_dir / f"{self.filename_prefix}_{self.timestamp}.txt"

    def format_table_header(self) -> str:
        """테이블 헤더를 반환한다."""
        return (
            f"{'동시성':>10} | "
            f"{'요청수':>8} | "
            f"{'성공':>6} | "
            f"{'실패':>6} | "
            f"{'성공률':>7} | "
            f"{'TTFT평균(ms)':>13} | "
            f"{'TTFT최소(ms)':>13} | "
            f"{'TTFT최대(ms)':>13} | "
            f"{'TPS평균':>10} | "
            f"{'TPS최소':>10} | "
            f"{'TPS최대':>10}"
        )

    def format_table_row(self, metrics: ConcurrencyMetrics) -> str:
        """메트릭을 테이블 행으로 포매팅한다."""
        ttft_stats = metrics.get_ttft_stats()
        tps_stats = metrics.get_tps_stats()
        success_rate = metrics.get_success_rate()

        return (
            f"{metrics.concurrency:>10} | "
            f"{metrics.total_requests:>8} | "
            f"{metrics.successful_requests:>6} | "
            f"{metrics.failed_requests:>6} | "
            f"{success_rate:>6.1f}% | "
            f"{ttft_stats['avg'] * 1000:>13.2f} | "
            f"{ttft_stats['min'] * 1000:>13.2f} | "
            f"{ttft_stats['max'] * 1000:>13.2f} | "
            f"{tps_stats['avg']:>10.2f} | "
            f"{tps_stats['min']:>10.2f} | "
            f"{tps_stats['max']:>10.2f}"
        )

    def format_separator(self, length: int = 150) -> str:
        """테이블 구분선을 반환한다."""
        return "-" * length

    def print_console_table(
        self, all_metrics: list[ConcurrencyMetrics]
    ) -> None:
        """콘솔에 테이블을 출력한다."""
        print("\n" + self.format_separator())
        print(self.format_table_header())
        print(self.format_separator())
        for metrics in all_metrics:
            print(self.format_table_row(metrics))
        print(self.format_separator() + "\n")

    def create_ascii_chart(
        self, all_metrics: list[ConcurrencyMetrics], metric_name: str, values: list[float]
    ) -> str:
        """아스키 차트를 생성한다."""
        if not values:
            return f"[{metric_name}] 데이터가 없습니다.\n"
        
        valid_values = [v for v in values if v > 0]
        if not valid_values:
            return f"[{metric_name}] 유효한 데이터가 없습니다.\n"

        max_value = max(valid_values)
        chart_width = 60
        title = f"[{metric_name} 추이]"
        result = title + "\n"

        for metrics, value in zip(all_metrics, values):
            if value > 0:
                bar_length = int((value / max_value) * chart_width)
            else:
                bar_length = 0
            bar = "█" * bar_length
            result += f"{metrics.concurrency:>3}명 | {bar:<{chart_width}} | {value:.2f}\n"

        return result + "\n"

    def save_results(
        self, all_metrics: list[ConcurrencyMetrics], test_overview: str
    ) -> Path:
        """결과를 파일에 저장한다."""
        output_path = self.get_result_filepath()

        ttft_values = []
        tps_values = []
        
        for m in all_metrics:
            ttft_stats = m.get_ttft_stats()
            tps_stats = m.get_tps_stats()
            ttft_values.append(ttft_stats.get("avg", 0.0))
            tps_values.append(tps_stats.get("avg", 0.0))

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 150 + "\n")
            f.write("LLM Proxy 부하 테스트 결과\n")
            f.write("=" * 150 + "\n\n")

            # 테스트 개요
            f.write(test_overview + "\n\n")

            # 메인 테이블
            f.write(self.format_separator() + "\n")
            f.write(self.format_table_header() + "\n")
            f.write(self.format_separator() + "\n")
            for metrics in all_metrics:
                f.write(self.format_table_row(metrics) + "\n")
            f.write(self.format_separator() + "\n\n")

            # 아스키 차트
            f.write(self.create_ascii_chart(all_metrics, "TTFT 평균 (초)", ttft_values))
            f.write(self.create_ascii_chart(all_metrics, "TPS 평균 (토큰/초)", tps_values))

            # 상세 통계
            f.write("=" * 150 + "\n")
            f.write("상세 통계\n")
            f.write("=" * 150 + "\n\n")
            for metrics in all_metrics:
                ttft_stats = metrics.get_ttft_stats()
                tps_stats = metrics.get_tps_stats()
                f.write(f"[동시성: {metrics.concurrency}명]\n")
                f.write(f"- 총 요청: {metrics.total_requests}\n")
                f.write(f"- 성공: {metrics.successful_requests}\n")
                f.write(f"- 실패: {metrics.failed_requests}\n")
                f.write(f"- 성공률: {metrics.get_success_rate():.1f}%\n")
                f.write(
                    f"- TTFT: 평균={ttft_stats['avg']:.3f}초, "
                    f"최소={ttft_stats['min']:.3f}초, "
                    f"최대={ttft_stats['max']:.3f}초, "
                    f"표준편차={ttft_stats['stdev']:.3f}초\n"
                )
                f.write(
                    f"- TPS: 평균={tps_stats['avg']:.2f}, "
                    f"최소={tps_stats['min']:.2f}, "
                    f"최대={tps_stats['max']:.2f}, "
                    f"표준편차={tps_stats['stdev']:.2f}\n"
                )
                f.write(
                    f"- 토큰: 입력={metrics.total_input_tokens}, "
                    f"출력={metrics.total_output_tokens}\n"
                )
                f.write("\n")

        return output_path
