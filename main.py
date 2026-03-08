#!/usr/bin/env python3
"""
Multi-model performance test tool
- Single test: one streaming request per selected model
- Multi test: concurrent streaming requests per selected model
- Results stored in SQLite DB, charts saved as PNG
"""

import asyncio
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
import uuid
from datetime import datetime

import aiohttp
import matplotlib
import requests
import tiktoken
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Environment
load_dotenv()

API_KEY = (
    os.environ.get("API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("VLLM_API_KEY")
)
if not API_KEY:
    raise ValueError("API key is missing. Set API_KEY (or OPENAI_API_KEY / VLLM_API_KEY) in .env")

BASE_URL = os.environ.get("BASE_URL")
if not BASE_URL:
    raise ValueError("BASE_URL is missing in .env")

PROMPT = os.environ.get("BASE_PROMPT", "Hello, how are you?")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "180"))

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "results.db")
RESULT_DIR = os.path.join(BASE_DIR, "result")
PREV_RESULT_DIR = os.path.join(BASE_DIR, "prev_result")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

REQUIRED_COLUMNS = {
    "run_id": "TEXT",
    "total_requests": "INTEGER DEFAULT 1",
    "success_requests": "INTEGER DEFAULT 0",
    "error_requests": "INTEGER DEFAULT 0",
    "success_rate": "REAL DEFAULT 0",
    "ttft_p95": "REAL DEFAULT 0",
    "notes": "TEXT",
}


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def pctl(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(round((percentile / 100) * (len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


def archive_existing_result_images() -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PREV_RESULT_DIR, exist_ok=True)
    png_files = [f for f in os.listdir(RESULT_DIR) if f.lower().endswith(".png")]
    if not png_files:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for file_name in png_files:
        src = os.path.join(RESULT_DIR, file_name)
        dst_name = f"{timestamp}_{file_name}"
        dst = os.path.join(PREV_RESULT_DIR, dst_name)
        shutil.move(src, dst)
    logger.info("Archived %d previous chart images to prev_result/", len(png_files))


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS test_results (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            run_id           TEXT,
            test_type        TEXT,
            model_name       TEXT,
            concurrency      INTEGER DEFAULT 1,
            total_requests   INTEGER DEFAULT 1,
            success_requests INTEGER DEFAULT 0,
            error_requests   INTEGER DEFAULT 0,
            success_rate     REAL DEFAULT 0,
            total_tokens     INTEGER DEFAULT 0,
            total_duration   REAL DEFAULT 0,
            ttft             REAL DEFAULT 0,
            ttft_p95         REAL DEFAULT 0,
            tps              REAL DEFAULT 0,
            notes            TEXT
        )
        """
    )

    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(test_results)").fetchall()
    }
    for col_name, col_type in REQUIRED_COLUMNS.items():
        if col_name not in existing_columns:
            conn.execute(f"ALTER TABLE test_results ADD COLUMN {col_name} {col_type}")

    conn.execute("UPDATE test_results SET run_id = 'legacy' WHERE run_id IS NULL")
    conn.commit()
    conn.close()


def save_result(
    run_id: str,
    test_type: str,
    model_name: str,
    concurrency: int,
    total_requests: int,
    success_requests: int,
    total_tokens: int,
    total_duration: float,
    ttft: float,
    ttft_p95: float,
    tps: float,
    notes: str = "",
) -> None:
    error_requests = max(total_requests - success_requests, 0)
    success_rate = (success_requests / total_requests * 100.0) if total_requests > 0 else 0.0
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO test_results (
            timestamp, run_id, test_type, model_name, concurrency,
            total_requests, success_requests, error_requests, success_rate,
            total_tokens, total_duration, ttft, ttft_p95, tps, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            run_id,
            test_type,
            model_name,
            concurrency,
            total_requests,
            success_requests,
            error_requests,
            success_rate,
            total_tokens,
            total_duration,
            ttft,
            ttft_p95,
            tps,
            notes,
        ),
    )
    conn.commit()
    conn.close()


def fetch_models() -> list[str]:
    url = f"{BASE_URL}/models"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        models = [m["id"] for m in resp.json().get("data", []) if "id" in m]
        if not models:
            logger.error("No models found on server.")
            sys.exit(1)
        return models
    except Exception as exc:
        logger.error("Failed to fetch model names: %s", exc)
        sys.exit(1)


def choose_models(models: list[str]) -> list[str]:
    print("\nAvailable models")
    print("-" * 60)
    for idx, model in enumerate(models, start=1):
        print(f"  {idx:>2}. {model}")
    print("-" * 60)

    raw = input("Select models (all or comma index, example: 1,3): ").strip().lower()
    if raw == "all":
        return models

    try:
        indexes = [int(item.strip()) for item in raw.split(",") if item.strip()]
        selected = [models[i - 1] for i in indexes if 1 <= i <= len(models)]
    except ValueError:
        selected = []

    unique_selected = list(dict.fromkeys(selected))
    if not unique_selected:
        print("Invalid selection. Please try again.")
    return unique_selected


async def send_stream_request(
    session: aiohttp.ClientSession,
    model_name: str,
    request_id: int,
    stream_to_stdout: bool = False,
) -> dict:
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
    }

    req_start = time.time()
    first_token_latency = None
    tokens_generated = 0

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                body = await response.text()
                return {
                    "ok": False,
                    "error": f"HTTP {response.status}: {body[:200]}",
                    "tokens": 0,
                    "ttft": 0.0,
                    "duration": time.time() - req_start,
                }

            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue

                if first_token_latency is None:
                    first_token_latency = time.time() - req_start

                try:
                    data = json.loads(line[6:])
                    content = data["choices"][0]["delta"].get("content", "")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue

                if content:
                    tokens_generated += len(encoding.encode(content))
                    if stream_to_stdout:
                        print(content, end="", flush=True)

            duration = time.time() - req_start
            return {
                "ok": True,
                "error": "",
                "tokens": tokens_generated,
                "ttft": first_token_latency or 0.0,
                "duration": duration,
            }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"req#{request_id} failed: {exc}",
            "tokens": 0,
            "ttft": 0.0,
            "duration": time.time() - req_start,
        }


def _save_single_chart(model_name: str, total_tokens: int, total_duration: float, ttft: float, tps: float) -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    safe = sanitize_name(model_name)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = ["Total Tokens", "Duration (s)", "TTFT (s)", "TPS"]
    values = [total_tokens, total_duration, ttft, tps]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for ax, label, val, color in zip(axes, metrics, values, colors):
        ax.bar([label], [val], color=color, width=0.5)
        ax.set_title(label, fontsize=12)
        fmt = f"{val:.2f}" if isinstance(val, float) else str(val)
        ax.text(0, val, fmt, ha="center", va="bottom", fontsize=12, fontweight="bold")

    fig.suptitle(f"Single Test - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, f"{safe}_single_1.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Chart saved: %s", path)


def _save_multi_chart(model_name: str, all_stats: list[dict]) -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    safe = sanitize_name(model_name)

    concurrencies = [s["concurrency"] for s in all_stats]
    tps_vals = [s["system_tps"] for s in all_stats]
    ttft_p95_vals = [s["ttft_p95"] for s in all_stats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(concurrencies, tps_vals, marker="o", color="#1f77b4", linewidth=2, markersize=8)
    ax1.set_xlabel("Concurrent Requests", fontsize=12)
    ax1.set_ylabel("System Throughput (tokens/s)", fontsize=12)
    ax1.set_title("Throughput vs Concurrency", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(concurrencies, ttft_p95_vals, marker="s", color="#ff7f0e", linewidth=2, markersize=8)
    ax2.set_xlabel("Concurrent Requests", fontsize=12)
    ax2.set_ylabel("TTFT p95 (s)", fontsize=12)
    ax2.set_title("TTFT p95 vs Concurrency", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Multi Test - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    conc_str = "_".join(str(c) for c in concurrencies)
    path = os.path.join(RESULT_DIR, f"{safe}_multi_{conc_str}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Chart saved: %s", path)


async def single_test(model_name: str, run_id: str) -> None:
    logger.info("Starting single test | run_id=%s | model=%s", run_id, model_name)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        result = await send_stream_request(session, model_name, request_id=1, stream_to_stdout=True)

    print("\n")
    if not result["ok"]:
        logger.error("Single test failed | model=%s | error=%s", model_name, result["error"])
        save_result(
            run_id=run_id,
            test_type="single",
            model_name=model_name,
            concurrency=1,
            total_requests=1,
            success_requests=0,
            total_tokens=0,
            total_duration=result["duration"],
            ttft=0.0,
            ttft_p95=0.0,
            tps=0.0,
            notes=result["error"],
        )
        return

    tps = result["tokens"] / result["duration"] if result["duration"] > 0 else 0.0
    print("=" * 80)
    print("Single Test Results")
    print("=" * 80)
    print(f"Model               : {model_name}")
    print(f"Total Tokens        : {result['tokens']}")
    print(f"Total Duration (s)  : {result['duration']:.2f}")
    print(f"TTFT (s)            : {result['ttft']:.3f}")
    print(f"Throughput TPS      : {tps:.2f}")
    print("=" * 80)

    save_result(
        run_id=run_id,
        test_type="single",
        model_name=model_name,
        concurrency=1,
        total_requests=1,
        success_requests=1,
        total_tokens=result["tokens"],
        total_duration=result["duration"],
        ttft=result["ttft"],
        ttft_p95=result["ttft"],
        tps=tps,
        notes="",
    )
    _save_single_chart(model_name, result["tokens"], result["duration"], result["ttft"], tps)


async def multi_test(model_name: str, concurrency_levels: list[int], run_id: str) -> None:
    logger.info("Starting multi test | run_id=%s | model=%s", run_id, model_name)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    all_stats: list[dict] = []
    for idx, concurrency in enumerate(concurrency_levels, start=1):
        logger.info("[%d/%d] model=%s concurrency=%d", idx, len(concurrency_levels), model_name, concurrency)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                send_stream_request(session, model_name, request_id=req_id, stream_to_stdout=False)
                for req_id in range(1, concurrency + 1)
            ]
            results = await asyncio.gather(*tasks)

        success_results = [r for r in results if r["ok"]]
        ttft_vals = [r["ttft"] for r in success_results]
        duration_vals = [r["duration"] for r in success_results]
        total_tokens = sum(r["tokens"] for r in success_results)

        total_duration = max(duration_vals) if duration_vals else 0.0
        avg_ttft = (sum(ttft_vals) / len(ttft_vals)) if ttft_vals else 0.0
        ttft_p95 = pctl(ttft_vals, 95)
        system_tps = total_tokens / total_duration if total_duration > 0 else 0.0
        success_requests = len(success_results)
        error_requests = concurrency - success_requests
        success_rate = (success_requests / concurrency * 100.0) if concurrency > 0 else 0.0

        error_sample = ""
        for item in results:
            if not item["ok"]:
                error_sample = item["error"]
                break

        stats = {
            "concurrency": concurrency,
            "total_tokens": total_tokens,
            "total_duration": round(total_duration, 2),
            "avg_ttft": round(avg_ttft, 3),
            "ttft_p95": round(ttft_p95, 3),
            "system_tps": round(system_tps, 2),
            "success": success_requests,
            "error": error_requests,
            "success_rate": round(success_rate, 1),
        }
        all_stats.append(stats)

        save_result(
            run_id=run_id,
            test_type="multi",
            model_name=model_name,
            concurrency=concurrency,
            total_requests=concurrency,
            success_requests=success_requests,
            total_tokens=total_tokens,
            total_duration=total_duration,
            ttft=avg_ttft,
            ttft_p95=ttft_p95,
            tps=system_tps,
            notes=error_sample,
        )

        logger.info(
            "model=%s conc=%d success=%d/%d tps=%.2f ttft_p95=%.3f",
            model_name,
            concurrency,
            success_requests,
            concurrency,
            system_tps,
            ttft_p95,
        )

        if idx < len(concurrency_levels):
            await asyncio.sleep(2)

    print("\n" + "=" * 110)
    print(f"Multi Test Results - Model: {model_name}")
    print("=" * 110)
    print(
        f"{'Conc':>6} | {'Tokens':>8} | {'Dur(s)':>8} | {'TTFT Avg':>9} | "
        f"{'TTFT p95':>9} | {'TPS':>10} | {'Success':>9} | {'Rate':>7}"
    )
    print("-" * 110)
    for s in all_stats:
        print(
            f"{s['concurrency']:>6} | {s['total_tokens']:>8} | {s['total_duration']:>8.2f} | "
            f"{s['avg_ttft']:>9.3f} | {s['ttft_p95']:>9.3f} | {s['system_tps']:>10.2f} | "
            f"{s['success']:>4}/{(s['success'] + s['error']):<4} | {s['success_rate']:>6.1f}%"
        )
    print("=" * 110)

    _save_multi_chart(model_name, all_stats)


def view_results() -> None:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """
        SELECT id, timestamp, run_id, test_type, model_name, concurrency,
               total_requests, success_requests, success_rate,
               total_tokens, total_duration, ttft, ttft_p95, tps
        FROM test_results
        ORDER BY id DESC
        """
    ).fetchall()
    conn.close()

    if not rows:
        print("\nNo results found in database.")
        return

    print("\n" + "=" * 160)
    print("Test Results")
    print("=" * 160)
    print(
        f"{'ID':>4} | {'Timestamp':>19} | {'Run ID':>10} | {'Type':>6} | {'Model':>34} | "
        f"{'Conc':>4} | {'Req':>7} | {'Succ':>7} | {'Rate':>6} | {'Tokens':>8} | "
        f"{'Dur(s)':>7} | {'TTFT':>7} | {'P95':>7} | {'TPS':>8}"
    )
    print("-" * 160)
    for row in rows:
        (
            rid,
            ts,
            run_id,
            test_type,
            model_name,
            concurrency,
            total_requests,
            success_requests,
            success_rate,
            total_tokens,
            total_duration,
            ttft,
            ttft_p95,
            tps,
        ) = row
        ts_short = (ts or "")[:19]
        model_short = model_name[-34:] if len(model_name) > 34 else model_name
        run_short = (run_id or "-")[:10]

        print(
            f"{rid:>4} | {ts_short:>19} | {run_short:>10} | {test_type:>6} | {model_short:>34} | "
            f"{int(concurrency):>4} | {int(total_requests):>7} | {int(success_requests):>7} | "
            f"{float(success_rate):>5.1f}% | {int(total_tokens):>8} | {float(total_duration):>7.2f} | "
            f"{float(ttft):>7.3f} | {float(ttft_p95):>7.3f} | {float(tps):>8.2f}"
        )
    print("=" * 160)


def parse_concurrency_levels(raw: str) -> list[int]:
    levels = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            val = int(chunk)
        except ValueError:
            return []
        if val <= 0:
            return []
        levels.append(val)
    return sorted(list(dict.fromkeys(levels)))


def main() -> None:
    init_db()
    archive_existing_result_images()

    print("=" * 60)
    print("LLM Performance Test Tool")
    print("=" * 60)
    print("Fetching model list from server ...")
    models = fetch_models()
    print(f"Server  : {BASE_URL}")
    print(f"Models  : {len(models)} found")
    print("=" * 60)

    while True:
        print("\nMenu")
        print("  1. Single test")
        print("  2. Multi test")
        print("  3. View results")
        print("  4. Exit")

        choice = input("Select [1-4]: ").strip()

        if choice == "1":
            selected_models = choose_models(models)
            if not selected_models:
                continue

            run_id = uuid.uuid4().hex[:10]
            print(f"\nStarting single tests | run_id={run_id} | models={len(selected_models)}")
            for model_name in selected_models:
                print(f"\n[Single] model={model_name}")
                asyncio.run(single_test(model_name, run_id=run_id))

        elif choice == "2":
            selected_models = choose_models(models)
            if not selected_models:
                continue

            raw = input("Enter concurrency levels (example: 10, 100, 200): ").strip()
            levels = parse_concurrency_levels(raw)
            if not levels:
                print("Invalid input. Please use comma-separated positive integers.")
                continue

            run_id = uuid.uuid4().hex[:10]
            print(
                f"\nStarting multi tests | run_id={run_id} | models={len(selected_models)} | "
                f"concurrency={levels}"
            )
            for model_name in selected_models:
                print(f"\n[Multi] model={model_name}")
                asyncio.run(multi_test(model_name, levels, run_id=run_id))

        elif choice == "3":
            view_results()

        elif choice == "4":
            print("Goodbye")
            break

        else:
            print("Invalid selection. Choose a number between 1 and 4.")


if __name__ == "__main__":
    main()
