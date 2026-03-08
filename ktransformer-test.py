import time
import random
import string
import os
import sys
import requests
import logging
from openai import OpenAI
import tiktoken  # 정확한 토큰 커팅을 위해 필수
from dotenv import load_dotenv
load_dotenv()

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==========================================
# [설정 영역]
# ==========================================
BASE_URL = os.environ.get("BASE_URL")
API_KEY = "ktransformers"


# 테스트 반복 횟수 (신뢰도 향상을 위해 3회 반복)
ITERATIONS = 3

TEST_CASES = [
    {"label": "hi (2)",    "target_tokens": 2,    "output_len": 10},
    {"label": "1K (969)",  "target_tokens": 969,  "output_len": 300},
    {"label": "2K (1930)", "target_tokens": 1930, "output_len": 300},
    {"label": "4K (3846)", "target_tokens": 3846, "output_len": 300},
    {"label": "8K (7678)", "target_tokens": 7678, "output_len": 300},
]

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
# 범용적으로 Qwen 등 최신 모델과 토큰 수가 거의 유사한 cl100k_base 사용
tokenizer = tiktoken.get_encoding("cl100k_base") 

# ==========================================
# [함수 영역]
# ==========================================
def fetch_model_name() -> str:
    """GET /v1/models and return the first model id."""
    url = f"{BASE_URL}/models"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
        logger.error("No models found on server.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to fetch model name: {e}")
        sys.exit(1)

def generate_exact_prompt(target_tokens):
    """
    tiktoken을 사용하여 정확히 target_tokens 길이에 맞게 프롬프트를 자릅니다.
    """
    if target_tokens < 10:
        return "hi " * (target_tokens // 2 + 1)
        
    # 1. 캐시 무력화를 위한 무작위 난수(Nonce) 생성
    nonce = "".join(random.choices(string.ascii_letters + string.digits, k=16))
    prefix = f"System Nonce: {nonce}\n\n"
    prefix_tokens = tokenizer.encode(prefix)
    
    if target_tokens <= len(prefix_tokens):
        return prefix
        
    remaining_target = target_tokens - len(prefix_tokens)
    
    # 2. 반복할 베이스 텍스트
    base_text = "The deepseek coder is a powerful model for coding tasks and general reasoning benchmarks. It efficiently handles context and generation tasks with mixture of experts architecture. "
    base_tokens = tokenizer.encode(base_text)
    
    # 3. 필요한 토큰 수만큼 정확하게 리스트 슬라이싱
    repeat_count = (remaining_target // len(base_tokens)) + 1
    extended_tokens = (base_tokens * repeat_count)[:remaining_target]
    
    # 4. 토큰을 다시 문자열로 디코딩
    final_prompt = tokenizer.decode(prefix_tokens + extended_tokens)
    return final_prompt

def run_single_benchmark(case):
    prompt = generate_exact_prompt(case["target_tokens"])
    
    try:
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=case["output_len"],
            stream=True,
            temperature=0.0,
            # [핵심] 서버에서 실제 연산한 토큰 수를 응답 마지막에 포함시킴
            stream_options={"include_usage": True} 
        )

        first_token_time = None
        token_count = 0
        actual_prompt_tokens = case["target_tokens"] # 실패 시 기본값
        
        for chunk in response:
            # usage 정보가 있는 마지막 청크 처리
            if chunk.usage:
                actual_prompt_tokens = chunk.usage.prompt_tokens
                continue
                
            if not chunk.choices:
                continue

            if chunk.choices[0].delta.content is not None:
                if first_token_time is None:
                    first_token_time = time.time()
                token_count += 1
        
        end_time = time.time()

        if first_token_time:
            prefill_duration = first_token_time - start_time
        else:
            prefill_duration = 0

        decode_duration = end_time - first_token_time if first_token_time else 0

        # [수정됨] 분모를 가상의 target_tokens가 아닌, 서버가 반환한 실제(actual) 토큰 수로 계산
        prefill_speed = actual_prompt_tokens / prefill_duration if prefill_duration > 0 else 0
        decode_speed = (token_count - 1) / decode_duration if (decode_duration > 0 and token_count > 1) else 0

        return prefill_speed, decode_speed, actual_prompt_tokens

    except Exception as e:
        return None, None, f"Error: {str(e)[:50]}..."

# ==========================================
# [실행 영역]
# ==========================================
MODEL_NAME = fetch_model_name()
print(f"\n🚀 Starting Exact Benchmark for {MODEL_NAME}")
print(f"📡 Server: {BASE_URL} | 🔄 Iterations: {ITERATIONS}")
print("-" * 90)
print(f"{'Target Size':<15} | {'Avg Prefill (t/s)':<18} | {'Avg Decode (t/s)':<18} | {'Actual Server Tokens'}")
print("-" * 90)

# Warm-up
print(f"{'Warm-up...':<15} | {'...':<18} | {'...':<18} | Loading...")
try:
    client.chat.completions.create(
        model=MODEL_NAME, messages=[{"role": "user", "content": "hi"}], max_tokens=1
    )
except:
    pass

for case in TEST_CASES:
    prefill_speeds = []
    decode_speeds = []
    actual_tokens_list = []
    error_msg = ""

    for _ in range(ITERATIONS):
        p_speed, d_speed, act_tokens = run_single_benchmark(case)
        
        if p_speed is None:
            error_msg = act_tokens
            break
            
        prefill_speeds.append(p_speed)
        decode_speeds.append(d_speed)
        actual_tokens_list.append(act_tokens)
        
        time.sleep(0.5) # 부하 분산을 위한 대기

    if error_msg:
        print(f"{case['label']:<15} | {'Fail':<18} | {'Fail':<18} | {error_msg}")
    else:
        avg_prefill = sum(prefill_speeds) / len(prefill_speeds)
        avg_decode = sum(decode_speeds) / len(decode_speeds)
        avg_act_tokens = int(sum(actual_tokens_list) / len(actual_tokens_list))
        
        print(f"{case['label']:<15} | {avg_prefill:<18.2f} | {avg_decode:<18.2f} | {avg_act_tokens} tokens")

print("-" * 90)
print("✅ Benchmark Completed.")