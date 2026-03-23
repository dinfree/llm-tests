import base64
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

"""OpenAI 호환 프록시를 대상으로 text/vision/embedding(RAG) 테스트를 수행하는 CLI."""

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGE_PATH = ROOT_DIR / "sample.jpg"
TEXT_MODEL_NAME = "text"
VISION_MODEL_NAME = "vision"
EMBEDDING_MODEL_NAME = "text-embedding-nomic-embed-text-v1.5"


class ConfigError(Exception):
    """필수 설정값(.env)이 누락되었을 때 발생하는 예외."""

    pass


def get_required_env(key: str) -> str:
    """비어 있지 않은 환경 변수를 반환하고, 누락 시 예외를 발생시킨다."""

    value = os.getenv(key, "").strip()
    if not value:
        raise ConfigError(f"환경 변수 {key} 가 설정되어 있지 않습니다.")
    return value


def ask_input(label: str, allow_empty: bool = False) -> str:
    """사용자 입력을 받아 반환한다. allow_empty가 False면 빈 입력을 재요청한다."""

    while True:
        value = input(label).strip()
        if value or allow_empty:
            return value
        print("값을 입력해 주세요.")


def create_chat_model(model_name: str) -> ChatOpenAI:
    """텍스트/비전 테스트에 사용하는 Chat 모델 클라이언트를 생성한다."""

    api_key = get_required_env("API_KEY")
    base_url = get_required_env("BASE_URL")
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        model_kwargs={"stream_options": {"include_usage": True}},
    )


def create_embeddings_model(model_name: str) -> OpenAIEmbeddings:
    """RAG 임베딩 생성에 사용하는 임베딩 클라이언트를 생성한다."""

    api_key = get_required_env("API_KEY")
    base_url = get_required_env("BASE_URL")
    return OpenAIEmbeddings(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        # OpenAI 호환 서버에서 토큰 배열 입력 비호환 이슈를 피하기 위해
        # raw text를 그대로 전달하는 경로를 사용한다.
        check_embedding_ctx_length=False,
    )


def encode_image_to_data_url(image_path: Path) -> str:
    """이미지 파일을 base64 Data URL로 인코딩한다."""

    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


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


def stream_response(llm: ChatOpenAI, messages: list[Any]) -> dict[str, float | int | None]:
    """모델 응답을 스트리밍 출력하고 핵심 성능 지표를 계산한다."""

    start_time = time.perf_counter()
    first_token_time: float | None = None
    output_parts: list[str] = []
    usage: dict[str, int] = {}

    print("\n[응답]\n", end="", flush=True)

    for chunk in llm.stream(messages):
        # 청크별 텍스트를 즉시 출력해 사용자 체감 지연을 줄인다.
        chunk_text = _extract_text(chunk)
        if chunk_text and first_token_time is None:
            first_token_time = time.perf_counter()

        if chunk_text:
            print(chunk_text, end="", flush=True)
            output_parts.append(chunk_text)

        current_usage = _extract_usage(chunk)
        if current_usage:
            usage = current_usage

    end_time = time.perf_counter()
    print("\n")

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

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft": ttft,
        "tps": tps,
    }


def print_metrics(metrics: dict[str, float | int | None]) -> None:
    """응답 후 토큰/지연 관련 메트릭을 보기 좋은 형식으로 출력한다."""

    input_tokens = int(metrics["input_tokens"] or 0)
    output_tokens = int(metrics["output_tokens"] or 0)
    ttft = metrics["ttft"]
    tps = float(metrics["tps"] or 0.0)

    print("[메트릭]")
    print(f"- 입력 토큰 수: {input_tokens}")
    print(f"- 출력 토큰 수: {output_tokens}")

    if ttft is None:
        print("- TTFT: 측정 불가")
    else:
        print(f"- TTFT: {ttft:.3f}초")

    print(f"- TPS: {tps:.2f}")


def run_text_test() -> None:
    """메뉴 1: 텍스트 모델 단일 프롬프트 테스트."""

    prompt = ask_input("프롬프트를 입력하세요: ")

    llm = create_chat_model(TEXT_MODEL_NAME)
    messages = [HumanMessage(content=prompt)]
    metrics = stream_response(llm, messages)
    print_metrics(metrics)


def run_vision_test() -> None:
    """메뉴 2: sample.jpg + 프롬프트로 비전 모델 테스트."""

    if not SAMPLE_IMAGE_PATH.exists():
        print(f"샘플 이미지 파일이 없습니다: {SAMPLE_IMAGE_PATH.name}")
        return

    prompt = ask_input("이미지에 대한 프롬프트를 입력하세요: ")

    image_data_url = encode_image_to_data_url(SAMPLE_IMAGE_PATH)
    llm = create_chat_model(VISION_MODEL_NAME)

    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        )
    ]

    metrics = stream_response(llm, messages)
    print_metrics(metrics)


def load_documents(file_path: Path) -> list[Any]:
    """입력 파일 확장자에 맞춰 문서를 로딩한다(txt/pdf)."""

    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        loader = TextLoader(str(file_path), autodetect_encoding=True)
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. txt 또는 pdf 파일을 사용하세요.")

    return loader.load()


def run_embedding_test() -> None:
    """메뉴 3: 문서 임베딩 + FAISS 검색 기반 RAG 질의응답 테스트."""

    file_name = ask_input("첨부할 파일명을 입력하세요 (예: sample.txt, sample.pdf): ")
    file_path = (ROOT_DIR / file_name).resolve()

    if not file_path.exists() or not file_path.is_file():
        print("입력한 파일을 찾을 수 없습니다.")
        return

    question = ask_input("질문(프롬프트)을 입력하세요: ")

    docs = load_documents(file_path)
    # 검색 품질과 처리 비용의 균형을 위해 문서를 청크 단위로 분할한다.
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    split_docs = splitter.split_documents(docs)

    if not split_docs:
        print("문서에서 처리할 텍스트를 찾을 수 없습니다.")
        return

    embeddings = create_embeddings_model(EMBEDDING_MODEL_NAME)
    # 문서 청크를 벡터화해 FAISS 인덱스를 구성한다.
    vector_store = FAISS.from_documents(split_docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content.strip() for doc in retrieved_docs if doc.page_content.strip())

    if not context:
        print("검색된 문맥이 없어 답변을 생성할 수 없습니다.")
        return

    llm = create_chat_model(TEXT_MODEL_NAME)
    messages = [
        SystemMessage(
            content=(
                "당신은 문서 기반 질의응답 도우미입니다. "
                "제공된 문맥 안에서만 답변하고, 문맥이 부족하면 부족하다고 명시하세요."
            )
        ),
        HumanMessage(
            content=(
                f"질문:\n{question}\n\n"
                f"문맥:\n{context}\n\n"
                "위 문맥을 바탕으로 한국어로 답변해 주세요."
            )
        ),
    ]

    print(f"\n[검색된 문서 조각 수] {len(retrieved_docs)}")
    metrics = stream_response(llm, messages)
    print_metrics(metrics)


def print_menu() -> None:
    """CLI 메인 메뉴를 출력한다."""

    print("\n=== LLM 테스트 메뉴 ===")
    print("1. text 모델 테스트")
    print("2. image(vision) 모델 테스트")
    print("3. embedding 모델 테스트 (RAG)")
    print("4. 종료")


def main() -> None:
    """메뉴 루프를 실행하며 선택한 테스트를 호출한다."""

    print("LLM 테스트 프로그램을 시작합니다.")

    while True:
        try:
            print_menu()
            choice = ask_input("메뉴 번호를 선택하세요: ")

            if choice == "1":
                run_text_test()
            elif choice == "2":
                run_vision_test()
            elif choice == "3":
                run_embedding_test()
            elif choice == "4":
                print("프로그램을 종료합니다.")
                break
            else:
                print("유효한 메뉴 번호를 입력해 주세요.")

        except KeyboardInterrupt:
            print("\n사용자 요청으로 종료합니다.")
            break
        except ConfigError as e:
            print(f"설정 오류: {e}")
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
