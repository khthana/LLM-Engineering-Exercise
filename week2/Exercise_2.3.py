# LangChain

import os
import re
from dataclasses import dataclass
from typing import Optional, Iterator
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


# เป็น Class สำหรับเก็บข้อมูลการตั้งค่าของแต่ละ Provider เช่น ชื่อ, คลาสที่ใช้ใน LangChain, โมเดลเริ่มต้น, ตัวแปรสภาพแวดล้อมสำหรับ API key, prefix ของ API key, และอื่นๆ
@dataclass
class ProviderConfig:
    name: str
    langchain_cls: str     # "openai" | "anthropic" | "gemini" | "ollama"
    default_model: str
    env_var: Optional[str] = None
    prefix: Optional[str] = None
    requires_key: bool = True
    base_url: Optional[str] = None

# เป็น Class สำหรับเก็บผลลัพธ์ของการเรียกใช้งานโมเดลจากแต่ละ Provider รวมถึงข้อมูลต่างๆ เช่น ชื่อ Provider, โมเดลที่ใช้, เนื้อหาคำตอบ, สถานะความสำเร็จ, ข้อความแสดงข้อผิดพลาด (ถ้ามี), และจำนวนโทเค็นที่ใช้ใน prompt และ completion
@dataclass
class ChatResult:
    provider: str
    model: str
    content: str
    success: bool = True
    error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None


# เป็นการสร้าง registry ของ Providers ที่รองรับ โดยแต่ละ Provider จะมีการกำหนดค่าต่างๆ ผ่าน ProviderConfig ซึ่งจะช่วยให้ LLMManager สามารถจัดการและเรียกใช้งานแต่ละ Provider ได้อย่างยืดหยุ่นและง่ายดาย
PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="OpenAI",
        langchain_cls="openai",
        env_var="OPENAI_API_KEY",
        prefix="sk-proj-",
        default_model="gpt-4o-mini",
    ),
    "claude": ProviderConfig(
        name="Anthropic Claude",
        langchain_cls="anthropic",
        env_var="ANTHROPIC_API_KEY",
        prefix="sk-ant-",
        default_model="claude-sonnet-4-6",
    ),
    "gemini": ProviderConfig(
        name="Google Gemini",
        langchain_cls="gemini",
        env_var="GOOGLE_API_KEY",
        prefix="AI",
        default_model="gemini-2.0-flash",
    ),
    "openrouter": ProviderConfig(
        name="OpenRouter",
        langchain_cls="openai",  # OpenRouter รองรับ OpenAI-compatible API
        env_var="OPENROUTER_API_KEY",
        prefix="sk-or-",
        default_model="openai/gpt-oss-120b:free",
        base_url="https://openrouter.ai/api/v1",
    ),
    "ollama": ProviderConfig(
        name="Ollama (Local)",
        langchain_cls="ollama",
        env_var=None,
        prefix=None,
        default_model="gemma4:e4b",
        requires_key=False,
        base_url="http://localhost:11434",
    ),
}


# =============================================================================
# LLMClient  (ใช้ LangChain)
# =============================================================================

# เป็น Class ที่ทำหน้าที่เป็นตัวกลางในการติดต่อกับแต่ละ Provider ผ่าน LangChain 
# โดยจะมีการสร้าง instance ของโมเดลที่เหมาะสมตามการตั้งค่าของแต่ละ Provider 
# และจัดการกับการส่งข้อความและรับผลลัพธ์จากโมเดล 
# รวมถึงการจัดการข้อผิดพลาดที่อาจเกิดขึ้นระหว่างการเรียกใช้งานโมเดล
class LLMClient:
    def __init__(self, provider_id: str, config: ProviderConfig, api_key: str):
        self.provider_id = provider_id
        self.config = config    # คือ ProviderConfig ที่เก็บข้อมูลการตั้งค่าของ Provider นั้นๆ

    # เป็นฟังก์ชันที่ใช้สร้าง instance ของโมเดลที่เหมาะสมตามการตั้งค่าของแต่ละ Provider 
    # โดยจะตรวจสอบว่า Provider นั้นใช้คลาสอะไรใน LangChain และส่งค่า API key 
    # และพารามิเตอร์อื่นๆ ที่จำเป็นในการสร้าง instance ของโมเดลนั้นๆ
    def _make_llm(self, model: str = None, reasoning_effort: str = None):
        m = model or self.config.default_model
        cls = self.config.langchain_cls

        if cls == "openai":
            kwargs = {"model": m, "api_key": self._api_key, "timeout": 120}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            if reasoning_effort is not None:
                kwargs["model_kwargs"] = {"reasoning_effort": reasoning_effort}
            return ChatOpenAI(**kwargs)

        elif cls == "anthropic":
            return ChatAnthropic(model=m, api_key=self._api_key, timeout=120)

        elif cls == "gemini":
            return ChatGoogleGenerativeAI(model=m, google_api_key=self._api_key)

        elif cls == "ollama":
            return ChatOllama(model=m, base_url=self.config.base_url or "http://localhost:11434")

        raise ValueError(f"Unknown langchain_cls: '{cls}'")

    # เป็นฟังก์ชันที่ใช้ส่งข้อความไปยังโมเดลของแต่ละ Provider และรับผลลัพธ์กลับมา 
    # โดยจะมีการจัดการข้อผิดพลาดที่อาจเกิดขึ้นระหว่างการเรียกใช้งานโมเดล
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        selected_model = model or self.config.default_model
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        # เป็นฟังก์ชันภายในที่ใช้แปลงผลลัพธ์ที่ได้รับจากโมเดลให้เป็นรูปแบบของ ChatResult 
        # ซึ่งจะรวมข้อมูลต่างๆ เช่น ชื่อ Provider, โมเดลที่ใช้, เนื้อหาคำตอบ, 
        # จำนวนโทเค็นที่ใช้ใน prompt และ completion รวมถึงข้อมูลการ cache ถ้ามี
        def _parse_result(response) -> ChatResult:
            usage = getattr(response, "usage_metadata", None) or {}
            cached = None
            details = usage.get("input_token_details", {})
            if details:
                cached = details.get("cache_read")
            return ChatResult(
                provider=self.provider_id,
                model=selected_model,
                content=response.content,
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
                cached_tokens=cached,
            )

        try:
            response = self._make_llm(model, reasoning_effort).invoke(messages)
            return _parse_result(response)
        except Exception as e:
            # Model ไม่รองรับ reasoning_effort → ลองใหม่โดยไม่ส่ง parameter นั้น
            if reasoning_effort is not None and "reasoning_effort" in str(e):
                print(f"  ⚠️  '{selected_model}' ไม่รองรับ reasoning_effort — รันโดยไม่ใช้ parameter นี้")
                try:
                    response = self._make_llm(model).invoke(messages)
                    return _parse_result(response)
                except Exception as e2:
                    e = e2
            return ChatResult(
                provider=self.provider_id,
                model=selected_model,
                content="",
                success=False,
                error=str(e),
            )

    # เป็นฟังก์ชันที่ใช้ส่งข้อความไปยังโมเดลของแต่ละ Provider ในรูปแบบของการสตรีม streaming) 
    # ซึ่งจะทำให้เราสามารถรับผลลัพธ์ได้เป็นส่วนๆ ขณะที่โมเดลกำลังประมวลผลอยู่ 
    # โดยจะมีการจัดการข้อผิดพลาดที่อาจเกิดขึ้นระหว่างการเรียกใช้งานโมเดลเช่นเดียวกับฟังก์ชัน chat
    def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = None,
    ) -> Iterator[str]:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        for chunk in self._make_llm(model).stream(messages):
            yield chunk.content or ""

    def __repr__(self) -> str:
        return f"LLMClient(provider='{self.provider_id}', model='{self.config.default_model}')"


# =============================================================================
# LLMManager
# =============================================================================

# เป็น Class ที่ทำหน้าที่จัดการและประสานงานกับ LLMClient หลายๆ ตัวที่เชื่อมต่อกับ Providers ต่างๆ
class LLMManager:
    def __init__(self, registry: dict[str, ProviderConfig] = None):
        self._registry = registry or PROVIDER_REGISTRY
        self._clients: dict[str, LLMClient] = {}
        self._key_status: dict[str, dict] = {}
        self._validate_and_init()

    def _validate_key(self, provider_id, config):
        if not config.requires_key:
            return True, "no-key-required", None
        api_key = os.getenv(config.env_var)
        if not api_key:
            return False, None, f"No API key ('{config.env_var}' not set)"
        if not api_key.startswith(config.prefix):
            return False, None, f"Invalid prefix (expected '{config.prefix}')"
        if api_key.strip() != api_key:
            return False, None, "Extra spaces/tabs in key"
        return True, api_key, None

    def _validate_and_init(self):
        print("=== Checking API Keys ===")
        for provider_id, config in self._registry.items():
            valid, api_key, reason = self._validate_key(provider_id, config)
            self._key_status[provider_id] = {"valid": valid, "reason": reason}
            if valid:
                self._clients[provider_id] = LLMClient(provider_id, config, api_key)
                print(f"  ✅ [{config.name}]")
            else:
                print(f"  ❌ [{config.name}] {reason}")
        print(f"\nAvailable providers: {self.available_providers}\n")

    @property
    def available_providers(self) -> list[str]:
        return list(self._clients.keys())

    def get_client(self, provider: str) -> LLMClient:
        if provider not in self._registry:
            raise ValueError(f"Unknown provider: '{provider}'")
        if provider not in self._clients:
            reason = self._key_status.get(provider, {}).get("reason", "Not available")
            raise ValueError(f"Provider '{provider}' unavailable: {reason}")
        return self._clients[provider]

    # เป็นฟังก์ชันที่ใช้ส่งข้อความไปยังโมเดลของแต่ละ Provider ผ่าน LLMClient 
    # ที่จัดการอยู่ใน LLMManager โดยจะรับพารามิเตอร์ต่างๆ เช่น system_prompt, user_prompt, provider, model, และ reasoning_effort 
    # และจะส่งต่อไปยัง LLMClient ที่เกี่ยวข้องเพื่อทำการเรียกใช้งานโมเดลและรับผลลัพธ์กลับมาในรูปแบบของ ChatResult
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        provider: str,
        model: str = None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        return self.get_client(provider).chat(system_prompt, user_prompt, model, reasoning_effort)

    def __repr__(self) -> str:
        return f"LLMManager(available={self.available_providers})"


# =============================================================================
# Utility
# =============================================================================

def print_separator(title: str = "", width: int = 60):
    if title:
        pad = max(0, width - len(title) - 2)
        print(f"\n{'─' * (pad // 2)} {title} {'─' * (pad - pad // 2)}")
    else:
        print("─" * width)


def _messages_to_system_user(messages: list[dict]) -> tuple[str, str]:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_parts   = [m["content"] for m in messages if m["role"] == "user"]
    system_prompt = "\n".join(system_parts) if system_parts else "You are a helpful assistant."
    user_prompt   = user_parts[-1] if user_parts else ""
    return system_prompt, user_prompt


# =============================================================================
# Prompt runner
# =============================================================================

# เป็นฟังก์ชันที่ใช้รันข้อความสนทนากับ Provider ที่ระบุ โดยรับพารามิเตอร์ต่างๆ
# เช่น messages, provider, manager, model, และ reasoning_effort
# และจะส่งข้อความไปยัง Provider ผ่าน LLMManager และรับผลลัพธ์กลับมาในรูปแบบของ
# ChatResult ซึ่งจะรวมข้อมูลต่างๆ เช่น ชื่อ Provider, โมเดลที่ใช้, เนื้อหาคำตอบ, สถานะความสำเร็จ,
# ข้อความแสดงข้อผิดพลาด (ถ้ามี), และจำนวนโทเค็นที่ใช้ใน prompt และ completion
def run_prompt(
    messages: list[dict],
    provider: str,
    manager: LLMManager | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> ChatResult:
    """Run conversation messages against a single provider.

    Args:
        messages:          OpenAI-style list of {"role": ..., "content": ...} dicts.
        provider:          Provider ID to query (e.g. "openai", "openrouter").
        manager:           Shared LLMManager instance.  Created automatically if omitted.
        model:             Optional model override, e.g. "gpt-4o".
        reasoning_effort:  "low" | "medium" | "high" — passed to providers that support it.

    Returns:
        ChatResult for the provider.
    """
    if manager is None:
        manager = LLMManager()

    system_prompt, user_prompt = _messages_to_system_user(messages)

    label = f"Provider: {provider}" + (f" | reasoning: {reasoning_effort}" if reasoning_effort else "")
    print_separator(label)
    try:
        result = manager.chat(system_prompt, user_prompt, provider, model, reasoning_effort)
    except ValueError as exc:
        result = ChatResult(provider=provider, model=model or "?", content="", success=False, error=str(exc))

    if result.success:
        print(f"Model  : {result.model}")
        print(f"Answer : {result.content}")
        if result.prompt_tokens is not None:
            print(f"Input  : {result.prompt_tokens} tokens")
        if result.completion_tokens is not None:
            print(f"Output : {result.completion_tokens} tokens")
        if result.cached_tokens is not None:
            print(f"Cached : {result.cached_tokens} tokens")
    else:
        error_str = result.error or ""
        if "429" in error_str:
            retry_match = re.search(r"retry[^0-9]*([0-9]+(?:\.[0-9]+)?)s", error_str, re.IGNORECASE)
            retry_hint = f" (retry in {float(retry_match.group(1)):.0f}s)" if retry_match else ""
            print(f"Error : 429 Rate limit / Quota exceeded{retry_hint}")
        else:
            print(f"Error : {error_str}")

    print_separator()
    return result


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    manager = LLMManager()

    hard_puzzle = [
        {"role": "system", "content": "คุณคือผู้ช่วยที่ตอบคำถามเป็นภาษาไทยเท่านั้น"},
        {"role": "user", "content":
            "คุณและคู่ของคุณเป็นผู้เข้าแข่งขันในรายการเกมโชว์ คุณทั้งสองถูกแยกไปอยู่คนละห้อง และได้รับตัวเลือกดังนี้: "
            'ร่วมมือ (Cooperate): เลือก "แบ่งปัน" — หากคุณทั้งคู่เลือกแบบนี้ คุณแต่ละคนจะได้รับเงิน $1,000 '
            "หักหลัง (Defect): เลือก 'ขโมย' — หากคนหนึ่งเลือกขโมยและอีกคนเลือกแบ่งปัน คนที่ขโมยจะได้เงิน $2,000 ส่วนอีกคนจะไม่ได้อะไรเลย"
            'หากทั้งคู่เลือกขโมย คุณทั้งสองจะไม่ได้อะไรเลย คุณจะเลือก "ขโมย" หรือ "แบ่งปัน"? เลือกมาอย่างใดอย่างหนึ่ง'},
    ]

    question = next(m["content"] for m in hard_puzzle if m["role"] == "user")
    print_separator("Hard Puzzle — คำถาม")
    print(question)

    run_prompt(hard_puzzle, provider="openai",
               manager=manager, model="gpt-5", reasoning_effort="low")

    run_prompt(hard_puzzle, provider="openrouter",
               manager=manager, model="qwen/qwen3.6-35b-a3b")

    run_prompt(hard_puzzle, provider="openrouter",
               manager=manager, model="google/gemma-4-31b-it")
