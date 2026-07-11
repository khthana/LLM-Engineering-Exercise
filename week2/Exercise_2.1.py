import os
import re
from dataclasses import dataclass
from typing import Optional, Iterator
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

@dataclass
class ProviderConfig:
    name: str
    base_url: Optional[str]
    default_model: str
    env_var: Optional[str] = None
    prefix: Optional[str] = None
    requires_key: bool = True


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
    response_cost: Optional[float] = None

PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="OpenAI",
        env_var="OPENAI_API_KEY",
        prefix="sk-proj-",
        base_url=None,
        default_model="gpt-4o-mini",
    ),
    "claude": ProviderConfig(
        name="Anthropic Claude",
        env_var="ANTHROPIC_API_KEY",
        prefix="sk-ant-",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-sonnet-4-6",
    ),
    "gemini": ProviderConfig(
        name="Google Gemini",
        env_var="GOOGLE_API_KEY",
        prefix="AI",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-2.0-flash",
    ),
    "openrouter": ProviderConfig(
        name="OpenRouter",
        env_var="OPENROUTER_API_KEY",
        prefix="sk-or-",
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-oss-120b:free",
    ),
    "ollama": ProviderConfig(
        name="Ollama (Local)",
        env_var=None,
        prefix=None,
        base_url="http://localhost:11434/v1",
        default_model="gemma4:e4b",
        requires_key=False,
    ),
}



# =============================================================================
# LLMClient  (จาก Exercise_1.5v2 + stream_chat)
# =============================================================================

class LLMClient:
    def __init__(self, provider_id: str, config: ProviderConfig, api_key: str):
        self.provider_id = provider_id
        self.config = config
        self._client = OpenAI(api_key=api_key, base_url=config.base_url, timeout=120.0)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        selected_model = model or self.config.default_model
        extra = {}
        if reasoning_effort is not None:
            extra["reasoning_effort"] = reasoning_effort
        def _make_result(resp) -> ChatResult:
            usage = resp.usage
            details = getattr(usage, "prompt_tokens_details", None)
            cost = getattr(resp, "_hidden_params", {}).get("response_cost")
            return ChatResult(
                provider=self.provider_id,
                model=selected_model,
                content=resp.choices[0].message.content,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                cached_tokens=getattr(details, "cached_tokens", None),
                response_cost=cost,
            )

        try:
            response = self._client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                **extra,
            )
            return _make_result(response)
        except Exception as e:
            # Model ไม่รองรับ reasoning_effort → ลองใหม่โดยไม่ส่ง parameter นั้น
            if reasoning_effort is not None and "reasoning_effort" in str(e):
                print(f"  ⚠️  '{selected_model}' ไม่รองรับ reasoning_effort — รันโดยไม่ใช้ parameter นี้")
                try:
                    response = self._client.chat.completions.create(
                        model=selected_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_prompt},
                        ],
                    )
                    return _make_result(response)
                except Exception as e2:
                    e = e2
            return ChatResult(
                provider=self.provider_id,
                model=selected_model,
                content="",
                success=False,
                error=str(e),
            )

    def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = None,
    ) -> Iterator[str]:
        selected_model = model or self.config.default_model
        stream = self._client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""

    def __repr__(self) -> str:
        return f"LLMClient(provider='{self.provider_id}', model='{self.config.default_model}')"


# =============================================================================
# LLMManager  (จาก Exercise_1.5v2)
# =============================================================================

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


# =============================================================================
# Multi-provider prompt runner
# =============================================================================

def _messages_to_system_user(messages: list[dict]) -> tuple[str, str]:
    """Split a conversation list into (system_prompt, user_prompt).

    Supports a leading system message or user-only conversations.
    Only the last user message is used as the user prompt.
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_parts   = [m["content"] for m in messages if m["role"] == "user"]
    system_prompt = "\n".join(system_parts) if system_parts else "You are a helpful assistant."
    user_prompt   = user_parts[-1] if user_parts else ""
    return system_prompt, user_prompt


def run_prompt(
    messages: list[dict],
    providers: list[str] | None = None,
    manager: LLMManager | None = None,
    model_overrides: dict[str, str] | None = None,
    reasoning_effort: str | None = None,
) -> list[ChatResult]:
    """Run the same conversation messages against one or more providers.

    Args:
        messages:          OpenAI-style list of {"role": ..., "content": ...} dicts.
        providers:         Provider IDs to query.  Defaults to all available providers.
        manager:           Shared LLMManager instance.  Created automatically if omitted.
        model_overrides:   Optional per-provider model override, e.g. {"openai": "gpt-4o"}.
        reasoning_effort:  "low" | "medium" | "high" — passed to providers that support it
                           (e.g. OpenAI o-series).  Ignored silently by providers that don't.

    Returns:
        List of ChatResult, one per provider.
    """
    if manager is None:
        manager = LLMManager()

    targets = providers if providers is not None else manager.available_providers
    system_prompt, user_prompt = _messages_to_system_user(messages)
    model_overrides = model_overrides or {}

    results: list[ChatResult] = []
    for provider_id in targets:
        model = model_overrides.get(provider_id)
        label = f"Provider: {provider_id}" + (f" | reasoning: {reasoning_effort}" if reasoning_effort else "")
        print_separator(label)
        try:
            result = manager.chat(system_prompt, user_prompt, provider_id, model, reasoning_effort)
        except ValueError as exc:
            result = ChatResult(provider=provider_id, model=model or "?", content="", success=False, error=str(exc))

        if result.success:
            print(f"Model  : {result.model}")
            print(f"Answer : {result.content}")
            if result.prompt_tokens is not None:
                print(f"Input  : {result.prompt_tokens} tokens")
            if result.completion_tokens is not None:
                print(f"Output : {result.completion_tokens} tokens")
            if result.cached_tokens is not None:
                print(f"Cached : {result.cached_tokens} tokens")
            if result.response_cost is not None:
                print(f"Cost   : {result.response_cost * 100:.4f} cents")
        else:
            error_str = result.error or ""
            # แสดง 429 แบบกระชับพร้อม retry delay ถ้ามี
            if "429" in error_str:
                retry_match = re.search(r"retry[^0-9]*([0-9]+(?:\.[0-9]+)?)s", error_str, re.IGNORECASE)
                retry_hint = f" (retry in {float(retry_match.group(1)):.0f}s)" if retry_match else ""
                print(f"Error : 429 Rate limit / Quota exceeded{retry_hint}")
            else:
                print(f"Error : {error_str}")
        results.append(result)

    print_separator()
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    manager = LLMManager()

    easy_puzzle = [
        {"role": "system", "content": "คุณคือผู้ช่วยที่ตอบคำถามเป็นภาษาไทยเท่านั้น"},
        {"role": "user", "content":
            "โยนเหรียญ 2 เหรียญ เหรียญหนึ่งออกหัว ความน่าจะเป็นที่อีกเหรียญจะออกก้อยคือเท่าไร? ตอบเฉพาะค่าความน่าจะเป็นเท่านั้น"},
    ]

    # hard_puzzle = [
    #     {"role": "system", "content": "คุณคือผู้ช่วยที่ตอบคำถามเป็นภาษาไทยเท่านั้น"},
    #     {"role": "user", "content":
    #         "บนชั้นหนังสือมีหนังสือพุชกิน 2 เล่มวางเรียงกัน คือเล่มที่หนึ่งและเล่มที่สอง "
    #         "หน้าหนังสือทั้งหมดของแต่ละเล่มรวมกันมีความหนา 2 เซนติเมตร และปกแต่ละด้านหนา 2 มิลลิเมตร "
    #         "หนอนตัวหนึ่งแทะ (ในแนวตั้งฉากกับหน้ากระดาษ) จากหน้าแรกของเล่มที่หนึ่งไปยังหน้าสุดท้ายของเล่มที่สอง "
    #         "หนอนแทะผ่านระยะทางรวมเท่าใด? ตอบเฉพาะระยะทางเท่านั้น"},
    # ]

    hard_puzzle = [
        {"role": "system", "content": "คุณคือผู้ช่วยที่ตอบคำถามเป็นภาษาไทยเท่านั้น"},
        {"role": "user", "content":
            "คุณและคู่ของคุณเป็นผู้เข้าแข่งขันในรายการเกมโชว์ คุณทั้งสองถูกแยกไปอยู่คนละห้อง และได้รับตัวเลือกดังนี้: "
            'ร่วมมือ (Cooperate): เลือก “แบ่งปัน” — หากคุณทั้งคู่เลือกแบบนี้ คุณแต่ละคนจะได้รับเงิน $1,000 '
            “หักหลัง (Defect): เลือก 'ขโมย' — หากคนหนึ่งเลือกขโมยและอีกคนเลือกแบ่งปัน คนที่ขโมยจะได้เงิน $2,000 ส่วนอีกคนจะไม่ได้อะไรเลย”
            'หากทั้งคู่เลือกขโมย คุณทั้งสองจะไม่ได้อะไรเลย คุณจะเลือก “ขโมย” หรือ “แบ่งปัน”? เลือกมาอย่างใดอย่างหนึ่ง'},
    ]


    # print_separator("Easy Puzzle")
    # run_prompt(
    #     easy_puzzle,
    #     providers=["openai"],
    #     manager=manager,
    #     model_overrides={"openai": "o4-mini"},
    #     reasoning_effort="low",
    # )

    question = next(m["content"] for m in hard_puzzle if m["role"] == "user")
    print_separator("Hard Puzzle — คำถาม")
    print(question)

    run_prompt(hard_puzzle, providers=["openai"],
               manager=manager, model_overrides={"openai": "gpt-5"}, reasoning_effort="low")

    # run_prompt(hard_puzzle, providers=["gemini"],
    #            manager=manager)

    run_prompt(hard_puzzle, providers=["openrouter"],
               manager=manager, model_overrides={"openrouter": "qwen/qwen3.6-35b-a3b"})

    run_prompt(hard_puzzle, providers=["openrouter"],
               manager=manager, model_overrides={"openrouter": "google/gemma-4-31b-it"})