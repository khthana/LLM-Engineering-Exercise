# Lite LLM 

import os
import re
from dataclasses import dataclass
from typing import Optional, Iterator
from dotenv import load_dotenv
import litellm

litellm.set_verbose = False
litellm.suppress_debug_info = True

load_dotenv()


@dataclass
class ProviderConfig:
    name: str
    litellm_prefix: str
    default_model: str
    env_var: Optional[str] = None
    prefix: Optional[str] = None
    requires_key: bool = True
    api_base: Optional[str] = None


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
        litellm_prefix="openai",
        env_var="OPENAI_API_KEY",
        prefix="sk-proj-",
        default_model="gpt-4o-mini",
    ),
    "claude": ProviderConfig(
        name="Anthropic Claude",
        litellm_prefix="anthropic",
        env_var="ANTHROPIC_API_KEY",
        prefix="sk-ant-",
        default_model="claude-sonnet-4-6",
    ),
    "gemini": ProviderConfig(
        name="Google Gemini",
        litellm_prefix="gemini",
        env_var="GOOGLE_API_KEY",
        prefix="AI",
        default_model="gemini-2.0-flash",
    ),
    "openrouter": ProviderConfig(
        name="OpenRouter",
        litellm_prefix="openrouter",
        env_var="OPENROUTER_API_KEY",
        prefix="sk-or-",
        default_model="openai/gpt-oss-120b:free",
    ),
    "ollama": ProviderConfig(
        name="Ollama (Local)",
        litellm_prefix="ollama",
        env_var=None,
        prefix=None,
        default_model="gemma4:e4b",
        requires_key=False,
        api_base="http://localhost:11434",
    ),
}


# =============================================================================
# LLMClient  (ใช้ litellm แทน OpenAI SDK)
# =============================================================================

class LLMClient:
    def __init__(self, provider_id: str, config: ProviderConfig, api_key: str):
        self.provider_id = provider_id
        self.config = config
        self._api_key = api_key

    def _litellm_model(self, model: str = None) -> str:
        base = model or self.config.default_model
        return f"{self.config.litellm_prefix}/{base}"

    def _call_kwargs(self) -> dict:
        kwargs = {}
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self._api_key and self._api_key != "no-key-required":
            kwargs["api_key"] = self._api_key
        return kwargs

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        litellm_model = self._litellm_model(model)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        kwargs = self._call_kwargs()
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        def _parse_result(resp) -> ChatResult:
            usage = resp.usage
            cached = None
            details = getattr(usage, "prompt_tokens_details", None)
            if details:
                cached = getattr(details, "cached_tokens", None)
            try:
                cost = litellm.completion_cost(resp)
            except Exception:
                cost = None
            return ChatResult(
                provider=self.provider_id,
                model=litellm_model,
                content=resp.choices[0].message.content,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                cached_tokens=cached,
                response_cost=cost,
            )

        try:
            response = litellm.completion(model=litellm_model, messages=messages, **kwargs)
            return _parse_result(response)
        except Exception as e:
            # Model ไม่รองรับ reasoning_effort → ลองใหม่โดยไม่ส่ง parameter นั้น
            if reasoning_effort is not None and "reasoning_effort" in str(e):
                print(f"  ⚠️  '{litellm_model}' ไม่รองรับ reasoning_effort — รันโดยไม่ใช้ parameter นี้")
                try:
                    kwargs_no_re = {k: v for k, v in kwargs.items() if k != "reasoning_effort"}
                    response = litellm.completion(model=litellm_model, messages=messages, **kwargs_no_re)
                    return _parse_result(response)
                except Exception as e2:
                    e = e2
            return ChatResult(
                provider=self.provider_id,
                model=litellm_model,
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
        litellm_model = self._litellm_model(model)
        kwargs = self._call_kwargs()
        stream = litellm.completion(
            model=litellm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""

    def __repr__(self) -> str:
        return f"LLMClient(provider='{self.provider_id}', model='{self.config.default_model}')"


# =============================================================================
# LLMManager
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


def _messages_to_system_user(messages: list[dict]) -> tuple[str, str]:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_parts   = [m["content"] for m in messages if m["role"] == "user"]
    system_prompt = "\n".join(system_parts) if system_parts else "You are a helpful assistant."
    user_prompt   = user_parts[-1] if user_parts else ""
    return system_prompt, user_prompt


# =============================================================================
# Multi-provider prompt runner
# =============================================================================

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
        reasoning_effort:  "low" | "medium" | "high" — passed to providers that support it.

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

    run_prompt(hard_puzzle, providers=["openai"],
               manager=manager, model_overrides={"openai": "gpt-5"}, reasoning_effort="low")

    run_prompt(hard_puzzle, providers=["openrouter"],
               manager=manager, model_overrides={"openrouter": "qwen/qwen3.6-35b-a3b"})

    run_prompt(hard_puzzle, providers=["openrouter"],
               manager=manager, model_overrides={"openrouter": "google/gemma-4-31b-it"})
