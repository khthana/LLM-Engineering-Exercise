import os
import re
import sys
import sqlite3
from dataclasses import dataclass
from typing import Optional, Iterator
from dotenv import load_dotenv

# Windows terminals may default to cp874 — force UTF-8 so emoji prints correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import gradio as gr

load_dotenv()


@dataclass
class ProviderConfig:
    name: str
    langchain_cls: str     # "openai" | "anthropic" | "gemini" | "ollama"
    default_model: str
    env_var: Optional[str] = None
    prefix: Optional[str] = None
    requires_key: bool = True
    base_url: Optional[str] = None


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
        langchain_cls="openai",
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

class LLMClient:
    def __init__(self, provider_id: str, config: ProviderConfig, api_key: str):
        self.provider_id = provider_id
        self.config = config
        self._api_key = api_key

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

    def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[str]:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        for chunk in self._make_llm(model, reasoning_effort).stream(messages):
            yield chunk.content or ""

    def stream_messages(
        self,
        messages: list,
        model: str = None,
        reasoning_effort: str | None = None,
    ) -> Iterator[str]:
        """Stream response given a pre-built list of LangChain messages (for multi-turn)."""
        for chunk in self._make_llm(model, reasoning_effort).stream(messages):
            yield chunk.content or ""

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
    """Run the same conversation messages against one or more providers."""
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
# KeyCraft Shop — SQLite Database
# =============================================================================

DB_PATH = "keycraft.db"

# (id, category, name, price, original_price, description)
SEED_DATA = [
    # Keyboards
    ("KC-K1", "keyboards", "KC-K1 Pro 75% Wireless",   3490, None, "Wireless 75% layout, hot-swap, RGB backlight"),
    ("KC-K2", "keyboards", "KC-K2 TKL Mechanical",     2990, None, "TKL layout, Gateron Red switch, wired"),
    ("KC-K3", "keyboards", "KC-K3 65% Compact",        2490, None, "65% compact, Bluetooth 5.0, multi-device"),
    ("KC-K4", "keyboards", "KC-K4 Mini 60%",           1990, None, "60% ultra-compact, programmable, wired"),
    ("KC-K5", "keyboards", "KC-K5 Full-size 100%",     3990, None, "Full-size layout, Cherry MX Brown, wired"),
    ("KC-K6", "keyboards", "KC-K6 Split Ergonomic",    5490, None, "Split ergonomic design, hot-swap, wireless"),
    # Mice (ลด 50%)
    ("KC-M1", "mice", "KC-M1 Wireless Ergonomic",  900,  1800, "Ergonomic shape, 2.4GHz wireless, 6 buttons"),
    ("KC-M2", "mice", "KC-M2 Gaming Precision",    700,  1400, "Gaming sensor 25600 DPI, RGB, wired"),
    ("KC-M3", "mice", "KC-M3 Silent Travel",       600,  1200, "Silent click, compact, USB-C charging"),
    ("KC-M4", "mice", "KC-M4 Vertical Ergonomic",  850,  1700, "Vertical grip, wireless, reduces wrist strain"),
    # Headsets (บางตัวลด 20%)
    ("KC-H1", "headsets", "KC-H1 Gaming 7.1 Surround", 2490, 3100, "7.1 virtual surround, RGB, noise-cancel mic"),
    ("KC-H2", "headsets", "KC-H2 Wireless Studio",     3990, None, "40hr battery, Hi-Res Audio, foldable"),
    ("KC-H3", "headsets", "KC-H3 Budget Clear Voice",   990, 1240, "Clear voice, stereo, USB + 3.5mm"),
    # Mousepads (บางตัวลด 30%)
    ("KC-P1", "mousepads", "KC-P1 XXL Speed Pad",     490,  700,  "900x400mm, smooth surface, anti-slip base"),
    ("KC-P2", "mousepads", "KC-P2 RGB Gaming Pad",    890,  None, "RGB edge lighting, medium surface, 400x350mm"),
    ("KC-P3", "mousepads", "KC-P3 Desk Mat Leather", 1290, 1850,  "PU leather, full-desk, waterproof"),
]


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id             TEXT PRIMARY KEY,
                category       TEXT NOT NULL,
                name           TEXT NOT NULL,
                price          INTEGER NOT NULL,
                original_price INTEGER,
                description    TEXT NOT NULL
            )
        """)
        if conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO products VALUES (?,?,?,?,?,?)",
                SEED_DATA,
            )
            print(f"  🗄️  keycraft.db: seeded {len(SEED_DATA)} products")
        else:
            print("  🗄️  keycraft.db: loaded (existing data)")


_init_db()


# =============================================================================
# KeyCraft Shop — Tool & System Prompt
# =============================================================================

@tool
def get_products(category: str) -> str:
    """ดึงรายการสินค้าจากฐานข้อมูล ระบุ category: keyboards, mice, headsets, mousepads"""
    rows = _get_conn().execute(
        "SELECT id, name, price, original_price, description FROM products WHERE category = ?",
        (category,),
    ).fetchall()
    if not rows:
        return f"ไม่พบสินค้าในหมวด '{category}'"
    lines = []
    for id_, name, price, orig, desc in rows:
        if orig:
            lines.append(f"{id_}: {name} — ~~฿{orig:,}~~ ฿{price:,} ({desc})")
        else:
            lines.append(f"{id_}: {name} — ฿{price:,} ({desc})")
    return "\n".join(lines)


SHOP_TOOLS = [get_products]

SYSTEM_PROMPT = """คุณคือ "KeyBot" พนักงานขายออนไลน์ของร้าน KeyCraft ร้านขายอุปกรณ์คอมพิวเตอร์คุณภาพสูง

**บทบาทของคุณ:**
- แนะนำสินค้าให้ตรงกับความต้องการของลูกค้า
- ใช้ tool `get_products(category)` เพื่อดึงข้อมูลสินค้าและราคาที่ถูกต้อง
- category ที่มี: keyboards, mice, headsets, mousepads
- ตอบคำถามด้วยภาษาไทยเป็นหลัก สุภาพและเป็นมิตร
- ยังไม่รับออเดอร์ ให้แนะนำสินค้าอย่างเดียว

**โปรโมชั่นพิเศษ:**
🎉 Mouse ทุกรุ่นลด **50%** | 🎧 Headset บางรุ่นลด **20%** | 🖱️ Mousepad บางรุ่นลด **30%**

เมื่อลูกค้าถามเกี่ยวกับสินค้า ให้เรียก tool ก่อนเสมอ แล้วค่อยตอบด้วยข้อมูลที่ได้"""


# =============================================================================
# Gradio UI — KeyCraft Chatbot
# =============================================================================

_manager = LLMManager()

OPENAI_MODELS = [
    ("GPT-5 Nano     $0.05/M",   "gpt-5-nano"),
    ("GPT-5.4 Nano   $0.20/M",   "gpt-5.4-nano"),
    ("GPT-5 Mini     $0.25/M",   "gpt-5-mini"),
    ("GPT-5.4 Mini   $0.75/M",   "gpt-5.4-mini"),
    ("GPT-5          $1.25/M",   "gpt-5"),
    ("GPT-5.1        $1.25/M",   "gpt-5.1"),
    ("GPT-5.2        $1.75/M",   "gpt-5.2"),
    ("GPT-5.4        $2.50/M",   "gpt-5.4"),
    ("GPT-5.5        $5.00/M",   "gpt-5.5"),
    ("GPT-5 Pro      $15.00/M",  "gpt-5-pro"),
    ("GPT-5.2 Pro    $21.00/M",  "gpt-5.2-pro"),
    ("GPT-5.4 Pro    $30.00/M",  "gpt-5.4-pro"),
    ("GPT-5.5 Pro    $30.00/M",  "gpt-5.5-pro"),
]

CLAUDE_MODELS = [
    ("Claude Haiku 4.5",  "claude-haiku-4-5-20251001"),
    ("Claude Sonnet 4.6", "claude-sonnet-4-6"),
]

GEMINI_MODELS = [
    ("Gemini 2.0 Flash", "gemini-2.0-flash"),
    ("Gemini 1.5 Pro",   "gemini-1.5-pro"),
]

OLLAMA_MODELS = [
    ("Gemma 4 E4B (Local)", "gemma4:e4b"),
]

PROVIDER_MODELS: dict[str, list] = {
    "ollama": OLLAMA_MODELS,
    "openai": OPENAI_MODELS,
    "claude": CLAUDE_MODELS,
    "gemini": GEMINI_MODELS,
}

REASONING_MODELS = {"gpt-5"}

_AVAILABLE_PROVIDERS = [p for p in ["ollama", "openai", "claude", "gemini"] if p in _manager.available_providers]
_DEFAULT_PROVIDER = _AVAILABLE_PROVIDERS[0] if _AVAILABLE_PROVIDERS else "openai"


def _initial_lc_history() -> list:
    return [SystemMessage(content=SYSTEM_PROMPT)]


def update_model_dropdown(provider: str):
    models = PROVIDER_MODELS.get(provider, OPENAI_MODELS)
    default = models[0][1] if models else None
    return gr.update(choices=models, value=default)


def update_reasoning_visibility(model: str):
    return gr.update(visible=model in REASONING_MODELS)


def chat_fn(user_message: str, chatbot_history: list, lc_history: list, provider: str, model: str, reasoning_effort: str):
    if not user_message.strip():
        yield chatbot_history, lc_history
        return

    if not lc_history:
        lc_history = _initial_lc_history()

    lc_history_updated = lc_history + [HumanMessage(content=user_message)]
    chatbot_history = chatbot_history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": ""},
    ]

    effort = reasoning_effort if model in REASONING_MODELS else None
    accumulated = ""

    try:
        client = _manager.get_client(provider)
        llm_with_tools = client._make_llm(model, effort).bind_tools(SHOP_TOOLS)

        current_messages = list(lc_history_updated)
        tool_map = {"get_products": get_products}

        while True:
            response = llm_with_tools.invoke(current_messages)

            if not response.tool_calls:
                accumulated = response.content
                chatbot_history[-1]["content"] = accumulated
                yield chatbot_history, lc_history
                break

            chatbot_history[-1]["content"] = "✨ กำลังค้นหาข้อมูล..."
            yield chatbot_history, lc_history

            current_messages.append(response)
            for tc in response.tool_calls:
                if tc["name"] in tool_map:
                    result = tool_map[tc["name"]].invoke(tc["args"])
                    current_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    except Exception as e:
        accumulated = f"**Error:** {e}"
        chatbot_history[-1]["content"] = accumulated
        yield chatbot_history, lc_history
        return

    new_lc_history = current_messages + [AIMessage(content=accumulated)]
    yield chatbot_history, new_lc_history


def clear_fn():
    return [], _initial_lc_history()


_dark_mode_js = """
if (document.querySelector('gradio-app')) {
    document.querySelector('gradio-app').classList.add('dark');
}
document.documentElement.classList.add('dark');
"""

with gr.Blocks(title="KeyCraft Chatbot") as demo:
    gr.Markdown(
        "## 🎹 KeyCraft — ที่ปรึกษาสินค้า Keyboard, Mouse, Headset & Mousepad\n"
        "**โปรโมชั่น:** Mouse ลด 50% | Headset บางรุ่นลด 20% | Mousepad บางรุ่นลด 30%"
    )

    lc_history_state = gr.State(_initial_lc_history())

    with gr.Row():
        provider_dropdown = gr.Dropdown(
            label="Provider",
            choices=[(p.upper(), p) for p in ["ollama", "openai", "claude", "gemini"]],
            value=_DEFAULT_PROVIDER,
            scale=1,
        )
        model_dropdown = gr.Dropdown(
            label="Model",
            choices=PROVIDER_MODELS.get(_DEFAULT_PROVIDER, OPENAI_MODELS),
            value=PROVIDER_MODELS.get(_DEFAULT_PROVIDER, OPENAI_MODELS)[0][1],
            scale=2,
        )
        reasoning_input = gr.Dropdown(
            label="Reasoning Effort",
            choices=["low", "medium", "high"],
            value="medium",
            visible=False,
            scale=1,
        )

    chatbot = gr.Chatbot(label="KeyBot", height=480)

    with gr.Row():
        user_input = gr.Textbox(
            label="",
            placeholder="ถามเกี่ยวกับสินค้า เช่น 'มี headset อะไรบ้าง?' หรือ 'แนะนำ mousepad ราคาถูก'",
            lines=2,
            scale=5,
        )
        with gr.Column(scale=1, min_width=120):
            send_btn = gr.Button("ส่ง", variant="primary")
            clear_btn = gr.Button("ล้างการสนทนา")

    provider_dropdown.change(
        fn=update_model_dropdown,
        inputs=[provider_dropdown],
        outputs=[model_dropdown],
    )
    model_dropdown.change(
        fn=update_reasoning_visibility,
        inputs=[model_dropdown],
        outputs=[reasoning_input],
    )

    send_inputs = [user_input, chatbot, lc_history_state, provider_dropdown, model_dropdown, reasoning_input]
    send_outputs = [chatbot, lc_history_state]

    send_btn.click(
        fn=chat_fn,
        inputs=send_inputs,
        outputs=send_outputs,
    ).then(fn=lambda: "", outputs=[user_input])

    user_input.submit(
        fn=chat_fn,
        inputs=send_inputs,
        outputs=send_outputs,
    ).then(fn=lambda: "", outputs=[user_input])

    clear_btn.click(
        fn=clear_fn,
        outputs=[chatbot, lc_history_state],
    )


if __name__ == "__main__":
    demo.launch(js=_dark_mode_js)
