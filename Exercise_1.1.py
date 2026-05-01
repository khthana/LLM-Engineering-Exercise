import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

load_dotenv()  # โหลด .env

# เลือก provider ที่ต้องการใช้: "openai", "claude", "gemini", "openrouter", "ollama"
AI_PROVIDER = "openrouter"

# Config ของแต่ละ provider
# requires_key=False หมายถึงไม่ต้องการ API key (เช่น Ollama ที่รันในเครื่อง)
PROVIDER_CONFIG = {
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "prefix": "sk-proj-",
        "name": "OpenAI",
        "base_url": None,
        "model": "gpt-4o-mini",
        "requires_key": True,
    },
    "claude": {
        "env_var": "ANTHROPIC_API_KEY",
        "prefix": "sk-ant-",
        "name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-6",
        "requires_key": True,
    },
    "gemini": {
        "env_var": "GOOGLE_API_KEY",
        "prefix": "AI",
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.0-flash",
        "requires_key": True,
    },
    "openrouter": {
        "env_var": "OPENROUTER_API_KEY",
        "prefix": "sk-or-",
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-oss-120b:free",
        "requires_key": True,
    },
    "ollama": {
        "env_var": None,
        "prefix": None,
        "name": "Ollama (Local)",
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2",
        "requires_key": False,
    },
}

# ---- Prompt Library ----
# เก็บ system_prompt และ user_prompt สำหรับแต่ละวัตถุประสงค์
# user_prefix คือส่วนที่จะต่อหน้าเนื้อหาที่ส่งไป (เช่น เนื้อหาเว็บ, โค้ด)
PROMPTS = {
    "website_summary_snarky": {
        "system": (
            "You are a snarky assistant that analyzes the contents of a website, "
            "and provides a short, snarky, humorous summary, ignoring text that might be navigation related. "
            "Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown."
        ),
        "user_prefix": (
            "Here are the contents of a website. "
            "Provide a short summary of this website in thai language. "
            "If it includes news or announcements, then summarize these too.\n\n"
        ),
    },
    "website_summary_formal": {
        "system": (
            "You are a professional assistant that analyzes website content and provides a concise, "
            "neutral summary. Focus on the main purpose, key offerings, and any notable announcements. "
            "Respond in markdown."
        ),
        "user_prefix": "Please summarize the following website content in Thai language:\n\n",
    },
    "translate_thai": {
        "system": (
            "You are a professional translator. Translate the given content into Thai. "
            "Keep technical terms in English when appropriate. Respond only with the translation."
        ),
        "user_prefix": "Translate the following text to Thai:\n\n",
    },
    "code_review": {
        "system": (
            "You are an expert software engineer conducting a code review. "
            "Identify bugs, security issues, performance problems, and style issues. "
            "Be constructive and explain why each issue matters. Respond in markdown."
        ),
        "user_prefix": "Please review the following code:\n\n",
    },
    "explain_simple": {
        "system": (
            "You are a patient teacher who explains complex topics simply. "
            "Use analogies and plain language. Avoid jargon unless you define it first."
        ),
        "user_prefix": "Please explain the following in simple terms:\n\n",
    },
}

print("=== Checking API Keys ===")

API_KEY_STATUS = {}
for _provider, _config in PROVIDER_CONFIG.items():
    _name = _config["name"]

    # Ollama และ provider ที่ไม่ต้องการ key ให้ผ่านได้เลย
    if not _config.get("requires_key", True):
        API_KEY_STATUS[_provider] = {"valid": True, "api_key": "ollama", "reason": None}
        print(f"  ✅ [{_name}] No key required (local)")
        continue

    _api_key = os.getenv(_config["env_var"])

    if not _api_key:
        API_KEY_STATUS[_provider] = {"valid": False, "api_key": None, "reason": "No API key found"}
        print(f"  ❌ [{_name}] No API key found ('{_config['env_var']}')")
    elif not _api_key.startswith(_config["prefix"]):
        API_KEY_STATUS[_provider] = {"valid": False, "api_key": None, "reason": f"Key doesn't start with '{_config['prefix']}'"}
        print(f"  ❌ [{_name}] Invalid prefix (expected '{_config['prefix']}')")
    elif _api_key.strip() != _api_key:
        API_KEY_STATUS[_provider] = {"valid": False, "api_key": None, "reason": "Extra spaces/tabs found"}
        print(f"  ❌ [{_name}] Extra spaces/tabs in key")
    else:
        API_KEY_STATUS[_provider] = {"valid": True, "api_key": _api_key, "reason": None}
        print(f"  ✅ [{_name}] OK")

_available = [p for p, r in API_KEY_STATUS.items() if r["valid"]]
print(f"\nAvailable providers: {_available}\n")

# ---- Client Cache ----

_client_cache = {}  # เก็บ object client ที่เคยสร้างเอาไว้

def get_client(provider: str) -> OpenAI:
    if provider not in PROVIDER_CONFIG:
        raise ValueError(f"Unknown provider: '{provider}'. Choose from: {list(PROVIDER_CONFIG.keys())}")

    if not API_KEY_STATUS[provider]["valid"]:
        reason = API_KEY_STATUS[provider]["reason"]
        raise ValueError(f"Provider '{provider}' is not available: {reason}")

    if provider not in _client_cache:
        config = PROVIDER_CONFIG[provider]
        api_key = API_KEY_STATUS[provider]["api_key"]
        _client_cache[provider] = OpenAI(
            api_key=api_key,
            base_url=config["base_url"]
        )
    return _client_cache[provider]

# ---- Reusable Chat Function ----

def chat(messages: list, provider: str = "openrouter", model: str = None) -> str:
    config = PROVIDER_CONFIG.get(provider)
    if not config:
        raise ValueError(f"Unknown provider: '{provider}'")

    selected_model = model or config["model"]
    client = get_client(provider)

    response = client.chat.completions.create(
        model=selected_model,
        messages=messages
    )
    return response.choices[0].message.content

def fetch_website_contents(url: str, use_playwright: bool = False) -> str:
    if use_playwright:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded")
            text = page.inner_text("body")
            browser.close()
            return text
    else:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()


def basic_prompt(provider: str = "openrouter", model: str = None):
    message = "Hello! This is my first ever message to you! Hi!"
    messages = [{"role": "user", "content": message}]
    response = chat(messages, provider=provider, model=model)
    print("prompt : ", message)
    print("answer : ", response)

basic_prompt()


def analyze_website(
    url: str,
    system_prompt: str,
    user_prompt: str,
    provider: str = "openrouter",
    model: str = None,
) -> str:
    """
    ดึงเนื้อหาเว็บจาก url แล้ววิเคราะห์ด้วย system_prompt และ user_prompt ที่กำหนด
    คืนค่า response เป็น string เพื่อให้ผู้เรียกนำไปใช้ต่อได้
    """
    content = fetch_website_contents(url)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt + content},
    ]
    return chat(messages, provider=provider, model=model)


# ---- ตัวอย่างการเรียกใช้ด้วย prompt แต่ละแบบ ----

TARGET_URL = "https://www.cnn.com"

# # แบบ snarky (ค่าเริ่มต้นเดิม)
# snarky = PROMPTS["website_summary_snarky"]
# response = analyze_website(
#     url=TARGET_URL,
#     system_prompt=snarky["system"],
#     user_prompt=snarky["user_prefix"],
# )
# print("=== Snarky Summary ===")
# print(response)

# แบบ formal
formal = PROMPTS["website_summary_formal"]
response = analyze_website(
    url=TARGET_URL,
    system_prompt=formal["system"],
    user_prompt=formal["user_prefix"],
)
print("\n=== Formal Summary ===")
print(response)

# # แบบแปลเป็นภาษาไทย
# thai = PROMPTS["translate_thai"]
# response = analyze_website(
#     url=TARGET_URL,
#     system_prompt=thai["system"],
#     user_prompt=thai["user_prefix"],
# )
# print("\n=== Thai Translation ===")
# print(response)
