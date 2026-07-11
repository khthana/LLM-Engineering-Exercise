"""
Exercise_1.5v2.py
=================
Company Brochure Generator — ใช้ LLMManager จาก Exercise_1.2

เหมือน Exercise_1.5 แต่แทนที่ openai client ตรงๆ ด้วย LLMManager ที่ยืดหยุ่นกว่า
รองรับหลาย provider (Ollama, OpenAI, Claude, Gemini, OpenRouter)
ค่าเริ่มต้น: Ollama (Local)

ขั้นตอนการทำงาน:
  1. ดึงลิงก์ทั้งหมดจากหน้าหลักของบริษัท
  2. ให้ LLM คัดเลือกลิงก์ที่เกี่ยวข้อง (About, Careers ฯลฯ) → JSON
  3. ดึงเนื้อหาจากลิงก์ที่เลือก
  4. สร้าง Brochure แบบ Markdown ด้วย LLM
  5. แสดงผลแบบ streaming (typewriter effect) ในเทอร์มินัล
"""

import os
import json
import concurrent.futures
import requests as http_requests
from dataclasses import dataclass
from typing import Optional, Iterator
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI

load_dotenv()


# =============================================================================
# Data Classes  (จาก Exercise_1.2)
# =============================================================================

@dataclass
class ProviderConfig:
    """Config ของแต่ละ LLM provider"""
    name: str
    base_url: Optional[str]
    default_model: str
    env_var: Optional[str] = None
    prefix: Optional[str] = None
    requires_key: bool = True


@dataclass
class PromptTemplate:
    """คู่ system/user prompt สำหรับแต่ละวัตถุประสงค์"""
    system: str
    user_prefix: str = ""
    description: str = ""


@dataclass
class ChatResult:
    """ผลลัพธ์จากการเรียก LLM หนึ่งครั้ง"""
    provider: str
    model: str
    content: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class ModelTarget:
    """ระบุ provider + model สำหรับ chat_parallel"""
    provider: str
    model: str = None
    label: str = None

    def __post_init__(self):
        if self.label is None:
            model_part = self.model or "default"
            self.label = f"{self.provider}:{model_part}"


# =============================================================================
# Provider Registry  (จาก Exercise_1.2)
# =============================================================================

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
# Prompt Library  (จาก Exercise_1.2 + เพิ่ม brochure prompts)
# =============================================================================

PROMPTS: dict[str, PromptTemplate] = {
    "news_summary_thai": PromptTemplate(
        description="สรุปข่าวเป็นภาษาไทย",
        system=(
            "คุณเป็นนักข่าวมืออาชีพที่สรุปข่าวได้กระชับและตรงประเด็น "
            "สรุปเนื้อหาข่าวเป็นภาษาไทย โดยแบ่งเป็นหัวข้อย่อยตามหมวดหมู่ข่าว "
            "เช่น การเมือง เศรษฐกิจ เทคโนโลยี กีฬา บันเทิง ฯลฯ "
            "แต่ละหัวข้อให้สรุป 2-3 ประโยค ตอบในรูปแบบ markdown "
            "ห้ามใส่ markdown ใน code block — ตอบ markdown ตรงๆ"
        ),
        user_prefix=(
            "ต่อไปนี้คือเนื้อหาจากเว็บไซต์ข่าว "
            "กรุณาสรุปข่าวสำคัญที่พบเป็นภาษาไทย:\n\n"
        ),
    ),
    "brochure_link_selector": PromptTemplate(
        description="คัดเลือกลิงก์ที่เหมาะสำหรับ Brochure (JSON)",
        system=(
            "You are provided with a list of links found on a webpage. "
            "Decide which links are most relevant to include in a company brochure, "
            "such as About, Company, Careers/Jobs pages. "
            "Respond ONLY in JSON with this exact format:\n"
            '{"links": [{"type": "about page", "url": "https://full.url/about"}, ...]}'
        ),
        user_prefix=(
            "Here is the list of links on the website. "
            "Select relevant links for a company brochure. "
            "Respond with full https URLs in JSON. "
            "Do not include Terms of Service, Privacy, or email links.\n\n"
            "Links:\n"
        ),
    ),
    "brochure_generator": PromptTemplate(
        description="สร้าง Company Brochure จากเนื้อหาเว็บ",
        system=(
            "You are an assistant that analyzes the contents of several relevant pages "
            "from a company website and creates a short brochure about the company "
            "for prospective customers, investors and recruits. "
            "Respond in markdown without code blocks. "
            "Include details of company culture, customers and careers/jobs if available."
        ),
        user_prefix="",  # user prompt สร้างใน get_brochure_user_prompt()
    ),
}


# =============================================================================
# LLMClient  (จาก Exercise_1.2 + เพิ่ม json_mode และ stream_chat)
# =============================================================================

class LLMClient:
    """
    Wrapper สำหรับ OpenAI-compatible client ของ provider หนึ่งตัว
    เพิ่ม: json_mode และ stream_chat() จากของเดิม
    """

    def __init__(self, provider_id: str, config: ProviderConfig, api_key: str):
        self.provider_id = provider_id
        self.config = config
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = None,
        json_mode: bool = False,
    ) -> ChatResult:
        """ส่ง message ไปหา LLM แล้วคืน ChatResult"""
        selected_model = model or self.config.default_model
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                **kwargs,
            )
            return ChatResult(
                provider=self.provider_id,
                model=selected_model,
                content=response.choices[0].message.content,
            )
        except Exception as e:
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
        """ส่ง message แบบ streaming — yield ทีละ chunk"""
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
# LLMManager  (จาก Exercise_1.2 + เพิ่ม stream และ json_mode)
# =============================================================================

class LLMManager:
    """
    ตัวจัดการ LLM หลาย provider
    - ตรวจสอบ key อัตโนมัติตอน init
    - รองรับ json_mode และ streaming
    """

    def __init__(self, registry: dict[str, ProviderConfig] = None):
        self._registry = registry or PROVIDER_REGISTRY
        self._clients: dict[str, LLMClient] = {}
        self._key_status: dict[str, dict] = {}
        self._validate_and_init()

    def _validate_key(
        self, provider_id: str, config: ProviderConfig
    ) -> tuple[bool, Optional[str], Optional[str]]:
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
            raise ValueError(
                f"Unknown provider: '{provider}'. "
                f"Choose from: {list(self._registry.keys())}"
            )
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
        json_mode: bool = False,
    ) -> ChatResult:
        """ส่ง prompt ไปหา provider เดียว"""
        return self.get_client(provider).chat(
            system_prompt, user_prompt, model, json_mode=json_mode
        )

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        provider: str,
        model: str = None,
    ) -> Iterator[str]:
        """ส่ง prompt แบบ streaming — yield ทีละ chunk"""
        return self.get_client(provider).stream_chat(system_prompt, user_prompt, model)

    def chat_parallel(
        self,
        system_prompt: str,
        user_prompt: str,
        targets: list[ModelTarget] = None,
    ) -> dict[str, ChatResult]:
        """ส่ง prompt เดียวกันไปหลาย model พร้อมกัน"""
        if targets is None:
            targets = [ModelTarget(p) for p in self.available_providers]

        def _call(target: ModelTarget):
            client = self.get_client(target.provider)
            return target.label, client.chat(system_prompt, user_prompt, target.model)

        results: dict[str, ChatResult] = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(_call, t): t for t in targets}
            for future in concurrent.futures.as_completed(futures):
                label, result = future.result()
                results[label] = result
        return results

    def chat_with_template(
        self,
        template_key: str,
        content: str,
        provider: str,
        model: str = None,
        json_mode: bool = False,
    ) -> ChatResult:
        """ส่ง content ไปพร้อม prompt template"""
        template = _get_template(template_key)
        return self.chat(
            system_prompt=template.system,
            user_prompt=template.user_prefix + content,
            provider=provider,
            model=model,
            json_mode=json_mode,
        )

    def stream_with_template(
        self,
        template_key: str,
        content: str,
        provider: str,
        model: str = None,
    ) -> Iterator[str]:
        """ส่ง content แบบ streaming พร้อม prompt template"""
        template = _get_template(template_key)
        return self.stream(
            system_prompt=template.system,
            user_prompt=template.user_prefix + content,
            provider=provider,
            model=model,
        )

    def __repr__(self) -> str:
        return f"LLMManager(available={self.available_providers})"


# =============================================================================
# Utility Functions  (จาก Exercise_1.2)
# =============================================================================

def _get_template(template_key: str) -> PromptTemplate:
    template = PROMPTS.get(template_key)
    if not template:
        raise ValueError(
            f"Unknown template: '{template_key}'. "
            f"Available: {list(PROMPTS.keys())}"
        )
    return template


def print_result(result: ChatResult):
    config = PROVIDER_REGISTRY.get(result.provider)
    name = config.name if config else result.provider
    print(f"\n{'='*55}")
    print(f"  {name}  |  model: {result.model}")
    print(f"{'='*55}")
    if result.success:
        print(result.content)
    else:
        print(f"❌ Error: {result.error}")


# =============================================================================
# Web Scraper  (จาก scraper.py ใน day5)
# =============================================================================

SCRAPER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}

MAX_CONTENT_CHARS = 2_000
MAX_PROMPT_CHARS  = 5_000


def fetch_website_contents(url: str) -> str:
    """ดึง title + เนื้อหาจากหน้าเว็บ ตัดที่ MAX_CONTENT_CHARS"""
    response = http_requests.get(url, headers=SCRAPER_HEADERS, timeout=15)
    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.string if soup.title else "No title found"
    if soup.body:
        for tag in soup.body(["script", "style", "img", "input"]):
            tag.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""
    return (title + "\n\n" + text)[:MAX_CONTENT_CHARS]


def fetch_website_links(url: str) -> list[str]:
    """ดึง href ทั้งหมดจากหน้าเว็บ"""
    response = http_requests.get(url, headers=SCRAPER_HEADERS, timeout=15)
    soup = BeautifulSoup(response.content, "html.parser")
    links = [a.get("href") for a in soup.find_all("a")]
    return [link for link in links if link]


# =============================================================================
# Brochure Generator  (จาก Exercise_1.5 ปรับให้ใช้ LLMManager)
# =============================================================================

DEFAULT_PROVIDER = "ollama"


def select_relevant_links(url: str, manager: LLMManager, provider: str = DEFAULT_PROVIDER) -> dict:
    """
    ให้ LLM คัดเลือกลิงก์ที่เกี่ยวข้องกับบริษัท
    คืน dict รูปแบบ {"links": [{"type": ..., "url": ...}, ...]}
    """
    print(f"  🔍 กำลังคัดเลือกลิงก์จาก {url} ...")

    links_text = "\n".join(fetch_website_links(url))
    user_prompt = (
        f"Website URL: {url}\n\n"
        "Please select relevant links for a company brochure. "
        "Respond with full https URLs in JSON only. "
        "Do not include Terms of Service, Privacy, or email links.\n\n"
        f"Links:\n{links_text}"
    )

    template = _get_template("brochure_link_selector")
    result = manager.chat(
        system_prompt=template.system,
        user_prompt=user_prompt,
        provider=provider,
        json_mode=True,
    )

    if not result.success:
        print(f"  ⚠️  LLM error: {result.error} — ใช้ลิงก์ว่างแทน")
        return {"links": []}

    try:
        parsed = json.loads(result.content)
        print(f"  ✅ พบ {len(parsed.get('links', []))} ลิงก์ที่เกี่ยวข้อง")
        return parsed
    except json.JSONDecodeError:
        print("  ⚠️  parse JSON ไม่ได้ — ใช้ลิงก์ว่างแทน")
        return {"links": []}


def fetch_page_and_all_relevant_links(
    url: str,
    manager: LLMManager,
    provider: str = DEFAULT_PROVIDER,
) -> str:
    """รวมเนื้อหาจากหน้าหลัก + หน้าที่ LLM เลือกว่าเกี่ยวข้อง"""
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url, manager, provider)

    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links.get("links", []):
        result += f"\n\n### Link: {link['type']}\n"
        try:
            result += fetch_website_contents(link["url"])
        except Exception as e:
            result += f"(ดึงไม่ได้: {e})"
    return result


def get_brochure_user_prompt(
    company_name: str,
    url: str,
    manager: LLMManager,
    provider: str = DEFAULT_PROVIDER,
) -> str:
    """สร้าง user prompt สำหรับ brochure generation"""
    prompt = (
        f"You are looking at a company called: {company_name}\n"
        "Here are the contents of its landing page and other relevant pages; "
        "use this information to build a short brochure in markdown without code blocks.\n\n"
    )
    prompt += fetch_page_and_all_relevant_links(url, manager, provider)
    return prompt[:MAX_PROMPT_CHARS]


def create_brochure(
    company_name: str,
    url: str,
    manager: LLMManager,
    provider: str = DEFAULT_PROVIDER,
    model: str = None,
) -> ChatResult:
    """สร้าง Brochure แบบปกติ (รอครบแล้ว print)"""
    print(f"\n⏳ กำลังสร้าง Brochure สำหรับ {company_name} ...")
    template = _get_template("brochure_generator")
    result = manager.chat(
        system_prompt=template.system,
        user_prompt=get_brochure_user_prompt(company_name, url, manager, provider),
        provider=provider,
        model=model,
    )
    print_result(result)
    return result


def stream_brochure(
    company_name: str,
    url: str,
    manager: LLMManager,
    provider: str = DEFAULT_PROVIDER,
    model: str = None,
) -> str:
    """สร้าง Brochure แบบ streaming (typewriter effect)"""
    print(f"\n⏳ กำลังสร้าง Brochure สำหรับ {company_name} ด้วย [{provider}] ...")
    print("=" * 60)

    template = _get_template("brochure_generator")
    client = manager.get_client(provider)

    full_response = ""
    for chunk in client.stream_chat(
        system_prompt=template.system,
        user_prompt=get_brochure_user_prompt(company_name, url, manager, provider),
        model=model,
    ):
        print(chunk, end="", flush=True)
        full_response += chunk

    print("\n" + "=" * 60)
    return full_response


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    manager = LLMManager()

    # ---- สร้าง Brochure แบบ streaming ด้วย Ollama ----
    stream_brochure("HuggingFace", "https://huggingface.co", manager)

    # ---- ลองเปลี่ยน provider หรือบริษัทได้ตามต้องการ ----
    # stream_brochure("HuggingFace", "https://huggingface.co", manager, provider="openai")
    # create_brochure("Edward Donner", "https://edwarddonner.com", manager)
