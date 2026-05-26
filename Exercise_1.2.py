"""
Exercise_1.2.py
===============
Multi-provider LLM Manager — Modular / OOP Design

Features:
  - รองรับ OpenAI, Claude, Gemini, OpenRouter, Ollama
  - ตรวจสอบ API key จาก .env อัตโนมัติ
  - สร้าง client หลาย provider พร้อมกัน
  - ส่ง prompt (system + user) ไปหลาย model พร้อมกันแบบ parallel
  - Prompt Library สำหรับหลายวัตถุประสงค์
  - ออกแบบให้นำไปต่อยอดได้ง่าย
"""

import os
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from bs4 import BeautifulSoup
import urllib.request

load_dotenv()


# =============================================================================
# Data Classes
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
    """
    ระบุ provider + model ที่ต้องการยิงใน chat_parallel
    label ใช้เป็น key ใน dict ผลลัพธ์ (optional — ถ้าไม่ระบุจะสร้างให้อัตโนมัติ)

    ตัวอย่าง:
        ModelTarget("openrouter", "microsoft/phi-4-reasoning-plus:free")
        ModelTarget("openrouter", "google/gemma-3-27b-it:free", label="gemma-3")
        ModelTarget("ollama", "llama3.2")
        ModelTarget("ollama", "mistral")
    """
    provider: str
    model: str = None           # None = ใช้ default_model ของ provider
    label: str = None           # ชื่อที่ใช้เป็น key ผลลัพธ์

    def __post_init__(self):
        if self.label is None:
            model_part = self.model or "default"
            self.label = f"{self.provider}:{model_part}"


# =============================================================================
# Provider Registry
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
# Prompt Library
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
    "website_summary_formal": PromptTemplate(
        description="สรุปเว็บแบบเป็นทางการ (ภาษาไทย)",
        system=(
            "You are a professional assistant that analyzes website content "
            "and provides a concise, neutral summary in Thai. "
            "Focus on main purpose, key offerings, and notable announcements. "
            "Respond in markdown."
        ),
        user_prefix="Please summarize the following website content:\n\n",
    ),
    "explain_simple": PromptTemplate(
        description="อธิบายเรื่องซับซ้อนให้เข้าใจง่าย",
        system=(
            "You are a patient teacher who explains complex topics simply. "
            "Use analogies and plain language. Avoid jargon unless you define it first."
        ),
        user_prefix="Please explain the following in simple terms:\n\n",
    ),
}


# =============================================================================
# LLMClient — ห่อ client ของ provider เดียว
# =============================================================================

class LLMClient:
    """
    Wrapper สำหรับ OpenAI-compatible client ของ provider หนึ่งตัว
    ใช้งาน: client.chat(system_prompt, user_prompt)
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
    ) -> ChatResult:
        """ส่ง message ไปหา LLM แล้วคืน ChatResult"""
        selected_model = model or self.config.default_model
        try:
            response = self._client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
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

    def __repr__(self) -> str:
        return f"LLMClient(provider='{self.provider_id}', model='{self.config.default_model}')"


# =============================================================================
# LLMManager — จัดการ client หลายตัวพร้อมกัน
# =============================================================================

class LLMManager:
    """
    ตัวจัดการ LLM หลาย provider
    - ตรวจสอบ key อัตโนมัติตอน init
    - สร้าง client เฉพาะ provider ที่มี key ถูกต้อง
    - รองรับการยิง prompt ไปหลาย provider พร้อมกัน (parallel)
    """

    def __init__(self, registry: dict[str, ProviderConfig] = None):
        self._registry = registry or PROVIDER_REGISTRY
        self._clients: dict[str, LLMClient] = {}
        self._key_status: dict[str, dict] = {}
        self._validate_and_init()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _validate_key(
        self, provider_id: str, config: ProviderConfig
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """ตรวจสอบ API key → คืน (valid, api_key, reason)"""
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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available_providers(self) -> list[str]:
        """รายชื่อ provider ที่พร้อมใช้งาน"""
        return list(self._clients.keys())

    # ------------------------------------------------------------------
    # Core Chat Methods
    # ------------------------------------------------------------------

    def get_client(self, provider: str) -> LLMClient:
        """ดึง client ของ provider ที่ต้องการ"""
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
    ) -> ChatResult:
        """ส่ง prompt ไปหา provider เดียว"""
        return self.get_client(provider).chat(system_prompt, user_prompt, model)

    def chat_parallel(
        self,
        system_prompt: str,
        user_prompt: str,
        targets: list[ModelTarget] = None,
    ) -> dict[str, ChatResult]:
        """
        ส่ง prompt เดียวกันไปหลาย model พร้อมกัน (ThreadPoolExecutor)

        Args:
            targets : list ของ ModelTarget — รองรับ provider เดียวกันหลาย model
                      None = ใช้ default model ของทุก provider ที่พร้อม

        Returns:
            dict ที่ key คือ ModelTarget.label, value คือ ChatResult

        ตัวอย่าง:
            manager.chat_parallel(system, user, targets=[
                ModelTarget("openai"),
                ModelTarget("openrouter", "microsoft/phi-4-reasoning-plus:free"),
                ModelTarget("openrouter", "google/gemma-3-27b-it:free"),
                ModelTarget("ollama", "llama3.2"),
                ModelTarget("ollama", "mistral"),
            ])
        """
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

    # ------------------------------------------------------------------
    # Prompt Template Helpers
    # ------------------------------------------------------------------

    def chat_with_template(
        self,
        template_key: str,
        content: str,
        provider: str,
        model: str = None,
    ) -> ChatResult:
        """ส่ง content ไปพร้อม prompt template ที่กำหนดไว้ใน PROMPTS"""
        template = _get_template(template_key)
        return self.chat(
            system_prompt=template.system,
            user_prompt=template.user_prefix + content,
            provider=provider,
            model=model,
        )

    def compare_providers(
        self,
        template_key: str,
        content: str,
        targets: list[ModelTarget] = None,
    ) -> dict[str, ChatResult]:
        """เปรียบเทียบคำตอบจากหลาย provider/model ด้วย template เดียวกัน"""
        template = _get_template(template_key)
        return self.chat_parallel(
            system_prompt=template.system,
            user_prompt=template.user_prefix + content,
            targets=targets,
        )

    def __repr__(self) -> str:
        return f"LLMManager(available={self.available_providers})"


# =============================================================================
# Utility Functions
# =============================================================================

def _get_template(template_key: str) -> PromptTemplate:
    """ดึง PromptTemplate จาก PROMPTS พร้อม error message ที่ชัดเจน"""
    template = PROMPTS.get(template_key)
    if not template:
        available = list(PROMPTS.keys())
        raise ValueError(f"Unknown template: '{template_key}'. Available: {available}")
    return template


def print_result(result: ChatResult):
    """แสดงผล ChatResult ตัวเดียว"""
    config = PROVIDER_REGISTRY.get(result.provider)
    name = config.name if config else result.provider
    print(f"\n{'='*55}")
    print(f"  {name}  |  model: {result.model}")
    print(f"{'='*55}")
    if result.success:
        print(result.content)
    else:
        print(f"❌ Error: {result.error}")


def print_results(results: dict[str, ChatResult]):
    """แสดงผล ChatResult หลายตัวพร้อมกัน"""
    for result in results.values():
        print_result(result)


def list_prompts():
    """แสดง prompt template ที่มีทั้งหมด"""
    print("=== Available Prompt Templates ===")
    for key, template in PROMPTS.items():
        print(f"  {key:30s} — {template.description}")


# =============================================================================
# Web Scraper
# =============================================================================

def scrape_page(url: str) -> str:
    """
    ดึงเนื้อหาข้อความจากหน้าเว็บโดยใช้ Playwright
    รองรับเว็บที่ render ด้วย JavaScript (SPA, lazy load ฯลฯ)

    Args:
        url: URL ของเว็บที่ต้องการดึงข้อมูล

    Returns:
        เนื้อหาข้อความล้วน (stripped text) จาก body ของหน้าเว็บ
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "html.parser")

    # ตัด tag ที่ไม่มีประโยชน์ออก
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    return soup.get_text(separator="\n", strip=True)


def scrape_rss(rss_url: str) -> str:
    """ดึงข่าวจาก RSS feed — ใช้เป็น fallback สำหรับเว็บที่ block Playwright"""
    req = urllib.request.Request(
        rss_url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        xml = resp.read().decode("utf-8", errors="ignore")
    soup = BeautifulSoup(xml, "xml")
    items = soup.find_all("item")
    lines = []
    for item in items:
        title = item.find("title")
        desc = item.find("description")
        if title:
            lines.append(f"## {title.get_text(strip=True)}")
        if desc:
            lines.append(desc.get_text(strip=True))
        lines.append("")
    return "\n".join(lines)


# RSS feeds สำหรับเว็บที่ block scraping
RSS_FEEDS: dict[str, str] = {
    "bbc.com":            "https://feeds.bbci.co.uk/news/rss.xml",
    "bbc.co.uk":          "https://feeds.bbci.co.uk/news/rss.xml",
    "washingtonpost.com": "https://feeds.washingtonpost.com/rss/national",
}

# เว็บที่ block ทั้ง scraping และ RSS
BLOCKED_SITES: dict[str, str] = {
    "reuters.com": "Reuters ปิด public RSS feed แล้ว — ลองใช้ BBC, CNN, หรือ AP News แทน",
}

MIN_CONTENT_LENGTH = 500


def _fallback_to_rss(url: str, reason: str) -> str:
    """พยายามใช้ RSS แทน scraping — raise ValueError ถ้าไม่มี RSS สำหรับเว็บนั้น"""
    blocked_msg = next((msg for d, msg in BLOCKED_SITES.items() if d in url), None)
    if blocked_msg:
        raise ValueError(f"ไม่สามารถดึงข้อมูลจากเว็บนี้ได้ — {blocked_msg}")

    rss_domain = next((d for d in RSS_FEEDS if d in url), None)
    if rss_domain:
        print(f"⚠️  {reason} — เปลี่ยนเป็น RSS feed แทน")
        return scrape_rss(RSS_FEEDS[rss_domain])

    raise ValueError(f"{reason} — เว็บอาจบล็อก bot ลองใช้เว็บอื่นแทน")


def summarize_news_from_url(url: str, manager: "LLMManager", provider: str = "ollama") -> ChatResult:
    """ดึงข้อมูลจากเว็บแล้วสรุปข่าวเป็นภาษาไทย — ใช้ RSS อัตโนมัติถ้า scrape ไม่ได้"""
    print(f"\n⏳ กำลังดึงข้อมูลจาก {url} ...")

    try:
        page_text = scrape_page(url)
    except Exception as e:
        page_text = _fallback_to_rss(url, f"scrape ไม่สำเร็จ ({type(e).__name__})")

    if len(page_text) < MIN_CONTENT_LENGTH:
        page_text = _fallback_to_rss(url, f"ดึงข้อมูลได้น้อยเกินไป ({len(page_text)} ตัวอักษร)")

    print(f"✅ ดึงข้อมูลสำเร็จ ({len(page_text):,} ตัวอักษร)\n")
    print("⏳ กำลังสรุปข่าว ...")
    return manager.chat_with_template(
        template_key="news_summary_thai",
        content=page_text,
        provider=provider,
    )

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # สร้าง manager (validate key อัตโนมัติ)
    manager = LLMManager()
    list_prompts()

    # ---- สรุปข่าวจาก CNN ----
    # result1 = summarize_news_from_url("https://www.cnn.com", manager)
    # print_result(result1)
    result2 = summarize_news_from_url("https://www.thaipost.net/", manager)
    print_result(result2)

    

    # ---- เปรียบเทียบหลาย model (uncomment เพื่อใช้งาน) ----
    # print("\n⏳ เปรียบเทียบคำตอบจากหลาย model ...")
    # results = manager.compare_providers(
    #     template_key="news_summary_thai",
    #     content=page_text,
    #     targets=[
    #         ModelTarget("openrouter", "microsoft/phi-4-reasoning-plus:free", label="phi-4"),
    #         ModelTarget("openrouter", "google/gemma-3-27b-it:free",          label="gemma-3"),
    #         ModelTarget("ollama",     "llama3.2"),
    #     ],
    # )
    # print_results(results)
