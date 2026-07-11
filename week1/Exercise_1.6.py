"""
Exercise_1.6.py
===============
Thai News Summarizer — 3-Level Scraping + Date Filter + Category Grouping

ดึงข่าวจาก thaipost.net แบบ 3 ระดับ:
  Level 1 : โครงสร้างหมวดหมู่ (static — ไม่ต้องใช้ LLM)
  Level 2 : หน้า Category → คัดบทความของวันนี้ด้วย BeautifulSoup
  Level 3 : บทความแต่ละชิ้น → scrape เนื้อหา

LLM ใช้สำหรับสรุปเนื้อหาเท่านั้น (ไม่ใช้ LLM เลือก URL)
"""

import os
import re
import json
import concurrent.futures
import requests as http_requests
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Iterator
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI

load_dotenv()


# =============================================================================
# Constants
# =============================================================================

TODAY = date.today()   # วันที่ปัจจุบัน สำหรับ filter ข่าววันนี้

MAX_ARTICLES_PER_CAT = 3
MAX_ARTICLE_CHARS    = 3_000
DEFAULT_PROVIDER     = "ollama"

SCRAPER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}

# ------------------------------------------------------------------
# โครงสร้างเว็บ thaipost.net (static — ไม่ต้องให้ LLM คาดเดา)
# ------------------------------------------------------------------
SITE_CATEGORIES = [
    {"name": "การเมือง",    "url": "https://www.thaipost.net/politics/"},
    {"name": "เศรษฐกิจ",   "url": "https://www.thaipost.net/economy/"},
    {"name": "ต่างประเทศ",  "url": "https://www.thaipost.net/abroad/"},
    {"name": "บันเทิง",     "url": "https://www.thaipost.net/entertainment/"},
]

# เดือนไทย → เลขเดือน (ทั้งแบบย่อและแบบเต็ม)
THAI_MONTHS: dict[str, int] = {
    # แบบย่อ
    "ม.ค.": 1,  "ก.พ.": 2,  "มี.ค.": 3, "เม.ย.": 4,
    "พ.ค.": 5,  "มิ.ย.": 6, "ก.ค.": 7,  "ส.ค.": 8,
    "ก.ย.": 9,  "ต.ค.": 10, "พ.ย.": 11, "ธ.ค.": 12,
    # แบบเต็ม
    "มกราคม": 1,    "กุมภาพันธ์": 2, "มีนาคม": 3,    "เมษายน": 4,
    "พฤษภาคม": 5,   "มิถุนายน": 6,   "กรกฎาคม": 7,   "สิงหาคม": 8,
    "กันยายน": 9,   "ตุลาคม": 10,    "พฤศจิกายน": 11, "ธันวาคม": 12,
}


# =============================================================================
# Data Classes  (จาก Exercise_1.5v2)
# =============================================================================

@dataclass
class ProviderConfig:
    name: str
    base_url: Optional[str]
    default_model: str
    env_var: Optional[str] = None
    prefix: Optional[str] = None
    requires_key: bool = True


@dataclass
class PromptTemplate:
    system: str
    user_prefix: str = ""
    description: str = ""


@dataclass
class ChatResult:
    provider: str
    model: str
    content: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class ModelTarget:
    provider: str
    model: str = None
    label: str = None

    def __post_init__(self):
        if self.label is None:
            self.label = f"{self.provider}:{self.model or 'default'}"


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

    "news_category_summarizer": PromptTemplate(
        description="สรุปบทความข่าวหลายชิ้นในหมวดเดียวกัน (ภาษาไทย)",
        system=(
            "คุณเป็นนักข่าวมืออาชีพที่สรุปข่าวได้กระชับ ตรงประเด็น และเป็นกลาง\n\n"
            "งานของคุณ:\n"
            "1. อ่านเนื้อหาบทความข่าวแต่ละชิ้นที่ให้มา\n"
            "2. สรุปแต่ละบทความเป็นภาษาไทย 2-3 ประโยค เน้นประเด็นสำคัญ\n"
            "3. จัดรูปแบบดังนี้:\n\n"
            "### 📰 [หัวข้อข่าวจากบทความ]\n"
            "[สรุปเนื้อหา 2-3 ประโยค]\n\n"
            "กฎเคร่งครัด:\n"
            "- ใช้ภาษาไทยทั้งหมด\n"
            "- สรุปเฉพาะเนื้อหาที่อยู่ในบทความ อย่าเพิ่มเนื้อหาที่ไม่มีในข้อมูล\n"
            "- กระชับ ไม่ฟุ่มเฟือย\n"
            "- ตอบ Markdown ตรงๆ ห้ามใส่ใน code block"
        ),
        user_prefix="",  # สร้างใน summarize_category() โดยตรง
    ),
}


# =============================================================================
# LLMClient  (จาก Exercise_1.5v2 + stream_chat)
# =============================================================================

class LLMClient:
    def __init__(self, provider_id: str, config: ProviderConfig, api_key: str):
        self.provider_id = provider_id
        self.config = config
        self._client = OpenAI(api_key=api_key, base_url=config.base_url)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = None,
    ) -> ChatResult:
        selected_model = model or self.config.default_model
        try:
            response = self._client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
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
    ) -> ChatResult:
        return self.get_client(provider).chat(system_prompt, user_prompt, model)

    def __repr__(self) -> str:
        return f"LLMManager(available={self.available_providers})"


# =============================================================================
# Utility
# =============================================================================

def _get_template(key: str) -> PromptTemplate:
    t = PROMPTS.get(key)
    if not t:
        raise ValueError(f"Unknown template: '{key}'. Available: {list(PROMPTS.keys())}")
    return t


def normalize_url(href: str, base_url: str) -> str:
    return urljoin(base_url, href) if href else ""


def print_separator(title: str = "", width: int = 60):
    if title:
        pad = max(0, width - len(title) - 2)
        print(f"\n{'─' * (pad // 2)} {title} {'─' * (pad - pad // 2)}")
    else:
        print("─" * width)


# =============================================================================
# Date Parsing  (รองรับ <time datetime>, วันที่ภาษาไทย, ISO ใน URL)
# =============================================================================

def parse_thai_date(text: str) -> Optional[date]:
    """
    แปลงข้อความวันที่ภาษาไทย → date object
    รองรับ: "5 พฤษภาคม 2569", "5 พ.ค. 2569"
    หมายเหตุ: ปี พ.ศ. (เช่น 2569) แปลงเป็น ค.ศ. โดยลบ 543
    """
    # สร้าง pattern จาก key ทั้งหมดใน THAI_MONTHS (escape . สำหรับแบบย่อ)
    month_pattern = "|".join(re.escape(m) for m in THAI_MONTHS)
    pattern = rf"(\d{{1,2}})\s+({month_pattern})\s+(\d{{4}})"

    m = re.search(pattern, text)
    if not m:
        return None
    try:
        day   = int(m.group(1))
        month = THAI_MONTHS[m.group(2)]
        year  = int(m.group(3))
        # ถ้าปีมากกว่า 2500 ถือว่าเป็น พ.ศ. → แปลงเป็น ค.ศ.
        if year > 2500:
            year -= 543
        return date(year, month, day)
    except (ValueError, KeyError):
        return None


def extract_date_from_element(element) -> Optional[date]:
    """
    ดึงวันที่จาก HTML element โดยลองหลาย strategy
    1. <time datetime="YYYY-MM-DD">
    2. ข้อความวันที่ภาษาไทย
    3. ISO date ใน URL
    """
    # Strategy 1: <time datetime="...">
    time_tag = element.find("time")
    if time_tag:
        dt_str = time_tag.get("datetime", "")
        if dt_str:
            try:
                return datetime.fromisoformat(dt_str[:10]).date()
            except ValueError:
                pass
        # ลองอ่านข้อความใน <time>
        thai = parse_thai_date(time_tag.get_text())
        if thai:
            return thai

    # Strategy 2: ข้อความวันที่ภาษาไทยใน element
    thai = parse_thai_date(element.get_text())
    if thai:
        return thai

    # Strategy 3: ISO date ใน href URL (เช่น /2026/05/05/)
    link = element.find("a")
    if link:
        href = link.get("href", "")
        m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", href)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass

    return None


# =============================================================================
# Web Scraper
# =============================================================================

def fetch_page_soup(url: str) -> Optional[BeautifulSoup]:
    """
    ดึง HTML ด้วย Playwright (รองรับ JS-rendered content)
    ถ้า Playwright ไม่ได้ติดตั้ง จะ fallback ไปใช้ requests
    """
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=SCRAPER_HEADERS["User-Agent"])
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(3_000)   # รอ JS render เพิ่ม 3 วิ
            html = page.content()
            browser.close()
        return BeautifulSoup(html, "html.parser")
    except ImportError:
        # Fallback: requests (ถ้าไม่มี Playwright)
        try:
            resp = http_requests.get(url, headers=SCRAPER_HEADERS, timeout=15)
            resp.encoding = "utf-8"
            return BeautifulSoup(resp.content, "html.parser")
        except Exception as e:
            print(f"    ⚠️  fetch_page_soup({url}): {e}")
            return None
    except Exception as e:
        print(f"    ⚠️  fetch_page_soup({url}): {e}")
        return None


def fetch_article_text(url: str) -> str:
    """ดึงเนื้อหาบทความ ตัด tag ที่ไม่จำเป็น"""
    soup = fetch_page_soup(url)
    if not soup:
        return "(ดึงข้อมูลไม่ได้)"

    title = soup.title.string.strip() if soup.title else ""

    for tag in soup(["script", "style", "noscript", "nav",
                     "header", "footer", "aside", "img", "input", "iframe"]):
        tag.decompose()

    # พยายามหา main content ก่อน
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(class_=re.compile(r"content|article|post|entry", re.I))
        or soup.body
    )

    body_text = main.get_text(separator="\n", strip=True) if main else ""
    full = f"{title}\n\n{body_text}" if title else body_text
    return full[:MAX_ARTICLE_CHARS]


# =============================================================================
# Level 2 : คัดบทความของวันนี้จากหน้า Category (ไม่ใช้ LLM)
# =============================================================================

def fetch_today_articles(category: dict) -> list[dict]:
    """
    ดึงบทความจากหน้า category โดย:
    1. กรอง URL ให้เป็น subpath ของ category เท่านั้น (ไม่หลุดหมวดอื่น)
    2. Parse วันที่ และกรองเฉพาะบทความของวันนี้
    3. ถ้าไม่มีบทความวันนี้ → ใช้ MAX_ARTICLES_PER_CAT บทความล่าสุด

    คืน: [{"url": ..., "headline": ..., "date": ...}, ...]
    """
    cat_url  = category["url"]
    cat_name = category["name"]

    print(f"\n  📂 หมวด [{cat_name}] — ดึงข้อมูลจาก {cat_url}")

    soup = fetch_page_soup(cat_url)
    if not soup:
        return []

    base_domain = urlparse(cat_url).netloc
    seen_urls   = set()
    candidates  = []

    # URL ที่ควรกรองออก (nav, pagination, ฯลฯ)
    # หมายเหตุ: thaipost ใช้ path ต่างจาก category URL
    # เช่น category /politics/ → บทความอยู่ที่ /politics-news/990951/
    SKIP_PATTERNS = re.compile(
        r"/(tag|author|page|search|feed|login|register|wp-admin|"
        r"contact|about|privacy|terms|sitemap|coming-soon|search-news|"
        r"thaipost-tv|news-paper|articles)/",
        re.I,
    )
    NAV_URLS = {
        cat_url.rstrip("/"), cat_url,
        f"https://{base_domain}", f"https://{base_domain}/",
    }

    def is_article_url(url: str) -> bool:
        """
        ตรวจว่า URL น่าจะเป็นบทความ
        ไม่บังคับ subpath ของ category เพราะ thaipost ใช้ path ต่างกัน
        เช่น /politics/ → บทความที่ /politics-news/990951/
        """
        if not url.startswith("http"):
            return False
        parsed = urlparse(url)
        if parsed.netloc != base_domain:
            return False
        if url in seen_urls or url in NAV_URLS:
            return False
        path = parsed.path.rstrip("/")
        segments = [s for s in path.split("/") if s]
        # บทความต้องมีอย่างน้อย 2 segments: /category-name/article-id/
        if len(segments) < 2:
            return False
        if SKIP_PATTERNS.search(path):
            return False
        return True

    # ── Strategy 1: หา <article> หรือ card containers ────────────────
    containers = (
        soup.find_all("article")
        or soup.find_all(class_=re.compile(
            r"post[-_]?item|news[-_]?item|article[-_]?item|entry[-_]?item|"
            r"card|block[-_]?news|list[-_]?item", re.I
        ))
    )

    if containers:
        for el in containers:
            link_tag = el.find("a", href=True)
            if not link_tag:
                continue
            url = normalize_url(link_tag["href"], cat_url)
            if not is_article_url(url):
                continue
            seen_urls.add(url)
            headline = (
                (el.find(["h1", "h2", "h3", "h4"]) or link_tag)
                .get_text(strip=True)
            )
            art_date = extract_date_from_element(el)
            candidates.append({"url": url, "headline": headline[:120], "date": art_date})

    # ── Strategy 2: Fallback — scan ทุก <a> ในหน้า ───────────────────
    if not candidates:
        for a in soup.find_all("a", href=True):
            url = normalize_url(a["href"], cat_url)
            if not is_article_url(url):
                continue
            seen_urls.add(url)

            # หาวันที่จาก parent (ขึ้นสูงสุด 6 ระดับ)
            parent   = a.parent
            art_date = None
            for _ in range(6):
                if parent is None:
                    break
                art_date = extract_date_from_element(parent)
                if art_date:
                    break
                parent = parent.parent

            candidates.append({
                "url":      url,
                "headline": a.get_text(strip=True)[:120],
                "date":     art_date,
            })

    print(f"     พบ {len(candidates)} บทความในหมวดนี้ทั้งหมด")

    # ── กรองวันนี้ ─────────────────────────────────────────────────────
    today_articles = [a for a in candidates if a.get("date") == TODAY]

    if today_articles:
        result = today_articles[:MAX_ARTICLES_PER_CAT]
        print(f"     ✅ เลือก {len(result)} บทความของวันนี้ ({TODAY})")
    else:
        result = candidates[:MAX_ARTICLES_PER_CAT]
        has_date = sum(1 for a in result if a.get("date"))
        print(
            f"     ⚠️  ไม่พบบทความของวันนี้ ({TODAY}) "
            f"— ใช้ {len(result)} บทความล่าสุดแทน"
            + (f" (parse วันที่ได้ {has_date}/{len(result)})" if has_date else "")
        )

    for art in result:
        date_str = str(art["date"]) if art.get("date") else "ไม่ทราบวันที่"
        print(f"       • [{date_str}] {art['headline'][:55]}")

    return result


# =============================================================================
# Level 3 : Scrape Article Content (parallel)
# =============================================================================

def scrape_articles(articles: list[dict]) -> list[dict]:
    """Scrape เนื้อหาบทความทุกชิ้นพร้อมกัน"""

    def _fetch(art: dict) -> dict:
        content = fetch_article_text(art["url"])
        return {**art, "content": content}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_fetch, art): art for art in articles}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


# =============================================================================
# LLM Summarization : สรุปบทความในหมวด (streaming)
# =============================================================================

def summarize_category(
    category_name: str,
    articles: list[dict],
    manager: LLMManager,
    provider: str = DEFAULT_PROVIDER,
    stream: bool = True,
) -> str:
    """
    ให้ LLM สรุปบทความทุกชิ้นในหมวด
    stream=True → typewriter effect ในเทอร์มินัล
    """
    template = _get_template("news_category_summarizer")

    user_prompt = (
        f"หมวดหมู่: **{category_name}**\n"
        f"วันที่: {TODAY}\n"
        f"จำนวนบทความ: {len(articles)} ชิ้น\n\n"
        "กรุณาสรุปแต่ละบทความตามเนื้อหาที่ให้มาเท่านั้น:\n\n"
    )
    for i, art in enumerate(articles, 1):
        headline = art.get("headline") or f"บทความ {i}"
        content  = art.get("content", "(ไม่มีเนื้อหา)")
        date_str = str(art["date"]) if art.get("date") else "ไม่ทราบ"
        user_prompt += f"---\n**บทความที่ {i}** | วันที่: {date_str}\nหัวข้อ: {headline}\n{content}\n\n"

    if stream:
        client = manager.get_client(provider)
        full = ""
        for chunk in client.stream_chat(template.system, user_prompt):
            print(chunk, end="", flush=True)
            full += chunk
        print()
        return full
    else:
        result = manager.chat(template.system, user_prompt, provider)
        return result.content if result.success else f"❌ {result.error}"


# =============================================================================
# Main Pipeline
# =============================================================================

def run_news_summary(
    categories: list[dict] = None,
    manager: LLMManager = None,
    provider: str = DEFAULT_PROVIDER,
    stream: bool = True,
):
    """
    Pipeline หลัก:
      Level 1: โครงสร้าง category (static)
      Level 2: ดึงบทความวันนี้จากแต่ละ category
      Level 3: scrape เนื้อหา → LLM สรุป
    """
    if manager is None:
        manager = LLMManager()
    if categories is None:
        categories = SITE_CATEGORIES

    print_separator("Thai News Summarizer — thaipost.net", width=65)
    print(f"วันที่: {TODAY}  |  Provider: [{provider}]  |  บทความ/หมวด: {MAX_ARTICLES_PER_CAT}")
    print(f"หมวดหมู่: {[c['name'] for c in categories]}\n")

    all_summaries = []

    for category in categories:
        cat_name = category["name"]

        # ── Level 2: คัดบทความวันนี้ ─────────────────────────────────
        articles = fetch_today_articles(category)
        if not articles:
            print(f"     ⚠️  ไม่พบบทความในหมวด {cat_name} — ข้าม")
            continue

        # ── Level 3: Scrape เนื้อหา ──────────────────────────────────
        print(f"     ⬇️  กำลัง scrape {len(articles)} บทความ ...")
        articles_with_content = scrape_articles(articles)

        # ── Summarize ─────────────────────────────────────────────────
        print_separator(f"🤖 สรุปหมวด: {cat_name}", width=65)
        summary = summarize_category(
            cat_name, articles_with_content, manager, provider, stream=stream
        )
        all_summaries.append({"category": cat_name, "summary": summary})

    # ── แสดงผลรวม ────────────────────────────────────────────────────
    print_separator("สรุปข่าวประจำวัน — thaipost.net", width=65)
    for item in all_summaries:
        print(f"\n## 📌 หมวด: {item['category']}")
        print_separator(width=50)
        print(item["summary"])

    print_separator(width=65)
    print(f"✅ สรุปครบ {len(all_summaries)}/{len(categories)} หมวด  |  วันที่: {TODAY}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    manager = LLMManager()

    run_news_summary(
        categories=SITE_CATEGORIES,
        manager=manager,
        provider=DEFAULT_PROVIDER,
        stream=True,
    )

    # ── ลองเลือกเฉพาะบางหมวด ──
    # run_news_summary(
    #     categories=[
    #         {"name": "การเมือง",  "url": "https://www.thaipost.net/politics/"},
    #         {"name": "เศรษฐกิจ", "url": "https://www.thaipost.net/economy/"},
    #     ],
    #     manager=manager,
    #     provider="openai",   # เปลี่ยน provider ได้
    # )
