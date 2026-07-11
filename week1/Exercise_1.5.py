"""
Exercise_1.5.py
===============
Company Brochure Generator — แปลงจาก day5.ipynb

สร้าง Brochure สำหรับบริษัทโดยอัตโนมัติ จากชื่อบริษัทและ URL หลัก

ขั้นตอนการทำงาน:
  1. ดึงลิงก์ทั้งหมดจากหน้าหลักของบริษัท
  2. ให้ LLM คัดเลือกลิงก์ที่เกี่ยวข้อง (About, Careers, ฯลฯ) → JSON
  3. ดึงเนื้อหาจากลิงก์ที่เลือก
  4. สร้าง Brochure แบบ Markdown ด้วย LLM
  5. แสดงผลแบบ streaming (typewriter effect) ในเทอร์มินัล
"""

import os
import json
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI

load_dotenv(override=True)


# =============================================================================
# Constants
# =============================================================================

MODEL_LINKS   = "gpt-4.1-nano"   # ใช้สำหรับคัดเลือกลิงก์ (เร็ว/ถูก)
MODEL_BROCHURE = "gpt-4.1-mini"  # ใช้สำหรับสร้าง Brochure (คุณภาพสูงกว่า)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}

MAX_CONTENT_CHARS = 2_000   # ตัดเนื้อหาแต่ละหน้าไม่เกิน 2,000 ตัวอักษร
MAX_PROMPT_CHARS  = 5_000   # ตัด user prompt ไม่เกิน 5,000 ตัวอักษร


# =============================================================================
# OpenAI Client
# =============================================================================

api_key = os.getenv("OPENAI_API_KEY")
if api_key and api_key.startswith("sk-proj-") and len(api_key) > 10:
    print("✅ API key looks good")
else:
    print("⚠️  API key อาจมีปัญหา — กรุณาตรวจสอบ .env")

openai = OpenAI()


# =============================================================================
# Web Scraper (รวม scraper.py ไว้ใน file เดียว)
# =============================================================================

def fetch_website_contents(url: str) -> str:
    """
    ดึง title + เนื้อหาข้อความจากหน้าเว็บ
    ตัดเนื้อหาเกิน MAX_CONTENT_CHARS ออก
    """
    response = requests.get(url, headers=HEADERS, timeout=15)
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
    response = requests.get(url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(response.content, "html.parser")
    links = [a.get("href") for a in soup.find_all("a")]
    return [link for link in links if link]


# =============================================================================
# Prompt Templates
# =============================================================================

LINK_SYSTEM_PROMPT = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

BROCHURE_SYSTEM_PROMPT = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""


# =============================================================================
# Step 1: Link Selection
# =============================================================================

def get_links_user_prompt(url: str) -> str:
    """สร้าง user prompt สำหรับให้ LLM คัดเลือกลิงก์"""
    prompt = (
        f"Here is the list of links on the website {url} -\n"
        "Please decide which of these are relevant web links for a brochure about the company,\n"
        "respond with the full https URL in JSON format.\n"
        "Do not include Terms of Service, Privacy, email links.\n\n"
        "Links (some might be relative links):\n\n"
    )
    prompt += "\n".join(fetch_website_links(url))
    return prompt


def select_relevant_links(url: str) -> dict:
    """
    เรียก LLM เพื่อคัดเลือกลิงก์ที่เกี่ยวข้องกับบริษัท
    คืน dict รูปแบบ {"links": [{"type": ..., "url": ...}, ...]}
    """
    print(f"  🔍 กำลังคัดเลือกลิงก์จาก {url} ด้วย {MODEL_LINKS} ...")
    response = openai.chat.completions.create(
        model=MODEL_LINKS,
        messages=[
            {"role": "system", "content": LINK_SYSTEM_PROMPT},
            {"role": "user",   "content": get_links_user_prompt(url)},
        ],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    print(f"  ✅ พบ {len(result.get('links', []))} ลิงก์ที่เกี่ยวข้อง")
    return result


# =============================================================================
# Step 2: Content Aggregation
# =============================================================================

def fetch_page_and_all_relevant_links(url: str) -> str:
    """
    รวมเนื้อหาจากหน้าหลัก + หน้าที่ LLM เลือกว่าเกี่ยวข้อง
    """
    contents = fetch_website_contents(url)
    relevant_links = select_relevant_links(url)

    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links.get("links", []):
        result += f"\n\n### Link: {link['type']}\n"
        try:
            result += fetch_website_contents(link["url"])
        except Exception as e:
            result += f"(ดึงไม่ได้: {e})"
    return result


# =============================================================================
# Step 3: Brochure Generation
# =============================================================================

def get_brochure_user_prompt(company_name: str, url: str) -> str:
    """สร้าง user prompt สำหรับสร้าง Brochure"""
    prompt = (
        f"You are looking at a company called: {company_name}\n"
        "Here are the contents of its landing page and other relevant pages; "
        "use this information to build a short brochure of the company in markdown without code blocks.\n\n"
    )
    prompt += fetch_page_and_all_relevant_links(url)
    return prompt[:MAX_PROMPT_CHARS]  # ตัดถ้าเกิน MAX_PROMPT_CHARS


def create_brochure(company_name: str, url: str) -> str:
    """
    สร้าง Brochure แบบปกติ (รอผลลัพธ์ครบแล้วค่อย print)
    คืน string เนื้อหา Brochure
    """
    print(f"\n⏳ กำลังสร้าง Brochure สำหรับ {company_name} ...")
    response = openai.chat.completions.create(
        model=MODEL_BROCHURE,
        messages=[
            {"role": "system", "content": BROCHURE_SYSTEM_PROMPT},
            {"role": "user",   "content": get_brochure_user_prompt(company_name, url)},
        ],
    )
    result = response.choices[0].message.content
    print(result)
    return result


def stream_brochure(company_name: str, url: str) -> str:
    """
    สร้าง Brochure แบบ streaming (typewriter effect ในเทอร์มินัล)
    คืน string เนื้อหา Brochure ทั้งหมด
    """
    print(f"\n⏳ กำลังสร้าง Brochure สำหรับ {company_name} (streaming) ...")
    print("=" * 60)

    stream = openai.chat.completions.create(
        model=MODEL_BROCHURE,
        messages=[
            {"role": "system", "content": BROCHURE_SYSTEM_PROMPT},
            {"role": "user",   "content": get_brochure_user_prompt(company_name, url)},
        ],
        stream=True,
    )

    full_response = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        full_response += delta

    print("\n" + "=" * 60)
    return full_response


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # ---- ทดสอบ: สร้าง Brochure แบบ streaming ----
    stream_brochure("HuggingFace", "https://huggingface.co")

    # ---- ลองเปลี่ยนบริษัทได้ตามต้องการ ----
    # stream_brochure("Edward Donner", "https://edwarddonner.com")
    # create_brochure("HuggingFace", "https://huggingface.co")
