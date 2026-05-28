# อธิบายโปรแกรม Exercise_2.5.py

## โครงสร้างโปรแกรม

### Data Layer

**`ProviderConfig` (dataclass)**
เก็บ config ของแต่ละ AI provider — ชื่อ, ประเภท LangChain class, default model, API key ที่ต้องใช้, base URL

**`ChatResult` (dataclass)**
ห่อผลลัพธ์จาก LLM — ข้อความตอบ, provider, token count, error (ถ้ามี)

**`PROVIDER_REGISTRY` (dict)**
ตาราง config ของทุก provider (openai, claude, gemini, openrouter, ollama) — เป็นแค่ข้อมูล ยังไม่ connect อะไร

---

### Core Classes

**`LLMClient`**
Wrapper รอบ LangChain สำหรับ provider เดียว มี 3 method หลัก:
- `_make_llm()` — สร้าง LangChain object ตาม provider (ChatOpenAI, ChatAnthropic, ChatOllama, ฯลฯ)
- `chat()` — เรียก LLM แบบ invoke ธรรมดา (non-streaming)
- `stream_messages()` — stream response ทีละ chunk

**`LLMManager`**
จัดการ `LLMClient` ทุกตัวพร้อมกัน มีหน้าที่:
- `_validate_and_init()` — ตรวจ API key ทุก provider ตอน startup แล้วสร้าง client เฉพาะตัวที่ผ่าน
- `get_client(provider)` — ดึง client ของ provider ที่ต้องการ
- `available_providers` — list provider ที่พร้อมใช้

---

### Shop Layer

**`PRODUCTS` (dict)**
ข้อมูลสินค้าดิบ (keyboards 4 รุ่น, mice 3 รุ่น) — อยู่ใน Python dict ล้วนๆ

**`list_keyboards()` / `list_mice()` (LangChain `@tool`)**
ดึงข้อมูลจาก `PRODUCTS` แล้ว format เป็น string ส่งกลับ — โปรแกรมเป็นคนเรียก ไม่ใช่ LLM

**`SHOP_TOOLS`**
List รวม tools ทั้งสอง ใช้ตอน bind กับ LLM

---

### UI Layer

**`chat_fn()`**
Gradio callback หลัก — รับข้อความจาก user แล้วส่งให้ LLM พร้อม tool loop

**`update_model_dropdown()` / `update_reasoning_visibility()`**
Gradio event callbacks — อัพเดต UI เมื่อ user เปลี่ยน provider หรือ model

---

## LLM เรียก Tool เองไหม?

**โปรแกรมนี้เป็นคนเรียก** — LLM ไม่ได้ execute tool เอง

LLM แค่ส่ง **intention** กลับมาในรูป structured data:

```json
response.tool_calls = [
  {
    "name": "list_keyboards",
    "args": {},
    "id": "call_abc123"
  }
]
```

LLM บอกว่า "ฉันอยากเรียก `list_keyboards`" แต่ไม่ได้รัน Python เอง

จากนั้นโปรแกรมเป็นคนรับแล้วเรียกจริงใน `chat_fn()`:

```python
for tc in response.tool_calls:
    if tc["name"] in tool_map:
        result = tool_map[tc["name"]].invoke(tc["args"])  # โปรแกรมเรียก
        current_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
```

แล้วส่งผล (`ToolMessage`) กลับให้ LLM รู้ว่าได้ข้อมูลอะไรมา

---

## Flow การทำงาน

```
User พิมพ์คำถาม
       │
       ▼
chat_fn() สร้าง llm_with_tools
   = client._make_llm().bind_tools(SHOP_TOOLS)
       │
       ▼
llm_with_tools.invoke(messages)   ← LLM ตัดสินใจว่าต้องเรียก tool ไหม
       │
   ┌───┴──────────────────────────────┐
   │ response.tool_calls มีข้อมูล     │ response.tool_calls ว่าง
   ▼                                 ▼
แสดง "✨ กำลังค้นหาข้อมูล..."      แสดงคำตอบ → จบ
       │
       ▼
เรียก list_keyboards() หรือ list_mice()
   (อ่านจาก PRODUCTS dict → return string)
       │
       ▼
เพิ่ม ToolMessage (ผลลัพธ์จาก tool) เข้า messages
       │
       ▼
invoke() อีกครั้ง → LLM ใช้ผลลัพธ์ tool ตอบ
```

กุญแจสำคัญคือ `.bind_tools(SHOP_TOOLS)` ที่ทำให้ LLM รู้ว่ามี tools ใดบ้าง — LLM เป็นคนตัดสินใจเองว่าจะเรียก tool หรือตอบตรงๆ โดยดูจาก system prompt และคำถามของ user

นี่คือเหตุผลที่ต้องมี `while True` loop — เพราะโปรแกรมต้องวน invoke จนกว่า LLM จะตอบเป็น text ธรรมดา (ไม่มี tool_calls อีก)
