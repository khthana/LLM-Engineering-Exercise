# อธิบายโปรแกรม Exercise_2.6.py

## ความแตกต่างจาก 2.5

| จุด | Exercise_2.5 | Exercise_2.6 |
|-----|-------------|-------------|
| ข้อมูลสินค้า | Python dict `PRODUCTS` | SQLite `keycraft.db` |
| จำนวนสินค้า | 7 (4 kb + 3 mouse) | 16 (6 kb + 4 mouse + 3 headset + 3 mousepad) |
| Tools | `list_keyboards()` + `list_mice()` | `get_products(category)` ตัวเดียว |
| DB lifecycle | ไม่มี | persist ข้ามรัน, seed ถ้าตารางว่าง |

---

## โครงสร้างโปรแกรม

### Data Layer

**`ProviderConfig` (dataclass)**
เก็บ config ของแต่ละ AI provider — ชื่อ, ประเภท LangChain class, default model, API key, base URL

**`ChatResult` (dataclass)**
ห่อผลลัพธ์จาก LLM — ข้อความตอบ, provider, token count, error (ถ้ามี)

**`PROVIDER_REGISTRY` (dict)**
ตาราง config ของทุก provider (ollama, openai, claude, gemini, openrouter)

---

### Core Classes

**`LLMClient`**
Wrapper รอบ LangChain สำหรับ provider เดียว มี 3 method หลัก:
- `_make_llm()` — สร้าง LangChain object ตาม provider (ChatOpenAI, ChatAnthropic, ChatOllama ฯลฯ)
- `chat()` — เรียก LLM แบบ invoke ธรรมดา (non-streaming)
- `stream_messages()` — stream response ทีละ chunk

**`LLMManager`**
จัดการ `LLMClient` ทุกตัวพร้อมกัน มีหน้าที่:
- `_validate_and_init()` — ตรวจ API key ทุก provider ตอน startup
- `get_client(provider)` — ดึง client ของ provider ที่ต้องการ
- `available_providers` — list provider ที่พร้อมใช้

---

### SQLite Layer (ใหม่ใน 2.6)

**`SEED_DATA` (list of tuples)**
ข้อมูลสินค้า 16 รายการ 4 ประเภท พร้อม seed เข้า DB ตอนรันครั้งแรก

```python
# (id, category, name, price, original_price, description)
("KC-K1", "keyboards", "KC-K1 Pro 75% Wireless", 3490, None, "...")
("KC-M1", "mice",      "KC-M1 Wireless Ergonomic", 900, 1800, "...")
("KC-H1", "headsets",  "KC-H1 Gaming 7.1 Surround", 2490, 3100, "...")
("KC-P1", "mousepads", "KC-P1 XXL Speed Pad", 490, 700, "...")
```

**Schema ตาราง `products`:**
```sql
CREATE TABLE IF NOT EXISTS products (
    id             TEXT PRIMARY KEY,
    category       TEXT NOT NULL,      -- keyboards | mice | headsets | mousepads
    name           TEXT NOT NULL,
    price          INTEGER NOT NULL,
    original_price INTEGER,            -- NULL = ไม่มีส่วนลด
    description    TEXT NOT NULL
)
```

**`_get_conn()`**
เปิด SQLite connection ไปยัง `keycraft.db` ใน working directory

**`_init_db()`**
เรียกตอน import — สร้างตารางถ้ายังไม่มี, seed ข้อมูลถ้าตารางว่าง (COUNT = 0)

---

### Shop Layer

**`get_products(category)` (LangChain `@tool`)**
Tool เดียวที่รับ `category` เป็น parameter แทน tools แยกของ 2.5:
- Query DB ด้วย `WHERE category = ?`
- Format ราคาพร้อมแสดงราคาเดิมถ้ามี `original_price`
- Return string ให้ LLM นำไปตอบลูกค้า

**`SHOP_TOOLS`**
List ที่มีแค่ `[get_products]` — tool เดียวรองรับทุก category

---

### UI Layer

**`chat_fn()`**
Gradio callback หลัก — รับข้อความจาก user วน tool loop จนได้คำตอบ

**`update_model_dropdown()` / `update_reasoning_visibility()`**
Gradio event callbacks — อัพเดต UI เมื่อ user เปลี่ยน provider หรือ model

---

## LLM เรียก Tool เองไหม?

**โปรแกรมนี้เป็นคนเรียก** — LLM ส่ง intention กลับมาเป็น structured data:

```json
response.tool_calls = [
  { "name": "get_products", "args": {"category": "headsets"}, "id": "call_abc" }
]
```

โปรแกรมรับแล้วเรียก Python function จริงใน `chat_fn()`:

```python
for tc in response.tool_calls:
    if tc["name"] in tool_map:
        result = tool_map[tc["name"]].invoke(tc["args"])   # query SQLite จริง
        current_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
```

---

## กรณี LLM ถามสินค้าหลายประเภทพร้อมกัน

เมื่อถาม "มี keyboard และ headset อะไรบ้าง?" — LLM เรียก tool สองครั้งในรอบเดียว:

```json
response.tool_calls = [
  { "name": "get_products", "args": {"category": "keyboards"}, "id": "call_001" },
  { "name": "get_products", "args": {"category": "headsets"},  "id": "call_002" }
]
```

loop ใน `chat_fn` วนผ่านทุก tool_calls ก่อน invoke ครั้งถัดไป — ทำให้ tool เดียวรองรับได้ทุกกรณีโดยไม่ต้องเพิ่ม tool ใหม่เมื่อมี category เพิ่มขึ้น

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
llm_with_tools.invoke(messages)   ← LLM ตัดสินใจ category ที่ต้องการ
       │
   ┌───┴──────────────────────────────┐
   │ response.tool_calls มีข้อมูล     │ response.tool_calls ว่าง
   ▼                                 ▼
แสดง "✨ กำลังค้นหาข้อมูล..."      แสดงคำตอบ → จบ
       │
       ▼
get_products(category) → query SQLite
   (keycraft.db → return formatted string)
       │
       ▼
เพิ่ม ToolMessage เข้า messages
       │
       ▼
invoke() อีกครั้ง → LLM ใช้ผลลัพธ์ตอบ
```

กุญแจคือ `.bind_tools(SHOP_TOOLS)` + `while True` loop — LLM ตัดสินใจเองว่าจะเรียก category ใด และกี่ครั้ง โดยโปรแกรมเป็นคนรัน query จริงทุกครั้ง

---

## DB Lifecycle

```
รันครั้งแรก:
  _init_db() → CREATE TABLE → COUNT=0 → INSERT 16 rows → "seeded"

รันครั้งถัดไป:
  _init_db() → CREATE TABLE IF NOT EXISTS (ข้าม) → COUNT=16 → "loaded"
```

ไฟล์ `keycraft.db` อยู่ใน working directory และ persist ข้ามการรัน — สามารถลบไฟล์เพื่อ reset และ seed ใหม่ได้
