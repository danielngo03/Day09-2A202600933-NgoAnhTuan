from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
import json
import mimetypes
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any

from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(ROOT_DIR / ".env")

import chainlit as cl
import httpx
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients.base import BaseStorageClient
from chainlit.input_widget import Select, Slider, Switch, Tab
from chainlit.user import User

from app.config import Settings
from app.graph import ShoppingAssistant, _extract_entities


CHAINLIT_DB_PATH = ROOT_DIR / "src" / "artifacts" / "chainlit_history.db"
CHAINLIT_DB_URL = f"sqlite+aiosqlite:///{CHAINLIT_DB_PATH}"
CHAINLIT_FILES_DIR = ROOT_DIR / "src" / "artifacts" / "chainlit_files"


MODEL_SETTINGS: dict[str, dict[str, str]] = {
    "Gemini - gemini-3.1-flash-lite": {
        "provider": "gemini",
        "model": "gemini-3.1-flash-lite",
    },
    "OpenAI - gpt-4.1-mini": {
        "provider": "openai",
        "model": "gpt-4.1-mini",
    },
    "OpenRouter - openai/gpt-4.1-mini": {
        "provider": "openrouter",
        "model": "openai/gpt-4.1-mini",
    },
    "OpenRouter - Nemotron 3.5 Content Safety (free)": {
        "provider": "openrouter",
        "model": "nvidia/nemotron-3.5-content-safety:free",
    },
}

PROVIDER_PROFILES: dict[str, dict[str, str]] = {
    "gemini_flash": {
        "label": "Gemini - gemini-3.1-flash-lite",
        "provider": "gemini",
        "model": "gemini-3.1-flash-lite",
    },
    "openai_mini": {
        "label": "OpenAI - gpt-4.1-mini",
        "provider": "openai",
        "model": "gpt-4.1-mini",
    },
    "openrouter_openai_mini": {
        "label": "OpenRouter - openai/gpt-4.1-mini",
        "provider": "openrouter",
        "model": "openai/gpt-4.1-mini",
    },
    "openrouter_nemotron_safety": {
        "label": "OpenRouter - Nemotron 3.5 Content Safety (free)",
        "provider": "openrouter",
        "model": "nvidia/nemotron-3.5-content-safety:free",
    },
}

EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
]

IMAGE_MODELS = [
    "sourceful/riverflow-v2.5-pro:free",
]

STARTERS = [
    "Chính sách hoàn trả hàng ra sao?",
    "Đơn hàng 1971 bao giờ được giao?",
    "Đơn hàng 1971 có được hoàn trả không?",
    "Voucher của khách hàng C001 còn những mã nào dùng được?",
]


class SQLiteChainlitDataLayer(SQLAlchemyDataLayer):
    """Small SQLite adapter for Chainlit's SQLAlchemy data layer."""

    async def execute_sql(
        self,
        query: str,
        parameters: dict[str, Any],
    ) -> list[dict[str, Any]] | int | None:
        cleaned = {key: _sqlite_value(value) for key, value in parameters.items()}
        return await super().execute_sql(query, cleaned)


class LocalFileStorageClient(BaseStorageClient):
    """Persist Chainlit elements to local files for this lab."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    async def upload_file(
        self,
        object_key: str,
        data: bytes | str,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
        content_disposition: str | None = None,
    ) -> dict[str, Any]:
        path = self._path_for_key(object_key)
        if path.exists() and not overwrite:
            return {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_bytes(data)
        path.with_suffix(path.suffix + ".meta.json").write_text(
            json.dumps({"mime": mime}, ensure_ascii=False),
            encoding="utf-8",
        )
        return {"object_key": object_key, "url": await self.get_read_url(object_key)}

    async def delete_file(self, object_key: str) -> bool:
        path = self._path_for_key(object_key)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        return True

    async def get_read_url(self, object_key: str) -> str:
        path = self._path_for_key(object_key)
        if not path.exists():
            return object_key
        mime = self._mime_for_path(path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    async def close(self) -> None:
        return None

    def _path_for_key(self, object_key: str) -> Path:
        clean_parts = [part for part in Path(object_key).parts if part not in {"", ".", ".."}]
        path = (self.root / Path(*clean_parts)).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise ValueError("Invalid storage object key")
        return path

    def _mime_for_path(self, path: Path) -> str:
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("mime"):
                return str(metadata["mime"])
        return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


@cl.data_layer
def get_data_layer() -> SQLAlchemyDataLayer:
    _ensure_chainlit_sqlite_schema()
    return SQLiteChainlitDataLayer(
        conninfo=CHAINLIT_DB_URL,
        storage_provider=LocalFileStorageClient(CHAINLIT_FILES_DIR),
    )


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> User | None:
    expected_user = os.getenv("CHAINLIT_AUTH_USER", "admin")
    expected_password = os.getenv("CHAINLIT_AUTH_PASSWORD", "vinshop")
    if username == expected_user and password == expected_password:
        return User(
            identifier=username,
            display_name="VinShop Admin",
            metadata={"role": "local_lab_user"},
        )
    return None


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _ensure_chainlit_sqlite_schema() -> None:
    CHAINLIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CHAINLIT_DB_PATH) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                "id" TEXT PRIMARY KEY,
                "identifier" TEXT NOT NULL UNIQUE,
                "metadata" TEXT NOT NULL,
                "createdAt" TEXT
            );

            CREATE TABLE IF NOT EXISTS threads (
                "id" TEXT PRIMARY KEY,
                "createdAt" TEXT,
                "name" TEXT,
                "userId" TEXT,
                "userIdentifier" TEXT,
                "tags" TEXT,
                "metadata" TEXT,
                FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS steps (
                "id" TEXT PRIMARY KEY,
                "name" TEXT NOT NULL,
                "type" TEXT NOT NULL,
                "threadId" TEXT NOT NULL,
                "parentId" TEXT,
                "streaming" BOOLEAN NOT NULL,
                "waitForAnswer" BOOLEAN,
                "isError" BOOLEAN,
                "metadata" TEXT,
                "tags" TEXT,
                "input" TEXT,
                "output" TEXT,
                "createdAt" TEXT,
                "command" TEXT,
                "start" TEXT,
                "end" TEXT,
                "generation" TEXT,
                "showInput" TEXT,
                "language" TEXT,
                "indent" INTEGER,
                "defaultOpen" BOOLEAN,
                "autoCollapse" BOOLEAN,
                "modes" TEXT,
                FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS elements (
                "id" TEXT PRIMARY KEY,
                "threadId" TEXT,
                "type" TEXT,
                "url" TEXT,
                "chainlitKey" TEXT,
                "name" TEXT NOT NULL,
                "display" TEXT,
                "objectKey" TEXT,
                "size" TEXT,
                "page" INTEGER,
                "language" TEXT,
                "forId" TEXT,
                "mime" TEXT,
                "props" TEXT,
                FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feedbacks (
                "id" TEXT PRIMARY KEY,
                "forId" TEXT NOT NULL,
                "threadId" TEXT NOT NULL,
                "value" INTEGER NOT NULL,
                "comment" TEXT,
                FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
            );
            """
        )
        _ensure_sqlite_column(connection, "steps", "autoCollapse", "BOOLEAN")


def _ensure_sqlite_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    column_type: str,
) -> None:
    rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    existing_columns = {row[1] for row in rows}
    if column not in existing_columns:
        connection.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {column_type}')


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = await _send_chat_settings()
    await _reset_session(settings)
    await cl.Message(
        content=(
            "Mình đã sẵn sàng với RAG policy, dữ liệu đơn hàng/khách hàng/voucher và tạo ảnh qua OpenRouter.\n\n"
            "Mở **Settings** để đổi model, embedding, temperature, top-k hoặc bật/tắt evidence. "
            "Gõ `/clear` để xóa lịch sử hội thoại trong session hiện tại."
        ),
        author="VinShop Assistant",
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: dict[str, Any]) -> None:
    settings = await _send_chat_settings()
    await _reset_session(settings)
    restored_history: list[dict[str, Any]] = []
    for step in thread.get("steps", []):
        step_type = str(step.get("type", ""))
        role = "assistant" if step_type == "assistant_message" else "user"
        content = step.get("output") or step.get("input") or ""
        if content:
            restored_history.append({"role": role, "content": content})
    cl.user_session.set("history", restored_history)


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    await _reset_session(settings)
    model = settings.get("Model", next(iter(MODEL_SETTINGS)))
    await cl.Message(
        content=f"Đã chuyển sang **{model}**.",
        author="System",
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    command = message.content.strip().lower()
    if command == "/clear":
        cl.user_session.set("history", [])
        await cl.Message(
            content="Đã xóa lịch sử hội thoại trong session hiện tại. Các thread đã lưu vẫn nằm trong History của Chainlit.",
            author="System",
        ).send()
        return
    if command == "/settings":
        await _send_chat_settings()
        await cl.Message(
            content="Mình đã mở lại Chat Settings. Bạn có thể đổi model, temperature, embedding, top-k và image model ở nút settings của khung nhập.",
            author="System",
        ).send()
        return

    assistant = cl.user_session.get("assistant")
    if assistant is None:
        await _reset_session(cl.user_session.get("chat_settings") or _default_settings())
        assistant = cl.user_session.get("assistant")

    history = cl.user_session.get("history") or []
    history.append({"role": "user", "content": message.content})
    cl.user_session.set("history", history)

    if _is_image_request(message.content):
        await _handle_image_request(message.content, assistant)
        return

    thinking = cl.Message(content="Đang định tuyến qua LangGraph...", author="VinShop Assistant")
    await thinking.send()

    payload = await asyncio.to_thread(assistant.ask, message.content)
    final_answer = payload["final_answer"]

    history.append(
        {
            "role": "assistant",
            "content": final_answer,
            "status": payload["status"],
            "route": payload["route"].get("workers", []),
        }
    )
    cl.user_session.set("history", history)

    thinking.content = _format_chat_answer(payload)
    await thinking.update()


async def _send_chat_settings() -> dict[str, Any]:
    return await cl.ChatSettings(
        [
            Tab(
                id="model",
                label="Model",
                inputs=[
                    Select(
                        id="Model",
                        label="Chat model",
                        values=list(MODEL_SETTINGS),
                        initial_index=0,
                        tooltip="LLM dùng cho supervisor, policy worker, data worker và response worker.",
                    ),
                    Slider(
                        id="Temperature",
                        label="Temperature",
                        initial=0,
                        min=0,
                        max=1,
                        step=0.1,
                    ),
                ],
            ),
            Tab(
                id="rag",
                label="RAG",
                inputs=[
                    Select(
                        id="Embedding",
                        label="Embedding model",
                        values=EMBEDDING_MODELS,
                        initial_index=0,
                        tooltip="Embedding dùng để index/search policy trong Chroma.",
                    ),
                    Slider(
                        id="TopK",
                        label="Policy top-k",
                        initial=4,
                        min=2,
                        max=8,
                        step=1,
                    ),
                    Switch(
                        id="ShowEvidence",
                        label="Show evidence",
                        initial=True,
                    ),
                ],
            ),
            Tab(
                id="image",
                label="Image",
                inputs=[
                    Select(
                        id="ImageModel",
                        label="Image model",
                        values=IMAGE_MODELS,
                        initial_index=0,
                        tooltip="Model OpenRouter dùng khi người dùng yêu cầu tạo ảnh/sơ đồ.",
                    ),
                ],
            ),
        ]
    ).send()


async def _reset_session(settings: dict[str, Any]) -> None:
    model_label = settings.get("Model") or next(iter(MODEL_SETTINGS))
    model_config = MODEL_SETTINGS.get(model_label, next(iter(MODEL_SETTINGS.values())))
    base = Settings.load()
    configured = replace(
        base,
        provider=model_config["provider"],
        model=model_config["model"],
        raw_model=model_config["model"],
        temperature=float(settings.get("Temperature", base.temperature)),
        embedding_model_name=str(settings.get("Embedding", base.embedding_model_name)),
        top_k=int(settings.get("TopK", base.top_k)),
    )
    assistant = await asyncio.to_thread(ShoppingAssistant, configured)
    cl.user_session.set("assistant", assistant)
    cl.user_session.set("model_label", model_label)
    cl.user_session.set("show_evidence", bool(settings.get("ShowEvidence", True)))
    cl.user_session.set("image_model", settings.get("ImageModel", IMAGE_MODELS[0]))
    cl.user_session.set("history", cl.user_session.get("history") or [])


def _default_settings() -> dict[str, Any]:
    return {
        "Model": next(iter(MODEL_SETTINGS)),
        "Temperature": 0,
        "Embedding": EMBEDDING_MODELS[0],
        "TopK": 4,
        "ShowEvidence": True,
        "ImageModel": IMAGE_MODELS[0],
    }


def _format_chat_answer(payload: dict[str, Any]) -> str:
    answer = payload.get("final_answer", "")
    if answer.startswith("Status: clarification_needed"):
        question = answer.split("Question:", 1)[-1].strip()
        return f"Mình cần thêm thông tin để kiểm tra chính xác: {question}"
    if answer.startswith("Status: not_found"):
        message = answer.split("Message:", 1)[-1].strip()
        return message or "Mình chưa tìm thấy dữ liệu phù hợp."
    if answer.startswith("Answer:"):
        answer = answer.removeprefix("Answer:").strip()
        evidence_marker = "\nEvidence:"
        evidence = ""
        if evidence_marker in answer:
            answer, evidence = answer.split(evidence_marker, 1)
        if cl.user_session.get("show_evidence") and evidence.strip():
            return f"{answer.strip()}\n\n<details><summary>Evidence</summary>\n\n{evidence.strip()}\n\n</details>"
        return answer.strip()
    return answer


def _is_image_request(text: str) -> bool:
    lowered = text.lower()
    return any(
        keyword in lowered
        for keyword in [
            "tạo ảnh",
            "vẽ ảnh",
            "tạo hình",
            "vẽ sơ đồ",
            "tạo sơ đồ",
            "minh họa",
            "generate image",
            "draw",
            "diagram",
        ]
    )


async def _handle_image_request(question: str, assistant: ShoppingAssistant) -> None:
    # 1. Check database for actual data
    history = cl.user_session.get("history") or []
    
    # Try to extract entities from current question
    entities = _extract_entities(question)
    customer_id = entities["customer_ids"][0] if entities["customer_ids"] else None
    order_id = entities["order_ids"][0] if entities["order_ids"] else None
    
    # If not found in current question, look in chat history backwards
    if not customer_id and not order_id:
        for msg in reversed(history):
            msg_content = msg.get("content", "")
            hist_entities = _extract_entities(msg_content)
            if hist_entities["customer_ids"] and not customer_id:
                customer_id = hist_entities["customer_ids"][0]
            if hist_entities["order_ids"] and not order_id:
                order_id = hist_entities["order_ids"][0]
                
    # Define what data we will use
    target_desc = ""
    data_summary = ""
    formatted_data_for_prompt = ""
    
    if order_id:
        # Check order data
        order_res = assistant.data_store.get_order_detail_by_order_id(order_id)
        if order_res["status"] == "ok":
            o = order_res["order"]
            target_desc = f"chi tiết đơn hàng #{order_id} của khách hàng {o.get('customer_name')}"
            data_summary = f"Đơn hàng #{order_id} (Trạng thái: {o.get('order_status')}, Tổng tiền: {o.get('final_total'):,} VND)"
            
            # Format detailed order data for prompt
            items_str = ", ".join([f"{item.get('product_name')} x{item.get('quantity')}" for item in o.get("items", [])])
            formatted_data_for_prompt = (
                f"Order ID: #{o.get('order_id')}\n"
                f"Customer: {o.get('customer_name')} ({o.get('customer_id')})\n"
                f"Status: {o.get('order_status')}\n"
                f"Payment Method: {o.get('payment_method')} ({o.get('payment_status')})\n"
                f"Carrier: {o.get('carrier')} (Tracking: {o.get('tracking_number')})\n"
                f"Created At: {o.get('created_at')}\n"
                f"Destination: {o.get('destination_district')}, {o.get('destination_city')}\n"
                f"Total Price: {o.get('final_total'):,} VND\n"
                f"Items: {items_str}"
            )
        else:
            target_desc = f"đơn hàng #{order_id}"
            data_summary = f"Đơn hàng #{order_id} (Không tìm thấy dữ liệu trong hệ thống)"
            formatted_data_for_prompt = f"Order #{order_id} not found."
    else:
        # Fallback to customer_id or default to C001
        if not customer_id:
            customer_id = "C001"
            
        cust_res = assistant.data_store.get_customer_by_id(customer_id)
        if cust_res["status"] == "ok":
            c = cust_res["customer"]
            target_desc = f"danh sách đơn hàng của khách hàng {c.get('customer_name')} ({customer_id})"
            
            orders_res = assistant.data_store.get_orders_by_customer_id(customer_id)
            orders_list = orders_res.get("orders", [])
            data_summary = f"Khách hàng {customer_id} ({c.get('customer_name')}, Hạng: {c.get('tier')}) có {len(orders_list)} đơn hàng."
            
            # Format customer orders for prompt
            orders_details = []
            for o in orders_list:
                items_str = ", ".join([f"{item.get('product_name')} x{item.get('quantity')}" for item in o.get("items", [])])
                orders_details.append(
                    f"- Đơn #{o.get('order_id')}: {o.get('order_status')} | Tổng: {o.get('final_total'):,} VND | Giao hàng qua: {o.get('carrier')} ({o.get('tracking_number')}) | Sản phẩm: {items_str}"
                )
            formatted_data_for_prompt = (
                f"Customer: {c.get('customer_name')} ({customer_id}), Hạng: {c.get('tier')}\n"
                "Danh sách đơn hàng:\n" + "\n".join(orders_details)
            )
        else:
            target_desc = f"khách hàng {customer_id}"
            data_summary = f"Khách hàng {customer_id} (Không tìm thấy dữ liệu)"
            formatted_data_for_prompt = f"Customer {customer_id} not found."

    # 2. Propose Plan and ask for confirmation
    plan_message_content = (
        f"### 📋 Kế hoạch tạo ảnh danh sách đơn hàng\n\n"
        f"Để đảm bảo hình ảnh liên quan chính xác tới hệ thống, mình đã lập kế hoạch tạo ảnh sau:\n"
        f"- **Đối tượng dữ liệu**: {target_desc}\n"
        f"- **Thông tin thực tế**: {data_summary}\n"
        f"- **Giao diện**: Thiết kế bảng danh sách đơn hàng hiện đại bằng tiếng Việt với các trạng thái thật.\n\n"
        f"Bạn có đồng ý tiến hành tạo ảnh theo kế hoạch này không?"
    )
    
    actions = [
        cl.Action(name="confirm_image", value="confirm", label="Đồng ý tạo ảnh"),
        cl.Action(name="cancel_image", value="cancel", label="Hủy bỏ"),
    ]
    
    choice = await cl.AskActionMessage(
        content=plan_message_content,
        actions=actions,
        author="VinShop Assistant"
    ).send()
    
    if not choice or choice.get("value") == "cancel":
        await cl.Message(
            content="❌ Đã hủy yêu cầu tạo ảnh. Bạn có thể yêu cầu tạo ảnh khác hoặc cung cấp thêm thông tin.",
            author="VinShop Assistant"
        ).send()
        return
        
    # 3. User confirmed: Start progress display
    status_message = cl.Message(content="⏳ Bắt đầu quá trình tạo ảnh...", author="VinShop Assistant")
    await status_message.send()
    
    await asyncio.sleep(0.6)
    status_message.content = "🔍 **Bước 1/3**: Đang truy xuất dữ liệu thực tế từ hệ thống..."
    await status_message.update()
    
    await asyncio.sleep(0.8)
    status_message.content = "🎨 **Bước 2/3**: Đang nạp dữ liệu thật và gửi yêu cầu tạo ảnh tới mô hình tạo ảnh..."
    await status_message.update()
    
    # 4. Generate prompt with real data
    prompt = (
        "Create a clean, modern Vietnamese web dashboard UI of an order list for a shopping app named 'VinShop'.\n"
        "The visual design must be premium, professional, using curated harmony colors (blue and white theme) and clear Vietnamese typography.\n"
        "The screen MUST display the following REAL system data exactly, with correct order IDs, statuses, tracking numbers, and total amounts (in VND):\n\n"
        f"{formatted_data_for_prompt}\n\n"
        "Do NOT use placeholders like #DH000000, --/--/---- or generic placeholder text. Make sure all numbers and names are readable and match the data provided."
    )
    
    image_result = await _generate_openrouter_image(prompt)
    
    await asyncio.sleep(0.6)
    status_message.content = "⚡ **Bước 3/3**: Đang xử lý kết quả ảnh phản hồi từ mô hình..."
    await status_message.update()
    
    if image_result.get("status") != "ok":
        status_message.content = _fallback_diagram_message(question, image_result["message"])
        await status_message.update()
        return

    # Success: display the image
    image_element = _image_element_from_data_url(image_result["image_url"])
    status_message.content = f"✅ Đây là bảng đơn hàng được vẽ từ dữ liệu thực tế của {target_desc}."
    status_message.elements = [image_element]
    await status_message.update()


async def _generate_openrouter_image(prompt: str) -> dict[str, Any]:
    base_settings = Settings.load()
    if not base_settings.openrouter_api_key:
        return {
            "status": "error",
            "message": "Chưa có `OPENROUTER_API_KEY` hợp lệ trong `.env` nên chưa thể tạo ảnh.",
        }

    model = cl.user_session.get("image_model") or IMAGE_MODELS[0]
    headers = {
        "Authorization": f"Bearer {base_settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    if base_settings.openrouter_site_url:
        headers["HTTP-Referer"] = base_settings.openrouter_site_url
    if base_settings.openrouter_app_name:
        headers["X-Title"] = base_settings.openrouter_app_name

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{base_settings.openrouter_base_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "modalities": _openrouter_image_modalities(model),
                "image_config": {
                    "aspect_ratio": "16:9",
                    "image_size": "1K",
                },
                "messages": [{"role": "user", "content": prompt}],
            },
        )
    if response.status_code >= 400:
        return {
            "status": "error",
            "message": f"OpenRouter image request failed: {response.status_code} {response.text[:300]}",
        }
    payload = response.json()
    message = payload.get("choices", [{}])[0].get("message", {})
    images = message.get("images") or []
    if not images:
        return {
            "status": "error",
            "message": "Riverflow không trả về ảnh trong response.",
        }
    return {
        "status": "ok",
        "text": message.get("content") or "",
        "image_url": _extract_openrouter_image_url(images[0]),
    }


def _openrouter_image_modalities(model: str) -> list[str]:
    normalized = model.lower()
    image_only_prefixes = (
        "sourceful/",
        "black-forest-labs/",
        "recraft/",
        "stability-ai/",
    )
    if normalized.startswith(image_only_prefixes) or "flux" in normalized:
        return ["image"]
    return ["image", "text"]


def _extract_openrouter_image_url(image_payload: dict[str, Any]) -> str:
    image_url = image_payload.get("image_url") or image_payload.get("imageUrl") or {}
    return image_url.get("url", "")


def _image_element_from_data_url(data_url: str) -> cl.Image:
    if data_url.startswith("data:") and ";base64," in data_url:
        header, encoded = data_url.split(";base64,", 1)
        mime = header.removeprefix("data:")
        content = base64.b64decode(encoded)
        return cl.Image(name="riverflow-output", content=content, mime=mime, display="inline", size="large")
    return cl.Image(name="riverflow-output", url=data_url, display="inline", size="large")


def _needs_shopping_context(text: str) -> bool:
    lowered = text.lower()
    return any(
        keyword in lowered
        for keyword in [
            "c0",
            "đơn",
            "order",
            "voucher",
            "policy",
            "chính sách",
            "giao hàng",
            "hoàn trả",
            "khách hàng",
            "dữ liệu",
        ]
    )


def _default_visual_context() -> str:
    return (
        "Sơ đồ tổng quan hệ thống VinShop Shopping Assistant: User -> Supervisor Agent -> "
        "Policy/RAG Worker và Data Lookup Worker -> Response Agent -> Final Answer. "
        "Policy dùng Chroma + all-MiniLM-L6-v2; data lookup dùng mock order/customer/voucher JSON."
    )


def _fallback_diagram_message(question: str, error_message: str) -> str:
    return (
        "OpenRouter chưa trả ảnh cho model hiện tại nên mình dựng tạm sơ đồ Mermaid để bạn vẫn có output dùng được.\n\n"
        f"Lý do kỹ thuật: `{error_message}`\n\n"
        "```mermaid\n"
        "flowchart LR\n"
        "    U[User request] --> S[Supervisor Agent]\n"
        "    S -->|policy intent| P[Policy / RAG Worker]\n"
        "    S -->|data intent| D[Order / Customer / Voucher Worker]\n"
        "    P --> R[Response Agent]\n"
        "    D --> R\n"
        "    R --> A[Final Vietnamese Answer]\n"
        "    P -.-> C[(Chroma + all-MiniLM-L6-v2)]\n"
        "    D -.-> J[(Local mock JSON)]\n"
        "```\n\n"
        f"Prompt gốc: `{question}`"
    )
