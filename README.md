# Trợ lý Mua sắm Đa tác vụ VinShop (AI Shopping Assistant)

Dự án này triển khai một hệ thống Trợ lý Mua sắm thông minh đa tác vụ (**Multi-Agent System**) sử dụng **LangGraph**, **Chroma DB (RAG)** và giao diện trò chuyện trực quan **Chainlit**. Hệ thống được thiết kế để tự động định tuyến, tra cứu chính sách giao hàng/hoàn trả và truy vấn thông tin đơn hàng, khách hàng, voucher thực tế từ tập dữ liệu giả lập.

---

## 🏗️ Kiến trúc Hệ thống (Multi-Agent Architecture)

Hệ thống được tổ chức thành một luồng xử lý đồ thị tuần tự & có điều kiện bao gồm các Agent chuyên biệt:

```
                  ┌───────────────┐
                  │ User Request  │
                  └───────┬───────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Supervisor Agent │
                └─────────┬────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
 ┌────────────────────┐      ┌────────────────────┐
 │  Worker 1 (Policy) │      │   Worker 2 (Data)  │
 ├────────────────────┤      ├────────────────────┤
 │ RAG trên Policy MD │      │ Tra cứu 4 tools    │
 └──────────┬─────────┘      └──────────┬─────────┘
            │                           │
            └─────────────┬─────────────┘
                          ▼
                ┌──────────────────┐
                │ Worker 3 (Resp)  │
                └─────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │   Final Answer   │
                └──────────────────┘
```

1. **Supervisor Agent**: Nhận câu hỏi từ người dùng, trích xuất thực thể, xác định xem câu hỏi cần thông tin từ Chính sách (Policy), Dữ liệu hệ thống (Data), hay cả hai, sau đó định tuyến đến các Agent phù hợp. Tự động phát hiện thiếu thông tin định danh để hỏi lại (`clarification_needed`).
2. **Worker 1: Policy / RAG Agent**: Sử dụng RAG tìm kiếm trên tập chính sách của VinShop lưu trữ trong cơ sở dữ liệu vector Chroma.
3. **Worker 2: Order / Customer Lookup Agent**: Sử dụng 4 công cụ độc lập để tra cứu dữ liệu khách hàng, đơn hàng, và voucher trong tệp tin Mock dữ liệu.
4. **Worker 3: Response Agent**: Nhận kết xuất thông tin từ hai Worker trước, tổng hợp và định dạng câu trả lời kèm các trích dẫn tài liệu (Citations) và bằng chứng dữ liệu thực tế (Evidence) cụ thể.

---

## ✨ Các Tính năng Nổi bật & Cải tiến

*   **Giao diện Chainlit Hiện đại**: Cho phép trao đổi trực tiếp, đổi mô hình chat, điều chỉnh temperature, bật/tắt hiển thị bằng chứng (evidence) và đổi mô hình tạo ảnh trực quan từ Settings.
*   **Hiệu năng Chuyển đổi Tức thì (Global Caching)**: Áp dụng cơ chế cache toàn cục cho các đối tượng tải chậm (SentenceTransformers, Chroma DB, Mock Data JSON). Việc thay đổi mô hình LLM hoặc cấu hình không cần phải tải lại tài nguyên, giúp chuyển đổi mô hình chỉ trong tích tắc.
*   **Quy trình Tạo ảnh có Kế hoạch & Dữ liệu Thực tế**: Khi người dùng yêu cầu vẽ bảng đơn hàng hoặc sơ đồ:
    1.  Hệ thống tự động kiểm tra và truy xuất dữ liệu thật của khách hàng/đơn hàng từ database.
    2.  Đề xuất một **Kế hoạch tạo ảnh (Plan Proposal)** chi tiết và hỏi xác nhận từ người dùng qua nút bấm tương tác (`cl.AskActionMessage`).
    3.  Hiển thị **Tiến trình sinh động (Progress Updates)** 3 bước rõ ràng trong quá trình kết xuất.
    4.  Truyền tải dữ liệu thực tế vào prompt tạo ảnh để vẽ bảng biểu hiển thị thông tin thật thay vì dùng mã placeholder trống.
*   **Trace JSON Đầy đủ**: Tự động tích lũy nhật ký chạy chi tiết của từng Agent qua mỗi bước dưới định dạng JSON để debug.

---

## 🛠️ Công nghệ Sử dụng

*   **Core**: Python 3.12, LangGraph, LangChain Core
*   **Vector DB**: Chroma DB
*   **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (chạy hoàn toàn offline cục bộ)
*   **UI Framework**: Chainlit
*   **LLM Providers**: Hỗ trợ linh hoạt Gemini (`gemini-3.1-flash-lite`), OpenAI (`gpt-4.1-mini`), OpenRouter, Ollama, hoặc các mô hình Custom thông qua một lớp Abstraction sạch.

---

## 🚀 Hướng dẫn Cài đặt & Chạy ứng dụng

### 1. Chuẩn bị Môi trường

1.  Tạo tệp `.env` tại thư mục gốc của dự án với các thông số tối thiểu:

    ```env
    # Model Chat chính
    LLM_MODEL=gemini-3.1-flash-lite
    GOOGLE_API_KEY=your_google_api_key_here

    # Khóa kết nối tạo ảnh (Ví dụ qua OpenRouter)
    OPENROUTER_API_KEY=your_openrouter_api_key_here
    ```

2.  Cài đặt các gói thư viện phụ thuộc vào môi trường ảo:

    ```bash
    .venv/bin/pip install -r src/requirements.txt
    ```

### 2. Chạy Giao diện Chat UI (Chainlit)

Khởi động giao diện Web trực quan bằng lệnh:

```bash
.venv/bin/chainlit run src/ui/chainlit_app.py -w
```

Giao diện sẽ tự động mở tại địa chỉ `http://localhost:8000`. Đăng nhập bằng tài khoản mặc định:
*   **User**: `admin`
*   **Password**: `vinshop`

### 3. Chạy qua dòng lệnh (CLI - Single Question)

Để hỏi thử một câu thông qua CLI:

```bash
PYTHONPATH=src .venv/bin/python -m app.cli --question "Đơn hàng 1971 có được hoàn trả không?"
```

### 4. Chạy Batch Test (Kiểm thử Hàng loạt)

Để chạy kiểm thử tự động toàn bộ 22 câu hỏi mẫu trong `data/test.json` và xuất báo cáo kết quả cùng file trace JSON:

```bash
PYTHONPATH=src .venv/bin/python -m app.cli --batch
```

Báo cáo tóm tắt sẽ được lưu tại `src/artifacts/traces/summary.json`.
