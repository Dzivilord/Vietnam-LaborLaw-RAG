# Frontend — Next.js

Giao diện hỏi đáp cho người dùng cuối (`/`) và bảng quan sát nội bộ (`/admin`).
Next.js 16 (App Router), React 19, không dùng thư viện UI — chỉ CSS thuần.

> Phiên bản Next.js này có thay đổi so với các bản cũ. Trước khi sửa code, đọc
> tài liệu đi kèm trong `node_modules/next/dist/docs/` (xem `AGENTS.md`).

## Chạy

```powershell
cd app/frontend
npm ci              # cài đúng theo package-lock.json
npm run dev         # http://localhost:3000
npm run build       # build production + kiểm tra TypeScript (không có lint riêng)
```

Backend mặc định ở `http://localhost:8000`. Đổi bằng biến
`NEXT_PUBLIC_RAG_API_URL` (đặt trước `npm run dev`/`build`). Nếu đổi cổng
frontend, cần thêm origin vào CORS của backend.

## Giao diện chính (`/`)

- **Màn hình chào**: giới thiệu phạm vi và các câu hỏi mẫu theo chủ đề
  (bấm là gửi ngay).
- **Hội thoại nhiều lượt**: mỗi câu hỏi được tra cứu **độc lập** (backend không
  giữ ngữ cảnh hội thoại), các lượt hiển thị liên tiếp để xem lại.
- **Trích dẫn bấm được**: `[E1]` trong câu trả lời được hiển thị thành nhãn
  "Điều 25"; bấm để mở khung căn cứ và cuộn tới đúng điều luật.
- **Khung căn cứ pháp lý**: nguyên văn từng điều, chương/mục, đánh dấu điều được
  trích dẫn, thu gọn văn bản dài. Trên điện thoại mở toàn màn hình; `Esc` để đóng.
- **Trạng thái rõ ràng**: thông báo riêng cho *ngoài phạm vi*, *không đủ căn cứ*,
  *từ chối*, *lỗi kết nối* (có nút thử lại).
- **Câu hỏi quá chung**: hiển thị các chủ đề gợi ý (Điều + tiêu đề) dạng nút; bấm
  để gửi câu hỏi cụ thể hơn về điều đó.
- Sao chép câu trả lời, thời gian phản hồi, ô nhập tự giãn
  (Enter gửi, Shift+Enter xuống dòng), chế độ sáng/tối theo hệ điều hành.

## Cấu trúc

```text
src/
├── app/
│   ├── layout.tsx          # metadata, viewport, import globals.css
│   ├── page.tsx            # trang chính: quản lý các lượt hỏi đáp + khung căn cứ
│   ├── globals.css         # design tokens (sáng/tối) và style giao diện chính
│   └── admin/
│       ├── page.tsx        # bảng quan sát pipeline (trace, SSE, tuỳ chỉnh model)
│       └── admin.css       # style riêng, tách khỏi giao diện chính
├── components/
│   ├── Welcome.tsx         # màn hình chào + câu hỏi mẫu
│   ├── ChatTurn.tsx        # một lượt: câu hỏi, tiến trình, câu trả lời/thông báo
│   ├── AnswerMarkdown.tsx  # render markdown, biến [E1] thành nút trích dẫn
│   ├── EvidencePanel.tsx   # khung căn cứ pháp lý
│   ├── Composer.tsx        # ô nhập câu hỏi
│   └── Icons.tsx           # icon SVG nội tuyến
├── lib/status.ts           # ánh xạ status → thông báo; trích dẫn; định dạng Điều/Khoản
└── services/api.ts         # client gọi backend + kiểu dữ liệu ChatResponse
```

## Ghi chú thiết kế

- Phản hồi `/chat` có `status` và `suggestions`; `lib/status.ts` quyết định cách
  hiển thị từng trạng thái. Thêm trạng thái mới ở backend thì bổ sung tại đây.
- Trích dẫn được nhận diện bằng cách đổi `[E1]` thành link nội bộ `#cite-E1`
  trước khi render markdown, rồi thay link đó bằng nút — tránh phụ thuộc vào API
  nội bộ của `react-markdown`.
- Font: dùng font hệ thống hỗ trợ đầy đủ tiếng Việt. Tránh Georgia (thiếu glyph
  như "ề", "ấ" nên dấu hiển thị lệch).
- `/admin` không có xác thực; chỉ dùng nội bộ.
