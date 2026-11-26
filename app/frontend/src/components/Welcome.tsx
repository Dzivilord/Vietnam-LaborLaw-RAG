import { BookIcon, ScaleIcon } from "@/components/Icons";

const TOPICS: { title: string; questions: string[] }[] = [
  {
    title: "Hợp đồng & thử việc",
    questions: ["Thời gian thử việc tối đa là bao lâu?", "Hợp đồng xác định thời hạn được ký tối đa bao lâu?"],
  },
  {
    title: "Tiền lương & làm thêm giờ",
    questions: ["Làm thêm giờ vào ngày lễ được trả lương thế nào?", "Công ty được trừ lương tối đa bao nhiêu?"],
  },
  {
    title: "Nghỉ phép & thai sản",
    questions: ["Người lao động được nghỉ phép năm bao nhiêu ngày?", "Lao động nữ được nghỉ thai sản bao lâu?"],
  },
  {
    title: "Chấm dứt hợp đồng",
    questions: ["Nghỉ việc phải báo trước bao nhiêu ngày?", "Trợ cấp thôi việc được tính như thế nào?"],
  },
];

export default function Welcome({ onAsk }: { onAsk: (query: string) => void }) {
  return (
    <section className="welcome" aria-labelledby="welcome-title">
      <div className="welcome-mark"><ScaleIcon size={28} /></div>
      <h1 id="welcome-title">Hỏi đáp Bộ luật Lao động 2019</h1>
      <p className="welcome-lead">
        Đặt câu hỏi bằng ngôn ngữ thường ngày. Mỗi câu trả lời đều dẫn chiếu Điều, Khoản cụ thể và
        cho phép mở văn bản gốc để kiểm chứng.
      </p>

      <ul className="welcome-points">
        <li><BookIcon /> Chỉ dựa trên Bộ luật Lao động 2019 (45/2019/QH14)</li>
        <li><BookIcon /> Trích dẫn rõ ràng, bấm để xem nguyên văn</li>
        <li><BookIcon /> Không đủ căn cứ thì nói rõ, không suy đoán</li>
      </ul>

      <div className="topic-grid">
        {TOPICS.map((topic) => (
          <div className="topic-card" key={topic.title}>
            <h2>{topic.title}</h2>
            {topic.questions.map((question) => (
              <button type="button" key={question} onClick={() => onAsk(question)}>
                {question}
              </button>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}
