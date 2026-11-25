"use client";

import { FormEvent, useState } from "react";
import { chat } from "@/services/api";
import type { ChatResponse } from "@/services/api";
import AnswerMarkdown from "@/components/AnswerMarkdown";

interface ChatPanelProps {
  onResponse: (response: ChatResponse | null) => void;
  onCitationSelect: (evidenceId: string) => void;
  onConversationStart: () => void;
}

export default function ChatPanel({
  onResponse,
  onCitationSelect,
  onConversationStart,
}: ChatPanelProps) {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [submittedQuery, setSubmittedQuery] = useState<string | null>(null);
  const suggestions = [
    "Thời gian thử việc tối đa là bao lâu?",
    "Người lao động được nghỉ phép năm thế nào?",
    "Khi nào được đơn phương chấm dứt hợp đồng?",
  ];

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery || loading) return;

    setLoading(true);
    setError(null);
    setSubmittedQuery(trimmedQuery);
    setQuery("");
    onConversationStart();
    setResponse(null);
    try {
      const nextResponse = await chat(trimmedQuery);
      setResponse(nextResponse);
      onResponse(nextResponse);
    } catch (requestError) {
      const message =
        requestError instanceof Error ? requestError.message : "Request failed";
      setError(message);
      setResponse(null);
      onResponse(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel chat-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Trợ lý pháp luật lao động</p>
          <h1>Hỏi đáp cùng kho pháp luật</h1>
        </div>
        <span className="status-dot" aria-label="API ready" />
      </div>
      {!response && <><p className="muted">Đặt câu hỏi để tìm câu trả lời có căn cứ trong kho pháp luật Việt Nam.</p><div className="suggestions"><span>CÂU HỎI GỢI Ý</span>{suggestions.map((item) => <button type="button" key={item} onClick={() => setQuery(item)}>{item}</button>)}</div></>}

      <form onSubmit={handleSubmit} className="chat-form">
        <label htmlFor="query">Câu hỏi của bạn</label>
        <textarea
          id="query"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={submittedQuery ? "Tra cứu câu hỏi về luật lao động" : "Ví dụ: Thời gian thử việc tối đa là bao lâu?"}
          rows={5}
        />
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? "Đang tra cứu…" : "Gửi câu hỏi"}
        </button>
      </form>

      {error && <p className="error-message">{error}</p>}
      {submittedQuery && (
        <div className="conversation-stream">
          <div className="user-message"><span>{submittedQuery}</span></div>
          {loading && (
            <div className="assistant-progress" role="status" aria-live="polite">
              <span className="progress-mark">✦</span>
              <span>Đang tìm kiếm căn cứ pháp lý và chuẩn bị câu trả lời…</span>
              <span className="typing-dots" aria-hidden="true"><i /> <i /> <i /></span>
            </div>
          )}
        </div>
      )}
      {response && (
        <div className="answer-block">
          <div className="answer-label">Câu trả lời</div>
          {response.answer ? (
            <AnswerMarkdown
              answer={response.answer}
              evidence={response.evidence}
              onCitationSelect={onCitationSelect}
            />
          ) : (
            <p>Chưa có câu trả lời được tạo.</p>
          )}
        </div>
      )}
    </section>
  );
}
