import type { ChatResponse } from "@/services/api";

export default function DocumentViewer({
  response,
  selectedEvidenceId,
  isOpen,
}: {
  response: ChatResponse | null;
  selectedEvidenceId: string | null;
  isOpen: boolean;
}) {
  return (
    <section className={`panel evidence-panel ${isOpen ? "is-open" : ""}`}>
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Kiểm chứng nguồn</p>
          <h2>Căn cứ pháp lý</h2>
        </div>
        {response && (
          <span className="result-status">
            {response.debug.status || "Hoàn tất"}
          </span>
        )}
      </div>

      {!response ? (
        <div className="empty-state">
          Gửi câu hỏi để xem căn cứ pháp lý.
        </div>
      ) : (
        <>
          <div className="evidence-list">
            <h3>Tài liệu tham chiếu</h3>
            {response.evidence.length ? (
              response.evidence.map((citation) => (
                <article
                  className={`evidence-item ${selectedEvidenceId === citation.evidence_id ? "is-selected" : ""}`}
                  id={`evidence-${citation.evidence_id}`}
                  key={citation.evidence_id}
                >
                  <div className="evidence-heading">
                    <strong>{citation.evidence_id}</strong>
                    <span>
                      {citation.article_number
                        ? `Điều ${citation.article_number}`
                        : citation.document_title || "Source chunk"}
                    </span>
                  </div>
                  <p className="evidence-content">{citation.content}</p>
                  <p className="evidence-meta">
                    {[
                      citation.chapter_title,
                      citation.section_title,
                      citation.article_title,
                      citation.page ? `Page ${citation.page}` : null,
                    ]
                      .filter(Boolean)
                      .join(" / ")}
                  </p>
                </article>
              ))
            ) : (
              <p className="muted">No evidence was returned.</p>
            )}
          </div>
          <details>
            <summary>Thông tin chẩn đoán pipeline</summary>
            <pre>
              {JSON.stringify(response.debug.diagnostics || {}, null, 2)}
            </pre>
          </details>
        </>
      )}
    </section>
  );
}
