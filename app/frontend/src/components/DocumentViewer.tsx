import type { ChatResponse } from "@/services/api";

export default function DocumentViewer({
  response,
  selectedEvidenceId,
}: {
  response: ChatResponse | null;
  selectedEvidenceId: string | null;
}) {
  return (
    <section className="panel evidence-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Traceability</p>
          <h2>Evidence & diagnostics</h2>
        </div>
        {response && (
          <span className="result-status">
            {response.debug.status || "complete"}
          </span>
        )}
      </div>

      {!response ? (
        <div className="empty-state">
          Submit a question to see the evidence returned by the retrieval
          pipeline.
        </div>
      ) : (
        <>
          <div className="evidence-list">
            <h3>Citations</h3>
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
            <summary>Pipeline diagnostics</summary>
            <pre>
              {JSON.stringify(response.debug.diagnostics || {}, null, 2)}
            </pre>
          </details>
        </>
      )}
    </section>
  );
}
