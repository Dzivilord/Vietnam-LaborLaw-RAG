"use client";

import { useEffect, useState } from "react";
import { CloseIcon } from "@/components/Icons";
import { citedEvidenceIds, legalReference } from "@/lib/status";
import type { ChatResponse, EvidenceCitation } from "@/services/api";

interface EvidencePanelProps {
  query: string | null;
  response: ChatResponse | null;
  selectedEvidenceId: string | null;
  onClose: () => void;
}

const COLLAPSE_AT = 700; // characters before an evidence card is collapsed

export default function EvidencePanel({ query, response, selectedEvidenceId, onClose }: EvidencePanelProps) {
  useEffect(() => {
    if (!selectedEvidenceId) return;
    document.getElementById(`evidence-${selectedEvidenceId}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [selectedEvidenceId, response]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const evidence = response?.evidence ?? [];
  const cited = response ? citedEvidenceIds(response) : new Set<string>();
  // Cited evidence first, keeping the legal order within each group.
  const ordered = [...evidence].sort(
    (a, b) => Number(cited.has(b.evidence_id.toUpperCase())) - Number(cited.has(a.evidence_id.toUpperCase())),
  );

  return (
    <aside className="evidence" aria-label="Căn cứ pháp lý">
      <header className="evidence-header">
        <div>
          <p className="kicker">Kiểm chứng nguồn</p>
          <h2>Căn cứ pháp lý</h2>
          {query && <p className="evidence-query" title={query}>{query}</p>}
        </div>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Đóng khung căn cứ">
          <CloseIcon />
        </button>
      </header>

      <div className="evidence-scroll">
        {ordered.length === 0 ? (
          <p className="evidence-empty">Câu hỏi này không có căn cứ nào được sử dụng.</p>
        ) : (
          ordered.map((item) => (
            <EvidenceCard
              key={item.evidence_id}
              item={item}
              isCited={cited.has(item.evidence_id.toUpperCase())}
              isSelected={selectedEvidenceId === item.evidence_id.toUpperCase()}
            />
          ))
        )}
        <p className="evidence-footnote">
          Nguồn: Bộ luật Lao động 2019 (Luật số 45/2019/QH14). Đối chiếu văn bản gốc trước khi áp dụng.
        </p>
      </div>
    </aside>
  );
}

function EvidenceCard({ item, isCited, isSelected }: { item: EvidenceCitation; isCited: boolean; isSelected: boolean }) {
  const long = item.content.length > COLLAPSE_AT;
  const [expanded, setExpanded] = useState(!long);
  useEffect(() => {
    if (isSelected) setExpanded(true);
  }, [isSelected]);

  const breadcrumb = [
    item.chapter_number ? `Chương ${item.chapter_number}` : null,
    item.chapter_title,
    item.section_title,
  ].filter(Boolean).join(" · ");

  return (
    <article id={`evidence-${item.evidence_id.toUpperCase()}`} className={`evidence-card ${isSelected ? "is-selected" : ""}`}>
      <div className="evidence-card-head">
        <strong>{legalReference(item)}</strong>
        {isCited && <span className="tag">Được trích dẫn</span>}
      </div>
      {item.article_title && <h3>{item.article_title}</h3>}
      {breadcrumb && <p className="breadcrumb">{breadcrumb}</p>}
      <p className={`evidence-text ${expanded ? "" : "is-collapsed"}`}>{item.content}</p>
      {long && (
        <button type="button" className="link-button" onClick={() => setExpanded((value) => !value)}>
          {expanded ? "Thu gọn" : "Xem toàn văn"}
        </button>
      )}
    </article>
  );
}
