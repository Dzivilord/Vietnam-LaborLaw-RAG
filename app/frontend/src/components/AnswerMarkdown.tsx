import { Fragment, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import type { EvidenceCitation } from "@/services/api";

interface AnswerMarkdownProps {
  answer: string;
  evidence: EvidenceCitation[];
  onCitationSelect: (evidenceId: string) => void;
}

function CitationText({
  children,
  evidenceIds,
  onCitationSelect,
}: {
  children?: ReactNode;
  evidenceIds: Set<string>;
  onCitationSelect: (evidenceId: string) => void;
}) {
  if (typeof children !== "string") return children;

  const parts = children.split(/(\[E\d+\])/gi);
  return parts.map((part, index) => {
    const match = part.match(/^\[(E\d+)\]$/i);
    const evidenceId = match?.[1].toUpperCase();
    if (!evidenceId || !evidenceIds.has(evidenceId)) {
      return <Fragment key={`${part}-${index}`}>{part}</Fragment>;
    }

    return (
      <button
        className="citation-badge"
        key={`${evidenceId}-${index}`}
        type="button"
        onClick={() => onCitationSelect(evidenceId)}
        aria-label={`Open evidence ${evidenceId}`}
      >
        {evidenceId}
      </button>
    );
  });
}

export default function AnswerMarkdown({
  answer,
  evidence,
  onCitationSelect,
}: AnswerMarkdownProps) {
  const evidenceIds = new Set(
    evidence.map((item) => item.evidence_id.toUpperCase()),
  );
  const components: Components = {
    text: ({ children }) => (
      <CitationText
        evidenceIds={evidenceIds}
        onCitationSelect={onCitationSelect}
      >
        {children}
      </CitationText>
    ),
  };

  return (
    <div className="answer-markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {answer}
      </ReactMarkdown>
    </div>
  );
}
