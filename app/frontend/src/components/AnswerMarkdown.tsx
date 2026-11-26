import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import type { EvidenceCitation } from "@/services/api";
import { legalReference } from "@/lib/status";

interface AnswerMarkdownProps {
  answer: string;
  evidence: EvidenceCitation[];
  onCitationSelect: (evidenceId: string) => void;
}

const CITE_PREFIX = "#cite-";

/**
 * Rewrite "[E1]" markers into markdown links (`[E1](#cite-E1)`) so the markdown
 * parser keeps them as nodes; the `a` renderer below turns them into buttons.
 * Markers for unknown evidence IDs are left as plain text.
 */
function linkCitations(answer: string, known: Set<string>): string {
  return answer.replace(/\[(E\d+)\](?!\()/gi, (whole, id: string) =>
    known.has(id.toUpperCase()) ? `[${id.toUpperCase()}](${CITE_PREFIX}${id.toUpperCase()})` : whole,
  );
}

export default function AnswerMarkdown({ answer, evidence, onCitationSelect }: AnswerMarkdownProps) {
  const evidenceById = new Map(evidence.map((item) => [item.evidence_id.toUpperCase(), item]));

  const components: Components = {
    a: ({ href, children }) => {
      if (href?.startsWith(CITE_PREFIX)) {
        const evidenceId = href.slice(CITE_PREFIX.length);
        const item = evidenceById.get(evidenceId);
        if (item) {
          const reference = legalReference(item);
          return (
            <button
              className="citation"
              type="button"
              onClick={() => onCitationSelect(evidenceId)}
              title={`${reference}${item.article_title ? ` – ${item.article_title}` : ""}`}
              aria-label={`Mở căn cứ ${reference}`}
            >
              {reference}
            </button>
          );
        }
      }
      return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
    },
  };

  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {linkCitations(answer, new Set(evidenceById.keys()))}
      </ReactMarkdown>
    </div>
  );
}
