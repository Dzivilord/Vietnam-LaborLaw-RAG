"use client";

import { useEffect, useState } from "react";
import AnswerMarkdown from "@/components/AnswerMarkdown";
import { AlertIcon, BookIcon, CheckIcon, CopyIcon, RetryIcon, ScaleIcon } from "@/components/Icons";
import { citedEvidenceIds, formatSeconds, legalReference, statusMeta, statusOf, suggestionQuery } from "@/lib/status";
import type { ChatResponse } from "@/services/api";

export interface Turn {
  id: string;
  query: string;
  state: "loading" | "done" | "error";
  response?: ChatResponse;
  error?: string;
}

interface ChatTurnProps {
  turn: Turn;
  isActive: boolean;
  onCitationSelect: (turnId: string, evidenceId: string) => void;
  onOpenEvidence: (turnId: string) => void;
  onAsk: (query: string) => void;
  onRetry: (turn: Turn) => void;
}

const PROGRESS_STEPS = [
  "Đang tìm các điều luật liên quan…",
  "Đang đối chiếu căn cứ pháp lý…",
  "Đang soạn câu trả lời có trích dẫn…",
];

function Progress() {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setStep((value) => Math.min(value + 1, PROGRESS_STEPS.length - 1)), 2600);
    return () => window.clearInterval(timer);
  }, []);
  return (
    <div className="progress" role="status" aria-live="polite">
      <span className="dots" aria-hidden="true"><i /><i /><i /></span>
      <span>{PROGRESS_STEPS[step]}</span>
    </div>
  );
}

export default function ChatTurn({ turn, isActive, onCitationSelect, onOpenEvidence, onAsk, onRetry }: ChatTurnProps) {
  const [copied, setCopied] = useState(false);
  const response = turn.response;

  async function copyAnswer() {
    if (!response?.answer) return;
    try {
      await navigator.clipboard.writeText(response.answer);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable (e.g. insecure context): ignore */
    }
  }

  return (
    <article className="turn">
      <div className="bubble-user"><p>{turn.query}</p></div>

      <div className={`bubble-assistant ${isActive ? "is-active" : ""}`}>
        <div className="assistant-avatar" aria-hidden="true"><ScaleIcon size={16} /></div>
        <div className="assistant-body">
          {turn.state === "loading" && <Progress />}

          {turn.state === "error" && (
            <div className="notice tone-danger">
              <AlertIcon />
              <div>
                <strong>Không kết nối được máy chủ</strong>
                <p>{turn.error ?? "Yêu cầu thất bại."} Kiểm tra backend đang chạy rồi thử lại.</p>
                <button type="button" className="ghost-button" onClick={() => onRetry(turn)}>
                  <RetryIcon /> Thử lại
                </button>
              </div>
            </div>
          )}

          {turn.state === "done" && response && (
            <AssistantAnswer
              response={response}
              onCitationSelect={(evidenceId) => onCitationSelect(turn.id, evidenceId)}
              onOpenEvidence={() => onOpenEvidence(turn.id)}
              onAsk={onAsk}
              onCopy={copyAnswer}
              copied={copied}
            />
          )}
        </div>
      </div>
    </article>
  );
}

function AssistantAnswer({
  response, onCitationSelect, onOpenEvidence, onAsk, onCopy, copied,
}: {
  response: ChatResponse;
  onCitationSelect: (evidenceId: string) => void;
  onOpenEvidence: () => void;
  onAsk: (query: string) => void;
  onCopy: () => void;
  copied: boolean;
}) {
  const meta = statusMeta(response);
  const status = statusOf(response);
  const isAnswer = status === "generated" || status === "generated_with_warnings";
  const cited = citedEvidenceIds(response);
  const citedEvidence = response.evidence.filter((item) => cited.has(item.evidence_id.toUpperCase()));
  const suggestions = response.suggestions ?? [];
  const latency = formatSeconds(response.debug?.total_latency_ms);

  return (
    <>
      {meta.title && (
        <div className={`notice tone-${meta.tone}`}>
          <AlertIcon />
          <div>
            <strong>{meta.title}</strong>
            {meta.hint && <p>{meta.hint}</p>}
          </div>
        </div>
      )}

      {response.answer && (isAnswer || !meta.hint) ? (
        <AnswerMarkdown answer={response.answer} evidence={response.evidence} onCitationSelect={onCitationSelect} />
      ) : null}

      {suggestions.length > 0 && (
        <div className="suggestion-list" aria-label="Chủ đề gợi ý">
          {suggestions.map((item) => (
            <button type="button" key={`${item.label}-${item.article_number}`} onClick={() => onAsk(suggestionQuery(item.label, item.article_number))}>
              {item.label}
            </button>
          ))}
        </div>
      )}

      {citedEvidence.length > 0 && (
        <div className="sources">
          <span className="sources-label">Căn cứ</span>
          {citedEvidence.map((item) => (
            <button type="button" key={item.evidence_id} className="source-chip" onClick={() => onCitationSelect(item.evidence_id.toUpperCase())}>
              {legalReference(item)}
            </button>
          ))}
        </div>
      )}

      <div className="answer-actions">
        {response.evidence.length > 0 && (
          <button type="button" className="ghost-button" onClick={onOpenEvidence}>
            <BookIcon /> Xem căn cứ ({response.evidence.length})
          </button>
        )}
        {isAnswer && response.answer && (
          <button type="button" className="ghost-button" onClick={onCopy}>
            {copied ? <><CheckIcon /> Đã sao chép</> : <><CopyIcon /> Sao chép</>}
          </button>
        )}
        {latency && <span className="latency">{latency}</span>}
      </div>
    </>
  );
}
