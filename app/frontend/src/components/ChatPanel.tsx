"use client";

import { FormEvent, useState } from "react";
import { chat } from "@/services/api";
import type { ChatResponse } from "@/services/api";
import AnswerMarkdown from "@/components/AnswerMarkdown";

interface ChatPanelProps {
  onResponse: (response: ChatResponse | null) => void;
  onCitationSelect: (evidenceId: string) => void;
}

export default function ChatPanel({
  onResponse,
  onCitationSelect,
}: ChatPanelProps) {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuery = query.trim();
    if (!trimmedQuery || loading) return;

    setLoading(true);
    setError(null);
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
          <p className="eyebrow">Labor Law RAG</p>
          <h1>Ask the legal archive</h1>
        </div>
        <span className="status-dot" aria-label="API ready" />
      </div>
      <p className="muted">
        Search the indexed Vietnamese labor-law corpus and inspect the evidence
        behind each answer.
      </p>

      <form onSubmit={handleSubmit} className="chat-form">
        <label htmlFor="query">Your question</label>
        <textarea
          id="query"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Ví dụ: Thời gian thử việc tối đa là bao lâu?"
          rows={5}
        />
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? "Searching..." : "Ask the archive"}
        </button>
      </form>

      {error && <p className="error-message">{error}</p>}
      {response && (
        <div className="answer-block">
          <div className="answer-label">Answer</div>
          {response.answer ? (
            <AnswerMarkdown
              answer={response.answer}
              evidence={response.evidence}
              onCitationSelect={onCitationSelect}
            />
          ) : (
            <p>No answer was generated.</p>
          )}
        </div>
      )}
    </section>
  );
}
