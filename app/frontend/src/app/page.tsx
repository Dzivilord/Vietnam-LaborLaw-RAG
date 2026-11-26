"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ChatTurn, { type Turn } from "@/components/ChatTurn";
import Composer from "@/components/Composer";
import EvidencePanel from "@/components/EvidencePanel";
import { BookIcon, PlusIcon, ScaleIcon } from "@/components/Icons";
import Welcome from "@/components/Welcome";
import { chat } from "@/services/api";

let nextId = 0;
const newTurnId = () => `turn-${Date.now()}-${nextId++}`;

export default function Page() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [panel, setPanel] = useState<{ turnId: string; evidenceId: string | null } | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const loading = turns.some((turn) => turn.state === "loading");
  const panelTurn = panel ? turns.find((turn) => turn.id === panel.turnId) ?? null : null;
  const lastWithEvidence = [...turns].reverse().find((turn) => turn.response?.evidence.length);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  const run = useCallback(async (turnId: string, query: string) => {
    try {
      const response = await chat(query);
      setTurns((current) => current.map((turn) => (turn.id === turnId ? { ...turn, state: "done", response } : turn)));
    } catch (error) {
      const message = error instanceof Error ? error.message : "Yêu cầu thất bại.";
      setTurns((current) => current.map((turn) => (turn.id === turnId ? { ...turn, state: "error", error: message } : turn)));
    }
  }, []);

  const ask = useCallback((query: string) => {
    if (loading) return;
    const id = newTurnId();
    setTurns((current) => [...current, { id, query, state: "loading" }]);
    void run(id, query);
  }, [loading, run]);

  function retry(turn: Turn) {
    if (loading) return;
    setTurns((current) => current.map((item) => (item.id === turn.id ? { ...item, state: "loading", error: undefined } : item)));
    void run(turn.id, turn.query);
  }

  function reset() {
    setTurns([]);
    setPanel(null);
  }

  const closePanel = useCallback(() => setPanel(null), []);
  const hasTurns = turns.length > 0;

  return (
    <div className={`app ${panel ? "panel-open" : ""}`}>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true"><ScaleIcon size={18} /></span>
          <div>
            <strong>Trợ lý Luật Lao động</strong>
            <span>Bộ luật Lao động 2019 · 45/2019/QH14</span>
          </div>
        </div>
        <div className="topbar-actions">
          {lastWithEvidence && !panel && (
            <button type="button" className="ghost-button" onClick={() => setPanel({ turnId: lastWithEvidence.id, evidenceId: null })}>
              <BookIcon /> Căn cứ
            </button>
          )}
          {hasTurns && (
            <button type="button" className="ghost-button" onClick={reset} disabled={loading} aria-label="Cuộc trò chuyện mới" title="Cuộc trò chuyện mới">
              <PlusIcon /> <span className="label-wide">Cuộc trò chuyện mới</span>
            </button>
          )}
        </div>
      </header>

      <div className="layout">
        <main className="chat">
          <div className="chat-scroll">
            <div className="chat-column">
              {!hasTurns ? (
                <Welcome onAsk={ask} />
              ) : (
                <>
                  <p className="session-note">
                    Mỗi câu hỏi được tra cứu độc lập; hãy nêu đủ tình huống trong từng câu.
                  </p>
                  {turns.map((turn) => (
                    <ChatTurn
                      key={turn.id}
                      turn={turn}
                      isActive={panel?.turnId === turn.id}
                      onCitationSelect={(turnId, evidenceId) => setPanel({ turnId, evidenceId })}
                      onOpenEvidence={(turnId) => setPanel({ turnId, evidenceId: null })}
                      onAsk={ask}
                      onRetry={retry}
                    />
                  ))}
                </>
              )}
              <div ref={bottomRef} />
            </div>
          </div>
          <div className="chat-column composer-dock">
            <Composer onSubmit={ask} disabled={loading} autoFocus />
          </div>
        </main>

        {panel && (
          <EvidencePanel
            query={panelTurn?.query ?? null}
            response={panelTurn?.response ?? null}
            selectedEvidenceId={panel.evidenceId}
            onClose={closePanel}
          />
        )}
      </div>
    </div>
  );
}
