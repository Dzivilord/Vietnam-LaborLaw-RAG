"use client";

import ChatPanel from "@/components/ChatPanel";
import DocumentViewer from "@/components/DocumentViewer";
import type { ChatResponse } from "@/services/api";
import { useState } from "react";

export default function Page() {
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(
    null,
  );
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [conversationStarted, setConversationStarted] = useState(false);

  function handleCitationSelect(evidenceId: string) {
    setSelectedEvidenceId(evidenceId);
    setEvidenceOpen(true);
    window.setTimeout(() => {
      document.getElementById(`evidence-${evidenceId}`)?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }, 0);
  }

  return (
    <main className={`app-shell ${conversationStarted ? "has-conversation" : "is-welcome"} ${evidenceOpen ? "evidence-is-open" : ""}`}>
      <div className="page-intro">
        <span className="accent-line" />
        <p>Tra cứu pháp luật lao động / Bộ luật Lao động 2019</p>
      </div>
      {response && evidenceOpen && <button className="evidence-toggle" type="button" onClick={() => setEvidenceOpen(false)}>‹ Thu gọn căn cứ</button>}
      <div className="workspace-grid">
        <ChatPanel
          onConversationStart={() => setConversationStarted(true)}
          onResponse={(nextResponse) => {
            setResponse(nextResponse);
            setConversationStarted(true);
            setSelectedEvidenceId(null);
            setEvidenceOpen(false);
          }}
          onCitationSelect={handleCitationSelect}
        />
        <DocumentViewer
          response={response}
          selectedEvidenceId={selectedEvidenceId}
          isOpen={evidenceOpen}
        />
      </div>
    </main>
  );
}
