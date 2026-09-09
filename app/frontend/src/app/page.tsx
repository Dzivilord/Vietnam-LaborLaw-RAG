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

  function handleCitationSelect(evidenceId: string) {
    setSelectedEvidenceId(evidenceId);
    document.getElementById(`evidence-${evidenceId}`)?.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }

  return (
    <main className="app-shell">
      <div className="page-intro">
        <span className="accent-line" />
        <p>Research desk / 2019 labor code</p>
      </div>
      <div className="workspace-grid">
        <ChatPanel
          onResponse={(nextResponse) => {
            setResponse(nextResponse);
            setSelectedEvidenceId(null);
          }}
          onCitationSelect={handleCitationSelect}
        />
        <DocumentViewer
          response={response}
          selectedEvidenceId={selectedEvidenceId}
        />
      </div>
    </main>
  );
}
