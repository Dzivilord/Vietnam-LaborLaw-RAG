export interface ChatResponse {
  answer: string | null;
  citations: string[];
  evidence: EvidenceCitation[];
  debug: {
    status?: string;
    diagnostics?: Record<string, unknown>;
  };
}

export interface EvidenceCitation {
  evidence_id: string;
  chunk_id: string;
  content: string;
  document_title?: string | null;
  document_number?: string | null;
  source_file?: string | null;
  page?: string | null;
  chapter_number?: string | null;
  chapter_title?: string | null;
  section_number?: string | null;
  section_title?: string | null;
  article_number?: string | null;
  article_title?: string | null;
  clause_number?: string | null;
  point_number?: string | null;
}

export async function chat(query: string) {
  const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    throw new Error(`Chat request failed (${response.status})`);
  }

  return response.json() as Promise<ChatResponse>;
}
