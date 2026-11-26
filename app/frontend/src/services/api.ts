export interface ClarificationSuggestion {
  label: string;
  article_number?: string | null;
  clause_number?: string | null;
}

export interface ChatResponse {
  request_id: string;
  answer: string | null;
  /** Pipeline outcome: generated, insufficient_context, out_of_scope, refused, clarification_required, ... */
  status?: string | null;
  citations: string[];
  evidence: EvidenceCitation[];
  /** Grounded topic options returned when the question is too broad. */
  suggestions?: ClarificationSuggestion[];
  debug: {
    status?: string;
    diagnostics?: Record<string, unknown>;
    total_latency_ms?: number;
  };
}

export interface RequestTrace {
  request_id: string;
  created_at: string;
  query: string | null;
  query_hash: string;
  status: string;
  total_latency_ms: number;
  timings_ms: Record<string, number>;
  retrieved_count: number | null;
  reranked_count: number | null;
  evidence_count: number;
  citations: string[];
  provider: string | null;
  model: string | null;
  error: string | null;
}

export interface ChatOptions {
  enable_rerank?: boolean;
  generation?: {
    model?: string;
    fallback_models?: string[];
    temperature?: number;
    max_output_tokens?: number;
  };
}

export interface LiveTraceEvent {
  timestamp: string;
  stage: string;
  status: string;
  message: string;
  response?: ChatResponse;
  error?: string;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_RAG_API_URL ?? "http://localhost:8000";

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

export async function chat(query: string, options?: ChatOptions) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, options }),
  });

  if (!response.ok) {
    throw new Error(`Chat request failed (${response.status})`);
  }

  return response.json() as Promise<ChatResponse>;
}

export async function getRecentTraces(limit = 50) {
  const response = await fetch(`${API_BASE_URL}/admin/traces?limit=${limit}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`Trace request failed (${response.status})`);
  }
  return response.json() as Promise<{ traces: RequestTrace[] }>;
}

export async function startLiveRun(query: string, options?: ChatOptions) {
  const response = await fetch(`${API_BASE_URL}/admin/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, options }),
  });
  if (!response.ok) throw new Error(`Live run failed to start (${response.status})`);
  return response.json() as Promise<{ request_id: string; events_url: string }>;
}

export function liveEventsUrl(path: string) { return `${API_BASE_URL}${path}`; }
