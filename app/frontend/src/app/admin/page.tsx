"use client";

import "./admin.css";

import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getRecentTraces, liveEventsUrl, startLiveRun, type ChatOptions, type ChatResponse, type LiveTraceEvent, type RequestTrace } from "@/services/api";

const STAGE_LABELS: Record<string, string> = {
  retrieval: "Truy xuất", reranking: "Rerank", context_building: "Tạo context",
  ambiguity_check: "Kiểm tra mơ hồ", generation: "Tạo sinh",
};

function milliseconds(value: number | undefined) { return value === undefined ? "—" : `${Math.round(value).toLocaleString("vi-VN")} ms`; }
function timestamp(value: string) { return new Intl.DateTimeFormat("vi-VN", { dateStyle: "short", timeStyle: "medium" }).format(new Date(value)); }
function statusTone(status: string) { if (status === "generated" || status === "ready") return "is-success"; if (status.includes("warning") || status.includes("clarification")) return "is-warning"; return "is-danger"; }

export default function AdminPage() {
  const [query, setQuery] = useState("Thời gian thử việc tối đa là bao lâu?");
  const [rerankEnabled, setRerankEnabled] = useState(true);
  const [model, setModel] = useState("qwen3.5:4b");
  const [temperature, setTemperature] = useState(0.2);
  const [maxTokens, setMaxTokens] = useState(1200);
  const [liveEvents, setLiveEvents] = useState<LiveTraceEvent[]>([]);
  const [activeRequestId, setActiveRequestId] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [traces, setTraces] = useState<RequestTrace[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const refreshTraces = useCallback(async () => {
    try { setRefreshing(true); const next = await getRecentTraces(); setTraces(next.traces); setError(null); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Không thể tải traces."); }
    finally { setRefreshing(false); }
  }, []);
  useEffect(() => { void refreshTraces(); const timer = window.setInterval(() => void refreshTraces(), 5000); return () => window.clearInterval(timer); }, [refreshTraces]);
  async function runQuery(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); if (!query.trim() || loading) return; setLoading(true); setError(null);
    const options: ChatOptions = {
      enable_rerank: rerankEnabled,
      generation: { model: model.trim() || undefined, temperature, max_output_tokens: maxTokens },
    };
    try {
      eventSourceRef.current?.close();
      setResponse(null); setLiveEvents([]);
      const run = await startLiveRun(query.trim(), options);
      setActiveRequestId(run.request_id);
      const source = new EventSource(liveEventsUrl(run.events_url));
      eventSourceRef.current = source;
      source.addEventListener("trace", (event) => {
        const traceEvent = JSON.parse(event.data) as LiveTraceEvent;
        setLiveEvents((current) => [...current, traceEvent]);
        if (traceEvent.response) setResponse(traceEvent.response);
        if (traceEvent.error) setError(traceEvent.error);
      });
      source.addEventListener("end", () => { source.close(); setLoading(false); void refreshTraces(); });
      source.onerror = () => { source.close(); setLoading(false); setError("Live trace connection was interrupted."); };
      return;
    }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Yêu cầu thất bại."); setLoading(false); }
    finally { /* Live SSE owns the loading lifecycle after a successful start. */ }
  }
  const diagnostics = response?.debug.diagnostics ?? {};
  const pipeline = (diagnostics.pipeline ?? {}) as Record<string, unknown>;
  const generation = (diagnostics.generation ?? {}) as Record<string, unknown>;
  const timings = (pipeline.timings_ms ?? {}) as Record<string, number>;
  const stages = useMemo(() => [...Object.entries(timings), ...(typeof generation.latency_ms === "number" ? [["generation", generation.latency_ms] as [string, number]] : [])], [generation.latency_ms, timings]);

  return <main className="admin-shell">
    <header className="admin-header"><div><p className="admin-kicker">Labor Law RAG / internal</p><h1>Admin observability console</h1><p>Theo dõi pipeline từ truy xuất căn cứ đến câu trả lời được tạo sinh.</p></div><div className="admin-live"><span /> Auto refresh 5 giây</div></header>
    <section className="admin-query-card"><div><p className="admin-kicker">Trace một request mới</p><h2>Chạy pipeline có kiểm soát</h2></div><form onSubmit={runQuery}><textarea aria-label="Câu hỏi cần trace" value={query} onChange={(event) => setQuery(event.target.value)} rows={3} /><button type="submit" disabled={loading || !query.trim()}>{loading ? "Đang chạy pipeline…" : "Run trace"}</button></form>{error && <p className="admin-error">{error}</p>}</section>
    <section className="admin-card admin-config-card"><div className="admin-card-heading"><div><p className="admin-kicker">Per-request configuration</p><h2>Local Ollama controls</h2></div><span className="admin-status is-success">No model API</span></div><div className="admin-config-grid"><label className="admin-toggle"><input type="checkbox" checked={rerankEnabled} onChange={(event) => setRerankEnabled(event.target.checked)} /><span>Enable reranking</span><small>Tắt để so sánh retrieval gốc với rerank.</small></label><label>Ollama model<input value={model} onChange={(event) => setModel(event.target.value)} placeholder="qwen3.5:4b" /><small>Chạy local; không gửi prompt hay API key ra ngoài.</small></label><label>Temperature <output>{temperature}</output><input type="range" min="0" max="2" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label><label>Max output tokens<input type="number" min="64" max="4096" value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))} /></label></div></section>
    <section className="admin-kpis" aria-label="Tổng quan request gần đây"><article><span>Recent traces</span><strong>{traces.length}</strong></article><article><span>Successful</span><strong>{traces.filter((item) => item.status.startsWith("generated")).length}</strong></article><article><span>Latest latency</span><strong>{traces[0] ? milliseconds(traces[0].total_latency_ms) : "—"}</strong></article><article><span>Trace store</span><strong>{refreshing ? "Syncing" : "Live"}</strong></article></section>
    <section className="admin-card admin-live-timeline"><div className="admin-card-heading"><div><p className="admin-kicker">Live request stream</p><h2>{loading ? "Pipeline đang chạy" : "Pipeline activity"}</h2></div>{activeRequestId && <code>{activeRequestId}</code>}</div>{!liveEvents.length ? <p className="admin-empty">Sự kiện sẽ xuất hiện ngay khi request bắt đầu chạy.</p> : <ol className="live-event-list">{liveEvents.map((event, index) => <li key={`${event.timestamp}-${index}`} className={`is-${event.status}`}><span className="stage-dot" /><time>{timestamp(event.timestamp)}</time><strong>{STAGE_LABELS[event.stage] ?? event.stage}</strong><p>{event.message}</p><small>{event.status === "started" ? "Đang chạy" : event.status === "completed" ? "Hoàn tất" : event.status}</small></li>)}</ol>}</section>
    <div className="admin-grid"><section className="admin-card admin-run-detail"><div className="admin-card-heading"><div><p className="admin-kicker">Current run</p><h2>Pipeline timeline</h2></div>{response && <span className={`admin-status ${statusTone(response.debug.status ?? "error")}`}>{response.debug.status ?? "unknown"}</span>}</div>{!response ? <p className="admin-empty">Chạy một câu hỏi để xem thời gian và output của từng bước.</p> : <><div className="admin-request-meta"><span>Request <code>{response.request_id}</code></span><span>Total {milliseconds(response.debug.total_latency_ms)}</span></div><ol className="stage-list">{stages.map(([name, duration]) => <li key={name}><span className="stage-dot" /><span>{STAGE_LABELS[name] ?? name}</span><strong>{milliseconds(duration)}</strong></li>)}</ol><dl className="admin-facts"><div><dt>Retrieved</dt><dd>{String(pipeline.retrieved_count ?? "—")}</dd></div><div><dt>Reranked</dt><dd>{String(pipeline.reranked_count ?? "—")}</dd></div><div><dt>Evidence</dt><dd>{response.evidence.length}</dd></div><div><dt>Citations</dt><dd>{response.citations.join(", ") || "—"}</dd></div><div><dt>Model</dt><dd>{String(generation.model ?? "—")}</dd></div></dl></>}</section>
    <section className="admin-card admin-answer-detail"><div className="admin-card-heading"><div><p className="admin-kicker">Generated output</p><h2>Answer & evidence</h2></div></div>{!response ? <p className="admin-empty">Câu trả lời và căn cứ sẽ xuất hiện tại đây.</p> : <><p className="admin-answer">{response.answer ?? "Không có câu trả lời được tạo."}</p><div className="admin-evidence-list">{response.evidence.map((item) => <article key={item.evidence_id}><b>[{item.evidence_id}]</b><span>{item.article_number ? `Điều ${item.article_number}` : item.chunk_id}</span><p>{item.content}</p></article>)}</div></>}</section></div>
    <section className="admin-card admin-history"><div className="admin-card-heading"><div><p className="admin-kicker">Runtime events</p><h2>Recent request traces</h2></div><button className="admin-secondary-button" type="button" onClick={() => void refreshTraces()}>Refresh</button></div><div className="trace-table-wrap"><table><thead><tr><th>Time</th><th>Request</th><th>Status</th><th>Pipeline</th><th>Evidence</th><th>Model</th></tr></thead><tbody>{traces.map((trace) => <tr key={trace.request_id}><td>{timestamp(trace.created_at)}</td><td><code title={trace.request_id}>{trace.request_id.slice(0, 8)}</code><small>{trace.query ?? `hash:${trace.query_hash}`}</small></td><td><span className={`admin-status ${statusTone(trace.status)}`}>{trace.status}</span></td><td>{milliseconds(trace.total_latency_ms)}<small>{Object.entries(trace.timings_ms).map(([name, value]) => `${STAGE_LABELS[name] ?? name}: ${Math.round(value)}ms`).join(" · ")}</small></td><td>{trace.evidence_count} / {trace.citations.length}</td><td>{trace.model ?? "—"}</td></tr>)}{!traces.length && <tr><td colSpan={6} className="admin-empty">Chưa có trace trong process hiện tại.</td></tr>}</tbody></table></div></section>
  </main>;
}
