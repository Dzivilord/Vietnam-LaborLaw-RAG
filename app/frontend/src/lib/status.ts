import type { ChatResponse, EvidenceCitation } from "@/services/api";

export type Tone = "neutral" | "info" | "warning" | "danger";

export interface StatusMeta {
  tone: Tone;
  /** Short title shown above the assistant message; null for a normal answer. */
  title: string | null;
  hint?: string;
}

const STATUS: Record<string, StatusMeta> = {
  generated: { tone: "neutral", title: null },
  generated_with_warnings: {
    tone: "warning",
    title: "Câu trả lời cần kiểm tra lại",
    hint: "Một số trích dẫn chưa khớp hoàn toàn. Hãy đối chiếu với căn cứ pháp lý bên cạnh.",
  },
  insufficient_context: {
    tone: "warning",
    title: "Chưa tìm thấy đủ căn cứ",
    hint: "Không có quy định đủ liên quan trong Bộ luật Lao động 2019. Thử diễn đạt cụ thể hơn hoặc nêu số Điều nếu bạn biết.",
  },
  abstained: {
    tone: "warning",
    title: "Không đưa ra câu trả lời",
    hint: "Hệ thống không chắc chắn về căn cứ nên không trả lời để tránh thông tin sai.",
  },
  out_of_scope: {
    tone: "info",
    title: "Ngoài phạm vi hỗ trợ",
    hint: "Trợ lý chỉ trả lời các câu hỏi về Bộ luật Lao động 2019.",
  },
  refused: {
    tone: "danger",
    title: "Không thể hỗ trợ yêu cầu này",
  },
  clarification_required: {
    tone: "info",
    title: "Câu hỏi còn khá chung",
    hint: "Chọn một chủ đề bên dưới hoặc mô tả cụ thể hơn tình huống của bạn.",
  },
  ready: { tone: "info", title: "Đã tìm thấy căn cứ", hint: "Xem các điều luật liên quan ở khung căn cứ." },
  error: {
    tone: "danger",
    title: "Đã xảy ra lỗi khi xử lý",
    hint: "Vui lòng thử lại sau ít phút.",
  },
};

export function statusOf(response: ChatResponse): string {
  return response.status || (response.debug?.status as string | undefined) || "generated";
}

export function statusMeta(response: ChatResponse): StatusMeta {
  return STATUS[statusOf(response)] ?? { tone: "neutral", title: null };
}

/** Evidence IDs the model actually cited (falls back to every evidence item). */
export function citedEvidenceIds(response: ChatResponse): Set<string> {
  const generation = (response.debug?.diagnostics as { generation?: { citations_used?: string[] } } | undefined)?.generation;
  const used = generation?.citations_used;
  if (used && used.length) return new Set(used.map((id) => id.toUpperCase()));
  const inText = new Set<string>();
  for (const match of (response.answer ?? "").matchAll(/\[(E\d+)\]/gi)) inText.add(match[1].toUpperCase());
  return inText;
}

export function legalReference(item: EvidenceCitation): string {
  if (!item.article_number) return item.document_title || "Nguồn trích dẫn";
  let ref = `Điều ${item.article_number}`;
  if (item.clause_number) ref = `Khoản ${item.clause_number} ${ref}`;
  if (item.point_number) ref = `Điểm ${item.point_number} ${ref}`;
  return ref;
}

export function formatSeconds(ms: number | undefined): string | null {
  if (ms === undefined || ms === null || Number.isNaN(ms)) return null;
  return `${(ms / 1000).toLocaleString("vi-VN", { maximumFractionDigits: 1 })} giây`;
}

/** Turn a suggestion such as "Trả lương (Điều 95)" into a focused follow-up question. */
export function suggestionQuery(label: string, articleNumber?: string | null): string {
  const title = label.replace(/\s*\((?:Điều|Khoản)[^)]*\)\s*$/u, "").trim();
  if (articleNumber) return `Điều ${articleNumber} quy định thế nào về ${title.charAt(0).toLowerCase()}${title.slice(1)}?`;
  return `Quy định về ${title.charAt(0).toLowerCase()}${title.slice(1)} như thế nào?`;
}
