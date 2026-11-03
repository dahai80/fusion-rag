// fusion-rag REST API client. Same-origin fetch — standalone dev goes through
// the vite proxy, embedded through the gateway /shell/fusion-rag/ proxy.
//
// Issue #109 rule: API calls ride the SAME prefix the page is mounted under.
// A fetch("/kb/...") from inside the shell iframe resolves against the
// GATEWAY origin (:11432) and 404s — derive the base from location.pathname:
// /shell/fusion-rag/... → that prefix; standalone ("/" or /index.html) → "".
export function apiBase(
    pathname = globalThis.location?.pathname ?? "/",
): string {
    let p = pathname.replace(/index\.html$/, "");
    if (!p.endsWith("/")) p += "/";
    return p;
}

const API_BASE = apiBase();

export class ApiError extends Error {
    status: number;
    constructor(status: number, message: string) {
        super(message);
        this.status = status;
    }
}

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
    const resp = await fetch(API_BASE + path.replace(/^\//, ""), {
        ...init,
        headers: { "Content-Type": "application/json", ...(init.headers || {}) },
    });
    if (!resp.ok) {
        const body = (await resp.json().catch(() => null)) as
            | { detail?: string }
            | null;
        throw new ApiError(
            resp.status,
            body?.detail || `HTTP ${resp.status}`,
        );
    }
    return (await resp.json()) as T;
}

export interface KbSummary {
    id: string;
    name: string;
    embedding_model: string;
    chunk_strategy: string;
    file_count?: number;
}

export interface KbDoc {
    doc_id?: string;
    id?: string;
    doc_name?: string;
    name?: string;
    chunk_count?: number;
    chunks?: number;
    status?: string;
}

export interface SearchRow {
    chunk_id?: string;
    doc_id?: string;
    doc_name?: string;
    score?: number;
    text?: string;
    snippet?: string;
    page?: number;
}

export interface HealthInfo {
    service?: string;
    version?: string;
    llm_auth?: string;
    status?: string;
}

export const fetchHealth = () => api<HealthInfo>("/health");
export const fetchKbs = () => api<KbSummary[]>("/kb/bases");
export const createKb = (name: string) =>
    api<KbSummary>("/kb/bases", {
        method: "POST",
        body: JSON.stringify({ name }),
    });
export const deleteKb = (id: string) =>
    api<void>(`/kb/bases/${id}`, { method: "DELETE" });
export const fetchDocs = (kbId: string) =>
    api<KbDoc[] | { documents: KbDoc[] }>(`/kb/bases/${kbId}/documents`);
export const ingestDoc = (kbId: string, docName: string, content: string) =>
    api<{ chunks?: number }>(`/kb/bases/${kbId}/documents/ingest`, {
        method: "POST",
        body: JSON.stringify({ content_type: "text", content, doc_name: docName }),
    });
export const deleteDoc = (kbId: string, docId: string) =>
    api<void>(`/kb/bases/${kbId}/documents/${docId}`, { method: "DELETE" });
export const searchKb = (kbId: string, query: string, topK = 5) =>
    api<SearchRow[] | { results: SearchRow[] }>(`/kb/bases/${kbId}/search`, {
        method: "POST",
        body: JSON.stringify({ query, top_k: topK }),
    });

export function normalizeDocs(r: KbDoc[] | { documents: KbDoc[] }): KbDoc[] {
    return Array.isArray(r) ? r : r.documents || [];
}

export function normalizeRows(r: SearchRow[] | { results: SearchRow[] }): SearchRow[] {
    return Array.isArray(r) ? r : r.results || [];
}
