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

// ── Issue #113: previously API-only surfaces ──

export interface GraphStats {
    kb_id: string;
    entities: number;
    relations: number;
}

export interface GraphHit {
    chunk_id?: string;
    doc_id?: string;
    text?: string;
    [k: string]: unknown;
}

export const graphStats = (kbId: string) =>
    api<GraphStats>(`/kb/bases/${kbId}/graph/stats`);
export const graphBuild = (kbId: string, maxChunks = 30) =>
    api<{ status: string; entities: number; relations: number; chunks_processed: number }>(
        `/kb/bases/${kbId}/graph/build`,
        { method: "POST", body: JSON.stringify({ max_chunks: maxChunks }) },
    );
export const graphSearch = (kbId: string, query: string, maxHops = 2) =>
    api<GraphHit[]>(`/kb/bases/${kbId}/graph/search`, {
        method: "POST",
        body: JSON.stringify({ query, max_hops: maxHops }),
    });

export interface Snapshot {
    version_id: string;
    created_at?: string | number;
    description?: string;
    [k: string]: unknown;
}

export const listVersions = (kbId: string) =>
    api<Snapshot[]>(`/kb/bases/${kbId}/versions`);
export const createVersion = (kbId: string, description: string) =>
    api<Snapshot>(`/kb/bases/${kbId}/versions`, {
        method: "POST",
        body: JSON.stringify({ description }),
    });
export const rollbackVersion = (kbId: string, versionId: string) =>
    api<{ success?: boolean }>(`/kb/bases/${kbId}/versions/${versionId}/rollback`, {
        method: "POST",
    });
export const deleteVersion = (kbId: string, versionId: string) =>
    api<void>(`/kb/bases/${kbId}/versions/${versionId}`, { method: "DELETE" });

export interface SearchTemplateRow {
    name: string;
    description?: string;
    alpha?: number;
    rerank?: boolean;
    top_k?: number;
    threshold?: number;
    rewrite_mode?: string;
    [k: string]: unknown;
}

export const listTemplates = (kbId: string) =>
    api<SearchTemplateRow[]>(`/kb/bases/${kbId}/templates`);
export const deleteTemplate = (kbId: string, name: string) =>
    api<void>(`/kb/bases/${kbId}/templates/${name}`, { method: "DELETE" });
export const createTemplate = (kbId: string, tpl: Partial<SearchTemplateRow>) =>
    api<SearchTemplateRow>(`/kb/bases/${kbId}/templates`, {
        method: "POST",
        body: JSON.stringify(tpl),
    });

export interface AclRule {
    id?: string;
    subject: string;
    resource_type?: string;
    resource_path: string;
    actions: string[];
    [k: string]: unknown;
}

export const listPermissions = (kbId: string) =>
    api<AclRule[]>(`/kb/bases/${kbId}/permissions`);
export const addPermission = (kbId: string, rule: Partial<AclRule>) =>
    api<AclRule>(`/kb/bases/${kbId}/permissions`, {
        method: "POST",
        body: JSON.stringify(rule),
    });
export const deletePermission = (kbId: string, ruleId: string) =>
    api<void>(`/kb/bases/${kbId}/permissions/${ruleId}`, { method: "DELETE" });
export const checkPermission = (kbId: string, subject: string, action: string) =>
    api<{ allowed: boolean }>(`/kb/bases/${kbId}/permissions/check`, {
        method: "POST",
        body: JSON.stringify({ subject, action, resource_path: "/" }),
    });

export interface AuditRow {
    id?: number;
    caller?: string;
    action?: string;
    created_at?: string | number;
    [k: string]: unknown;
}

export const listAudit = (kbId: string, limit = 50) =>
    api<AuditRow[]>(`/kb/bases/${kbId}/audit?limit=${limit}`);
export const auditExportUrl = (kbId: string, fmt: "json" | "csv") =>
    `${API_BASE}kb/bases/${kbId}/audit/export?format=${fmt}`;

export interface ApiKeyRow {
    name?: string;
    key_hash_prefix?: string;
    created_at?: string | number;
    [k: string]: unknown;
}

export const listApiKeys = () => api<{ keys: ApiKeyRow[] }>("/kb/auth/keys");
export const createApiKey = (name: string) =>
    api<{ key: string; name: string }>("/kb/auth/keys", {
        method: "POST",
        body: JSON.stringify({ name }),
    });
export const deleteApiKey = (hashPrefix: string) =>
    api<void>(`/kb/auth/keys/${hashPrefix}`, { method: "DELETE" });
