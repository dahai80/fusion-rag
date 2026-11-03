import { ConfirmDialog, EmptyState, ErrorState, SkeletonLines } from "@fusion/web-ui";
import { useCallback, useEffect, useState } from "react";
import {
    type KbDoc,
    type KbSummary,
    deleteDoc,
    deleteKb,
    fetchDocs,
    ingestDoc,
    normalizeDocs,
    searchKb,
    type SearchRow,
    normalizeRows,
} from "../api";
import { sdk } from "../sdk";
import { t } from "../i18n";

// Right panel: selected KB detail — docs + search tester.
//
// Delete-KB fix (user report 2026-11-03): the old vanilla page used native
// confirm(), which the shell sandboxed iframe blocks silently — the button
// appeared dead. Spec S5: destructive ops use ConfirmDialog with the KB id
// typed back; toasts go through the shell SDK bridge.
export function KbDetail({
    kb,
    onDeleted,
}: {
    kb: KbSummary | null;
    onDeleted: () => void;
}) {
    const [confirmOpen, setConfirmOpen] = useState(false);

    if (!kb) {
        return (
            <div className="rounded-card border border-dashed border-border md:col-span-3">
                <EmptyState
                    title={t(
                        "选择左侧知识库，或新建一个",
                        "Pick a KB on the left, or create one",
                    )}
                />
            </div>
        );
    }

    const doDelete = async () => {
        try {
            await deleteKb(kb.id);
            sdk.toast.show({
                level: "info",
                message: t(`知识库 ${kb.name} 已删除`, `KB ${kb.name} deleted`),
            });
            onDeleted();
        } catch (e) {
            sdk.toast.show({
                level: "error",
                message: t(
                    `删除失败: ${e instanceof Error ? e.message : String(e)}`,
                    `Delete failed: ${e instanceof Error ? e.message : String(e)}`,
                ),
            });
        }
    };

    return (
        <main className="space-y-4 md:col-span-3">
            <div className="rounded-card border border-border bg-bg-elevated p-4">
                <div className="flex items-center justify-between gap-2">
                    <h2 className="min-w-0 truncate text-sm font-semibold text-text-primary">
                        {kb.name}{" "}
                        <span className="font-mono text-xs font-normal text-text-muted">
                            ({kb.id})
                        </span>
                    </h2>
                    <button
                        type="button"
                        onClick={() => setConfirmOpen(true)}
                        className="shrink-0 rounded-md border border-border px-3 py-1.5 text-xs text-error hover:bg-bg-muted"
                    >
                        {t("删除此库", "Delete KB")}
                    </button>
                </div>
                <DocsPanel kbId={kb.id} />
                <SearchPanel kbId={kb.id} />
            </div>
            <ConfirmDialog
                open={confirmOpen}
                danger
                title={t(`删除知识库 ${kb.name}？`, `Delete KB ${kb.name}?`)}
                description={t(
                    "向量、索引与文档元数据将一并删除，不可恢复。",
                    "Vectors, indexes and document metadata will be removed permanently.",
                )}
                confirmLabel={t("确认删除", "Delete")}
                cancelLabel={t("取消", "Cancel")}
                confirmName={kb.id}
                onConfirm={() => {
                    setConfirmOpen(false);
                    doDelete();
                }}
                onCancel={() => setConfirmOpen(false)}
            />
        </main>
    );
}

function DocsPanel({ kbId }: { kbId: string }) {
    const [docs, setDocs] = useState<KbDoc[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [docName, setDocName] = useState("");
    const [content, setContent] = useState("");
    const [busy, setBusy] = useState(false);

    const load = useCallback(async () => {
        try {
            setDocs(normalizeDocs(await fetchDocs(kbId)));
            setError(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }, [kbId]);

    useEffect(() => {
        setDocs(null);
        setError(null);
        load();
    }, [load]);

    const submit = async () => {
        const c = content.trim();
        if (!c || busy) return;
        setBusy(true);
        try {
            const r = await ingestDoc(
                kbId,
                docName.trim() || `inline-${Date.now()}.txt`,
                c,
            );
            setContent("");
            setDocName("");
            sdk.toast.show({
                level: "info",
                message: t(
                    `已入库 ${r.chunks ?? 0} 个分块`,
                    `Ingested ${r.chunks ?? 0} chunks`,
                ),
            });
            load();
        } catch (e) {
            sdk.toast.show({
                level: "error",
                message: t(
                    `入库失败: ${e instanceof Error ? e.message : String(e)}`,
                    `Ingest failed: ${e instanceof Error ? e.message : String(e)}`,
                ),
            });
        } finally {
            setBusy(false);
        }
    };

    const remove = async (docId: string) => {
        try {
            await deleteDoc(kbId, docId);
            sdk.toast.show({
                level: "info",
                message: t("文档已删除", "Document deleted"),
            });
            load();
        } catch (e) {
            sdk.toast.show({
                level: "error",
                message: t(
                    `删除失败: ${e instanceof Error ? e.message : String(e)}`,
                    `Delete failed: ${e instanceof Error ? e.message : String(e)}`,
                ),
            });
        }
    };

    return (
        <section className="mt-4 rounded-card border border-border">
            <div className="border-b border-border px-3 py-2 text-xs font-medium text-text-secondary">
                {t("文档", "Documents")}
            </div>
            <div className="flex gap-2 p-3">
                <input
                    value={docName}
                    onChange={(e) => setDocName(e.target.value)}
                    placeholder={t("文档名", "Doc name")}
                    className="w-32 rounded-md border border-border bg-bg px-2.5 py-1.5 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-brand"
                />
                <input
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && submit()}
                    placeholder={t("文本内容", "Text content")}
                    className="min-w-0 flex-1 rounded-md border border-border bg-bg px-2.5 py-1.5 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-brand"
                />
                <button
                    type="button"
                    disabled={!content.trim() || busy}
                    onClick={submit}
                    className="shrink-0 rounded-md bg-brand px-3 py-1.5 text-xs text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {t("入库", "Ingest")}
                </button>
            </div>
            {error && !docs ? (
                <ErrorState
                    message={error}
                    retryLabel={t("重试", "Retry")}
                    onRetry={load}
                />
            ) : !docs ? (
                <div className="p-3">
                    <SkeletonLines rows={3} />
                </div>
            ) : docs.length === 0 ? (
                <div className="px-3 py-6 text-center text-xs text-text-muted">
                    {t("暂无文档", "No documents")}
                </div>
            ) : (
                <ul className="max-h-52 divide-y divide-border overflow-y-auto text-xs">
                    {docs.map((d) => {
                        const id = d.doc_id || d.id || "";
                        return (
                            <li
                                key={id}
                                className="flex items-center justify-between gap-2 px-3 py-2"
                            >
                                <div className="min-w-0 truncate text-text-primary">
                                    <span className="font-medium">
                                        {d.doc_name || d.name || id}
                                    </span>
                                    <span className="ml-2 text-text-muted">
                                        {d.chunk_count ?? d.chunks ?? ""} chunks
                                        {d.status ? ` · ${d.status}` : ""}
                                    </span>
                                </div>
                                <button
                                    type="button"
                                    onClick={() => remove(id)}
                                    className="shrink-0 text-xs text-error hover:underline"
                                >
                                    {t("删除", "Delete")}
                                </button>
                            </li>
                        );
                    })}
                </ul>
            )}
        </section>
    );
}

function SearchPanel({ kbId }: { kbId: string }) {
    const [query, setQuery] = useState("");
    const [rows, setRows] = useState<SearchRow[] | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const run = async () => {
        const q = query.trim();
        if (!q || loading) return;
        setLoading(true);
        setError(null);
        try {
            setRows(normalizeRows(await searchKb(kbId, q)));
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
            setRows(null);
        } finally {
            setLoading(false);
        }
    };

    return (
        <section className="mt-4 rounded-card border border-border">
            <div className="border-b border-border px-3 py-2 text-xs font-medium text-text-secondary">
                {t("检索测试", "Search tester")}
            </div>
            <div className="flex gap-2 p-3">
                <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && run()}
                    placeholder={t("输入查询…", "Type a query…")}
                    className="min-w-0 flex-1 rounded-md border border-border bg-bg px-2.5 py-1.5 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-brand"
                />
                <button
                    type="button"
                    disabled={!query.trim() || loading}
                    onClick={run}
                    className="shrink-0 rounded-md bg-brand px-3 py-1.5 text-xs text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {t("搜索", "Search")}
                </button>
            </div>
            <div className="space-y-2 px-3 pb-3 text-xs">
                {loading ? (
                    <SkeletonLines rows={3} />
                ) : error ? (
                    <ErrorState message={error} onRetry={run} />
                ) : rows === null ? null : rows.length === 0 ? (
                    <div className="py-2 text-text-muted">
                        {t("无结果", "No results")}
                    </div>
                ) : (
                    rows.map((r, i) => (
                        <div
                            key={r.chunk_id || i}
                            className="rounded-card border border-border p-2.5"
                        >
                            <div className="mb-1 flex justify-between gap-2">
                                <span className="truncate text-text-secondary">
                                    {r.doc_name || r.doc_id || r.chunk_id}
                                    {r.page ? ` · p${r.page}` : ""}
                                </span>
                                <span className="shrink-0 font-mono text-brand">
                                    {(r.score ?? 0).toFixed(4)}
                                </span>
                            </div>
                            <p className="leading-relaxed text-text-primary">
                                {(r.text || r.snippet || "").slice(0, 400)}
                            </p>
                        </div>
                    ))
                )}
            </div>
        </section>
    );
}
