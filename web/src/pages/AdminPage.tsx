import { useCallback, useEffect, useState } from "react";
import { EmptyState, ErrorState, SkeletonLines, StatusBadge } from "@fusion/web-ui";
import {
    type AclRule,
    type ApiKeyRow,
    type AuditRow,
    type KbSummary,
    addPermission,
    auditExportUrl,
    checkPermission,
    createApiKey,
    deleteApiKey,
    deletePermission,
    fetchKbs,
    listApiKeys,
    listAudit,
    listPermissions,
} from "../api";
import { t } from "../i18n";
import { sdk } from "../sdk";

// Issue #113: admin page — ACL rules, audit log export, API-key management.
// Scope note: permissions + keys are per-service (not per-KB) in the UI;
// the ACL rules endpoint is per-KB so the page takes a KB picker.

export function AdminPage() {
    const [kbs, setKbs] = useState<KbSummary[] | null>(null);
    const [kbId, setKbId] = useState<string>("");

    useEffect(() => {
        fetchKbs()
            .then((list) => {
                setKbs(list);
                setKbId((cur) => cur || list[0]?.id || "");
            })
            .catch(() => setKbs([]));
    }, []);

    return (
        <div className="mx-auto max-w-5xl space-y-4 px-4 py-4">
            <header className="flex items-center justify-between border-b border-border pb-4">
                <h1 className="text-lg font-semibold tracking-tight text-text-primary">
                    {t("管理", "Admin")}
                </h1>
                <a href="#/" className="text-xs text-brand hover:underline">
                    {t("返回控制台", "Back to console")}
                </a>
            </header>
            <section className="rounded-card border border-border bg-bg-elevated p-4">
                <div className="mb-3 flex items-center gap-2 text-sm text-text-primary">
                    <span>{t("知识库", "KB")}</span>
                    <select
                        value={kbId}
                        onChange={(e) => setKbId(e.target.value)}
                        className="rounded-md border border-border bg-bg px-2 py-1 text-xs text-text-primary outline-none focus:border-brand"
                    >
                        {(kbs ?? []).map((k) => (
                            <option key={k.id} value={k.id}>
                                {k.name} ({k.id})
                            </option>
                        ))}
                    </select>
                </div>
                {kbId ? <AclSection kbId={kbId} /> : <SkeletonLines rows={2} />}
            </section>
            {kbId ? <AuditSection kbId={kbId} /> : null}
            <KeysSection />
        </div>
    );
}

function AclSection({ kbId }: { kbId: string }) {
    const [rules, setRules] = useState<AclRule[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [subject, setSubject] = useState("");
    const [actions, setActions] = useState("read");
    const [checkSubject, setCheckSubject] = useState("");
    const [checkResult, setCheckResult] = useState<boolean | null>(null);

    const load = useCallback(async () => {
        try {
            setRules(await listPermissions(kbId));
            setError(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }, [kbId]);

    useEffect(() => {
        setRules(null);
        setCheckResult(null);
        load();
    }, [load]);

    const add = async () => {
        const s = subject.trim();
        if (!s) return;
        try {
            await addPermission(kbId, {
                subject: s,
                resource_type: "kb",
                resource_path: "/",
                actions: actions.split(",").map((a) => a.trim()).filter(Boolean),
            });
            setSubject("");
            load();
        } catch (e) {
            sdk.toast.show({
                level: "error",
                message: t(
                    `添加失败: ${e instanceof Error ? e.message : String(e)}`,
                    `Add failed: ${e instanceof Error ? e.message : String(e)}`,
                ),
            });
        }
    };

    const runCheck = async () => {
        const s = checkSubject.trim();
        if (!s) return;
        try {
            setCheckResult((await checkPermission(kbId, s, "read")).allowed);
        } catch (e) {
            sdk.toast.show({
                level: "error",
                message: e instanceof Error ? e.message : String(e),
            });
        }
    };

    return (
        <div className="space-y-3">
            <div className="text-xs font-medium text-text-secondary">
                {t("ACL 规则", "ACL rules")}
            </div>
            {error && !rules ? (
                <ErrorState message={error} retryLabel={t("重试", "Retry")} onRetry={load} />
            ) : !rules ? (
                <SkeletonLines rows={3} />
            ) : rules.length === 0 ? (
                <div className="py-2 text-xs text-text-muted">
                    {t("暂无规则", "No rules")}
                </div>
            ) : (
                <ul className="divide-y divide-border text-xs">
                    {rules.map((r, i) => (
                        <li key={r.id || i} className="flex items-center justify-between gap-2 py-2">
                            <div className="min-w-0">
                                <span className="font-mono text-text-primary">{r.subject}</span>
                                <span className="ml-2 text-text-muted">
                                    {r.resource_path} · {(r.actions || []).join(",")}
                                </span>
                            </div>
                            <button
                                type="button"
                                onClick={() => r.id && remove(r.id)}
                                className="shrink-0 text-xs text-error hover:underline"
                            >
                                {t("删除", "Delete")}
                            </button>
                        </li>
                    ))}
                </ul>
            )}
            <div className="flex flex-wrap gap-2">
                <input
                    value={subject}
                    onChange={(e) => setSubject(e.target.value)}
                    placeholder={t("主体（如 user:alice）", "Subject (e.g. user:alice)")}
                    className="min-w-0 flex-1 rounded-md border border-border bg-bg px-2.5 py-1.5 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-brand"
                />
                <input
                    value={actions}
                    onChange={(e) => setActions(e.target.value)}
                    placeholder="read,write"
                    className="w-28 shrink-0 rounded-md border border-border bg-bg px-2.5 py-1.5 font-mono text-xs text-text-primary outline-none focus:border-brand"
                />
                <button
                    type="button"
                    disabled={!subject.trim()}
                    onClick={add}
                    className="shrink-0 rounded-md bg-brand px-3 py-1.5 text-xs text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {t("添加", "Add")}
                </button>
            </div>
            <div className="flex items-center gap-2">
                <input
                    value={checkSubject}
                    onChange={(e) => setCheckSubject(e.target.value)}
                    placeholder={t("权限检查：主体", "Check: subject")}
                    className="min-w-0 flex-1 rounded-md border border-border bg-bg px-2.5 py-1.5 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-brand"
                />
                <button
                    type="button"
                    disabled={!checkSubject.trim()}
                    onClick={runCheck}
                    className="shrink-0 rounded-md border border-border px-3 py-1.5 text-xs text-text-primary hover:bg-bg-muted disabled:opacity-40"
                >
                    {t("检查 read", "Check read")}
                </button>
                {checkResult !== null ? (
                    <StatusBadge
                        status={checkResult ? "success" : "error"}
                        label={checkResult ? t("允许", "allowed") : t("拒绝", "denied")}
                    />
                ) : null}
            </div>
        </div>
    );

    async function remove(id: string) {
        try {
            await deletePermission(kbId, id);
            load();
        } catch (e) {
            sdk.toast.show({
                level: "error",
                message: e instanceof Error ? e.message : String(e),
            });
        }
    }
}

function AuditSection({ kbId }: { kbId: string }) {
    const [rows, setRows] = useState<AuditRow[] | null>(null);
    const [error, setError] = useState<string | null>(null);

    const load = useCallback(async () => {
        try {
            setRows(await listAudit(kbId));
            setError(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }, [kbId]);

    useEffect(() => {
        setRows(null);
        load();
    }, [load]);

    return (
        <section className="rounded-card border border-border bg-bg-elevated p-4">
            <div className="mb-3 flex items-center justify-between">
                <div className="text-xs font-medium text-text-secondary">
                    {t("审计日志", "Audit log")}
                </div>
                <div className="flex gap-2">
                    <a
                        href={auditExportUrl(kbId, "json")}
                        target="_blank"
                        rel="noreferrer"
                        className="rounded-md border border-border px-2.5 py-1 text-xs text-text-primary hover:bg-bg-muted"
                    >
                        JSON
                    </a>
                    <a
                        href={auditExportUrl(kbId, "csv")}
                        target="_blank"
                        rel="noreferrer"
                        className="rounded-md border border-border px-2.5 py-1 text-xs text-text-primary hover:bg-bg-muted"
                    >
                        CSV
                    </a>
                </div>
            </div>
            {error && !rows ? (
                <ErrorState message={error} retryLabel={t("重试", "Retry")} onRetry={load} />
            ) : !rows ? (
                <SkeletonLines rows={4} />
            ) : rows.length === 0 ? (
                <EmptyState title={t("暂无审计记录", "No audit records")} />
            ) : (
                <ul className="max-h-64 divide-y divide-border overflow-y-auto font-mono text-xs">
                    {rows.map((r, i) => (
                        <li key={r.id || i} className="flex gap-3 py-1.5">
                            <span className="shrink-0 text-text-muted">
                                {String(r.created_at ?? "")}
                            </span>
                            <span className="text-text-primary">{r.caller ?? "—"}</span>
                            <span className="text-text-secondary">{r.action ?? ""}</span>
                        </li>
                    ))}
                </ul>
            )}
        </section>
    );
}

function KeysSection() {
    const [keys, setKeys] = useState<ApiKeyRow[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [name, setName] = useState("");
    const [minted, setMinted] = useState<string | null>(null);

    const load = useCallback(async () => {
        try {
            setKeys((await listApiKeys()).keys);
            setError(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }, []);

    useEffect(() => {
        load();
    }, [load]);

    const create = async () => {
        const n = name.trim();
        if (!n) return;
        try {
            const r = await createApiKey(n);
            setMinted(r.key);
            setName("");
            load();
        } catch (e) {
            sdk.toast.show({
                level: "error",
                message: t(
                    `创建失败: ${e instanceof Error ? e.message : String(e)}`,
                    `Create failed: ${e instanceof Error ? e.message : String(e)}`,
                ),
            });
        }
    };

    const remove = async (prefix: string) => {
        try {
            await deleteApiKey(prefix);
            load();
        } catch (e) {
            sdk.toast.show({
                level: "error",
                message: e instanceof Error ? e.message : String(e),
            });
        }
    };

    return (
        <section className="rounded-card border border-border bg-bg-elevated p-4">
            <div className="mb-3 text-xs font-medium text-text-secondary">
                {t("API 密钥", "API keys")}
            </div>
            {error && !keys ? (
                <ErrorState message={error} retryLabel={t("重试", "Retry")} onRetry={load} />
            ) : !keys ? (
                <SkeletonLines rows={3} />
            ) : (
                <ul className="divide-y divide-border text-xs">
                    {keys.map((k, i) => (
                        <li key={k.key_hash_prefix || i} className="flex items-center justify-between gap-2 py-2">
                            <div className="min-w-0">
                                <span className="font-medium text-text-primary">
                                    {k.name ?? "—"}
                                </span>
                                <span className="ml-2 font-mono text-text-muted">
                                    {k.key_hash_prefix ?? ""}
                                </span>
                            </div>
                            <button
                                type="button"
                                onClick={() => k.key_hash_prefix && remove(k.key_hash_prefix)}
                                className="shrink-0 text-xs text-error hover:underline"
                            >
                                {t("删除", "Delete")}
                            </button>
                        </li>
                    ))}
                </ul>
            )}
            <div className="mt-3 flex gap-2">
                <input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && create()}
                    placeholder={t("密钥名称", "Key name")}
                    className="min-w-0 flex-1 rounded-md border border-border bg-bg px-2.5 py-1.5 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-brand"
                />
                <button
                    type="button"
                    disabled={!name.trim()}
                    onClick={create}
                    className="shrink-0 rounded-md bg-brand px-3 py-1.5 text-xs text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {t("生成", "Create")}
                </button>
            </div>
            {minted ? (
                <div className="mt-2 rounded-card border border-border bg-bg p-2.5 font-mono text-xs text-text-primary">
                    <div className="mb-1 text-text-secondary">
                        {t("仅显示一次，立即复制：", "Shown once — copy it now:")}
                    </div>
                    {minted}
                </div>
            ) : null}
        </section>
    );
}
