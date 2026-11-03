import { useCallback, useEffect, useState } from "react";
import {
    StatusBadge,
    type StatusKind,
    ErrorState,
    SkeletonLines,
} from "@fusion/web-ui";
import {
    type HealthInfo,
    type KbSummary,
    createKb,
    fetchHealth,
    fetchKbs,
} from "../api";
import { t } from "../i18n";
import { KbDetail } from "./KbDetail";

// Console layout: header (title + health) / left KB list / right KB detail.
// Every data view implements the four mandatory states (loading / empty /
// error+retry / populated) per the union web UI spec §6.1.

export function ConsolePage() {
    const [kbs, setKbs] = useState<KbSummary[] | null>(null);
    const [kbError, setKbError] = useState<string | null>(null);
    const [health, setHealth] = useState<HealthInfo | null>(null);
    const [selected, setSelected] = useState<KbSummary | null>(null);

    const loadKbs = useCallback(async () => {
        try {
            const list = await fetchKbs();
            setKbs(list);
            setKbError(null);
            setSelected((cur) =>
                cur ? (list.find((k) => k.id === cur.id) ?? null) : cur,
            );
        } catch (e) {
            setKbError(e instanceof Error ? e.message : String(e));
        }
    }, []);

    const loadHealth = useCallback(async () => {
        try {
            setHealth(await fetchHealth());
        } catch {
            setHealth(null);
        }
    }, []);

    useEffect(() => {
        loadKbs();
        loadHealth();
        const timer = setInterval(loadHealth, 30000);
        return () => clearInterval(timer);
    }, [loadKbs, loadHealth]);

    return (
        <div className="mx-auto max-w-5xl space-y-4 px-4 py-4">
            <Header health={health} />
            <div className="grid gap-4 md:grid-cols-5">
                <KbListPanel
                    kbs={kbs}
                    error={kbError}
                    selected={selected}
                    onSelect={setSelected}
                    onReload={loadKbs}
                    onCreated={loadKbs}
                />
                <KbDetail
                    kb={selected}
                    onDeleted={() => {
                        setSelected(null);
                        loadKbs();
                    }}
                />
            </div>
        </div>
    );
}

function healthKind(h: HealthInfo | null): StatusKind {
    if (!h) return "offline";
    const s = String(h.status ?? "ok").toLowerCase();
    if (s === "ok" || s === "healthy") return "success";
    if (s === "degraded") return "warning";
    if (s === "unhealthy" || s === "error") return "error";
    return "offline";
}

function Header({ health }: { health: HealthInfo | null }) {
    return (
        <header className="flex items-center justify-between border-b border-border pb-4">
            <div>
                <h1 className="text-lg font-semibold tracking-tight text-text-primary">
                    {t("Fusion RAG 控制台", "Fusion RAG Console")}
                </h1>
                <p className="mt-0.5 font-mono text-xs text-text-muted">
                    {health?.version
                        ? `v${health.version} · llm_auth=${health.llm_auth ?? "?"}`
                        : ""}
                </p>
            </div>
            <StatusBadge
                status={healthKind(health)}
                label={
                    health
                        ? t("运行中", "Running")
                        : t("不可达", "Unreachable")
                }
            />
        </header>
    );
}

function KbListPanel({
    kbs,
    error,
    selected,
    onSelect,
    onReload,
    onCreated,
}: {
    kbs: KbSummary[] | null;
    error: string | null;
    selected: KbSummary | null;
    onSelect: (kb: KbSummary) => void;
    onReload: () => void;
    onCreated: () => void;
}) {
    const [name, setName] = useState("");
    const [creating, setCreating] = useState(false);

    const submit = async () => {
        const n = name.trim();
        if (!n || creating) return;
        setCreating(true);
        try {
            const kb = await createKb(n);
            setName("");
            onCreated();
            onSelect(kb);
        } catch (e) {
            console.error("[rag-web] create kb fail:", e);
        } finally {
            setCreating(false);
        }
    };

    return (
        <aside className="rounded-card border border-border bg-bg-elevated md:col-span-2">
            <div className="border-b border-border px-4 py-3">
                <div className="flex gap-2">
                    <input
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && submit()}
                        placeholder={t("新知识库名称", "New KB name")}
                        className="w-full min-w-0 flex-1 rounded-md border border-border bg-bg px-3 py-1.5 text-sm text-text-primary outline-none placeholder:text-text-muted focus:border-brand"
                    />
                    <button
                        type="button"
                        disabled={!name.trim() || creating}
                        onClick={submit}
                        className="shrink-0 rounded-md bg-brand px-3 py-1.5 text-sm text-white disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        {t("建库", "Create")}
                    </button>
                </div>
            </div>
            {error && !kbs ? (
                <ErrorState
                    message={error}
                    retryLabel={t("重试", "Retry")}
                    onRetry={onReload}
                />
            ) : !kbs ? (
                <div className="p-4">
                    <SkeletonLines rows={4} />
                </div>
            ) : kbs.length === 0 ? (
                <div className="px-4 py-10 text-center text-xs text-text-muted">
                    {t("暂无知识库，先建一个", "No KBs yet — create one")}
                </div>
            ) : (
                <ul className="max-h-96 divide-y divide-border overflow-y-auto">
                    {kbs.map((kb) => {
                        const active = selected?.id === kb.id;
                        return (
                            <li key={kb.id}>
                                <button
                                    type="button"
                                    onClick={() => onSelect(kb)}
                                    className={`w-full px-4 py-2.5 text-left text-sm ${
                                        active
                                            ? "bg-bg-muted text-text-primary"
                                            : "text-text-primary hover:bg-bg-muted/60"
                                    }`}
                                >
                                    <div className="font-medium">{kb.name}</div>
                                    <div className="mt-0.5 text-xs text-text-muted">
                                        {kb.id} · {kb.embedding_model} ·{" "}
                                        {kb.file_count ?? 0}{" "}
                                        {t("个文档", "docs")}
                                    </div>
                                </button>
                            </li>
                        );
                    })}
                </ul>
            )}
        </aside>
    );
}
