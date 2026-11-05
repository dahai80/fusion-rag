import { useCallback, useEffect, useState } from "react";
import { EmptyState, ErrorState, SkeletonLines } from "@fusion/web-ui";
import {
    type SearchTemplateRow,
    createTemplate,
    deleteTemplate,
    listTemplates,
} from "../api";
import { sdk } from "../sdk";
import { t } from "../i18n";

// Issue #113: preset + custom search templates panel. Builtin templates
// cannot be deleted server-side (404 on DELETE) — delete button hidden for
// the four builtin names instead of showing a doomed action.
const BUILTIN = new Set(["general", "code", "design", "precise"]);

export function TemplatesPanel({ kbId }: { kbId: string }) {
    const [templates, setTemplates] = useState<SearchTemplateRow[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [name, setName] = useState("");
    const [alpha, setAlpha] = useState("0.7");
    const [busy, setBusy] = useState(false);

    const load = useCallback(async () => {
        try {
            setTemplates(await listTemplates(kbId));
            setError(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }, [kbId]);

    useEffect(() => {
        setTemplates(null);
        load();
    }, [load]);

    const create = async () => {
        const n = name.trim();
        if (!n || busy) return;
        setBusy(true);
        try {
            await createTemplate(kbId, { name: n, alpha: Number(alpha) || 0.7 });
            setName("");
            sdk.toast.show({
                level: "success",
                message: t("模板已创建", "Template created"),
            });
            load();
        } catch (e) {
            sdk.toast.show({
                level: "error",
                message: t(
                    `创建失败: ${e instanceof Error ? e.message : String(e)}`,
                    `Create failed: ${e instanceof Error ? e.message : String(e)}`,
                ),
            });
        } finally {
            setBusy(false);
        }
    };

    const remove = async (n: string) => {
        try {
            await deleteTemplate(kbId, n);
            sdk.toast.show({
                level: "success",
                message: t("模板已删除", "Template deleted"),
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
        <div className="space-y-3">
            <div className="flex gap-2">
                <input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && create()}
                    placeholder={t("模板名", "Template name")}
                    className="min-w-0 flex-1 rounded-md border border-border bg-bg px-2.5 py-1.5 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-brand"
                />
                <input
                    value={alpha}
                    onChange={(e) => setAlpha(e.target.value)}
                    placeholder="alpha"
                    className="w-16 shrink-0 rounded-md border border-border bg-bg px-2.5 py-1.5 font-mono text-xs text-text-primary outline-none focus:border-brand"
                />
                <button
                    type="button"
                    disabled={!name.trim() || busy}
                    onClick={create}
                    className="shrink-0 rounded-md bg-brand px-3 py-1.5 text-xs text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {t("新建", "Create")}
                </button>
            </div>
            {error && !templates ? (
                <ErrorState message={error} retryLabel={t("重试", "Retry")} onRetry={load} />
            ) : !templates ? (
                <SkeletonLines rows={3} />
            ) : templates.length === 0 ? (
                <EmptyState title={t("暂无模板", "No templates")} />
            ) : (
                <ul className="divide-y divide-border text-xs">
                    {templates.map((tpl) => (
                        <li key={tpl.name} className="flex items-center justify-between gap-2 py-2">
                            <div className="min-w-0">
                                <span className="font-medium text-text-primary">{tpl.name}</span>
                                <span className="ml-2 text-text-muted">
                                    alpha={tpl.alpha ?? "—"} · top_k={tpl.top_k ?? "—"} ·{" "}
                                    {tpl.rerank ? "rerank" : t("混合", "hybrid")}
                                </span>
                            </div>
                            {BUILTIN.has(tpl.name) ? null : (
                                <button
                                    type="button"
                                    onClick={() => remove(tpl.name)}
                                    className="shrink-0 text-xs text-error hover:underline"
                                >
                                    {t("删除", "Delete")}
                                </button>
                            )}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
