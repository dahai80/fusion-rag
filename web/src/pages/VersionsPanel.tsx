import { useCallback, useEffect, useState } from "react";
import { ConfirmDialog, EmptyState, ErrorState, SkeletonLines } from "@fusion/web-ui";
import {
    type Snapshot,
    createVersion,
    deleteVersion,
    listVersions,
    rollbackVersion,
} from "../api";
import { sdk } from "../sdk";
import { t } from "../i18n";

// Issue #113: version snapshots + rollback panel. Rollback and delete are
// destructive — ConfirmDialog per spec S5 (no native confirm in the shell).
export function VersionsPanel({ kbId }: { kbId: string }) {
    const [versions, setVersions] = useState<Snapshot[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [desc, setDesc] = useState("");
    const [busy, setBusy] = useState(false);
    const [confirm, setConfirm] = useState<{ kind: "rollback" | "delete"; v: Snapshot } | null>(null);

    const load = useCallback(async () => {
        try {
            setVersions(await listVersions(kbId));
            setError(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }, [kbId]);

    useEffect(() => {
        setVersions(null);
        load();
    }, [load]);

    const snapshot = async () => {
        if (busy) return;
        setBusy(true);
        try {
            await createVersion(kbId, desc.trim());
            setDesc("");
            sdk.toast.show({
                level: "success",
                message: t("快照已创建", "Snapshot created"),
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

    const act = async () => {
        if (!confirm) return;
        const { kind, v } = confirm;
        setConfirm(null);
        try {
            if (kind === "rollback") {
                await rollbackVersion(kbId, v.version_id);
                sdk.toast.show({
                    level: "success",
                    message: t(`已回滚到 ${v.version_id}`, `Rolled back to ${v.version_id}`),
                });
            } else {
                await deleteVersion(kbId, v.version_id);
                sdk.toast.show({
                    level: "success",
                    message: t("快照已删除", "Snapshot deleted"),
                });
            }
            load();
        } catch (e) {
            sdk.toast.show({
                level: "error",
                message: t(
                    `操作失败: ${e instanceof Error ? e.message : String(e)}`,
                    `Operation failed: ${e instanceof Error ? e.message : String(e)}`,
                ),
            });
        }
    };

    return (
        <div className="space-y-3">
            <div className="flex gap-2">
                <input
                    value={desc}
                    onChange={(e) => setDesc(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && snapshot()}
                    placeholder={t("快照描述（可选）", "Snapshot description (optional)")}
                    className="min-w-0 flex-1 rounded-md border border-border bg-bg px-2.5 py-1.5 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-brand"
                />
                <button
                    type="button"
                    disabled={busy}
                    onClick={snapshot}
                    className="shrink-0 rounded-md bg-brand px-3 py-1.5 text-xs text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {t("创建快照", "Snapshot")}
                </button>
            </div>
            {error && !versions ? (
                <ErrorState message={error} retryLabel={t("重试", "Retry")} onRetry={load} />
            ) : !versions ? (
                <SkeletonLines rows={3} />
            ) : versions.length === 0 ? (
                <EmptyState
                    title={t("暂无快照", "No snapshots")}
                    description={t(
                        "创建快照以支持回滚",
                        "Create a snapshot to enable rollback",
                    )}
                />
            ) : (
                <ul className="divide-y divide-border text-xs">
                    {versions.map((v) => (
                        <li key={v.version_id} className="flex items-center justify-between gap-2 py-2">
                            <div className="min-w-0">
                                <div className="font-mono text-text-primary">{v.version_id}</div>
                                <div className="truncate text-text-muted">
                                    {v.description || "—"}
                                    {v.created_at ? ` · ${String(v.created_at)}` : ""}
                                </div>
                            </div>
                            <div className="flex shrink-0 gap-2">
                                <button
                                    type="button"
                                    onClick={() => setConfirm({ kind: "rollback", v })}
                                    className="text-xs text-brand hover:underline"
                                >
                                    {t("回滚", "Rollback")}
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setConfirm({ kind: "delete", v })}
                                    className="text-xs text-error hover:underline"
                                >
                                    {t("删除", "Delete")}
                                </button>
                            </div>
                        </li>
                    ))}
                </ul>
            )}
            <ConfirmDialog
                open={confirm !== null}
                danger={confirm?.kind === "delete"}
                title={
                    confirm?.kind === "rollback"
                        ? t(`回滚到 ${confirm.v.version_id}？`, `Roll back to ${confirm.v.version_id}?`)
                        : t(`删除快照 ${confirm?.v.version_id ?? ""}？`, `Delete snapshot ${confirm?.v.version_id ?? ""}?`)
                }
                description={
                    confirm?.kind === "rollback"
                        ? t("当前库内容将被快照内容覆盖。", "Live KB content will be replaced by the snapshot.")
                        : t("删除后不可恢复。", "This cannot be undone.")
                }
                confirmLabel={confirm?.kind === "rollback" ? t("确认回滚", "Roll back") : t("确认删除", "Delete")}
                cancelLabel={t("取消", "Cancel")}
                confirmName={confirm?.kind === "rollback" ? undefined : confirm?.v.version_id}
                onConfirm={act}
                onCancel={() => setConfirm(null)}
            />
        </div>
    );
}
