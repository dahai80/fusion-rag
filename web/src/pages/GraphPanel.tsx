import { useCallback, useEffect, useState } from "react";
import { EmptyState, ErrorState, SkeletonLines } from "@fusion/web-ui";
import {
    type GraphHit,
    type GraphStats,
    graphBuild,
    graphSearch,
    graphStats,
} from "../api";
import { sdk } from "../sdk";
import { t } from "../i18n";

// Issue #113: knowledge-graph tab — stats tiles, build action, search.
// Build is an LLM extraction pass over chunks (slow, seconds to minutes);
// UI keeps it explicit with a busy state, never auto-fires.
export function GraphPanel({ kbId }: { kbId: string }) {
    const [stats, setStats] = useState<GraphStats | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [building, setBuilding] = useState(false);
    const [query, setQuery] = useState("");
    const [hits, setHits] = useState<GraphHit[] | null>(null);
    const [searching, setSearching] = useState(false);

    const load = useCallback(async () => {
        try {
            setStats(await graphStats(kbId));
            setError(null);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }, [kbId]);

    useEffect(() => {
        setStats(null);
        setHits(null);
        setError(null);
        load();
    }, [load]);

    const build = async () => {
        if (building) return;
        setBuilding(true);
        try {
            const r = await graphBuild(kbId);
            sdk.toast.show({
                level: "success",
                message: t(
                    `图谱已构建：${r.entities} 实体 / ${r.relations} 关系`,
                    `Graph built: ${r.entities} entities / ${r.relations} relations`,
                ),
            });
            load();
        } catch (e) {
            sdk.toast.show({
                level: "error",
                message: t(
                    `构建失败: ${e instanceof Error ? e.message : String(e)}`,
                    `Build failed: ${e instanceof Error ? e.message : String(e)}`,
                ),
            });
        } finally {
            setBuilding(false);
        }
    };

    const runSearch = async () => {
        const q = query.trim();
        if (!q || searching) return;
        setSearching(true);
        try {
            setHits(await graphSearch(kbId, q));
        } catch (e) {
            sdk.toast.show({
                level: "error",
                message: t(
                    `搜索失败: ${e instanceof Error ? e.message : String(e)}`,
                    `Search failed: ${e instanceof Error ? e.message : String(e)}`,
                ),
            });
        } finally {
            setSearching(false);
        }
    };

    return (
        <div className="space-y-4">
            <div className="flex items-center justify-between gap-2">
                <div className="text-xs text-text-muted">
                    {stats
                        ? t(
                              `${stats.entities} 实体 · ${stats.relations} 关系`,
                              `${stats.entities} entities · ${stats.relations} relations`,
                          )
                        : ""}
                </div>
                <button
                    type="button"
                    disabled={building}
                    onClick={build}
                    className="rounded-md bg-brand px-3 py-1.5 text-xs text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {building
                        ? t("构建中…", "Building…")
                        : t("构建图谱", "Build graph")}
                </button>
            </div>
            {error && !stats ? (
                <ErrorState message={error} retryLabel={t("重试", "Retry")} onRetry={load} />
            ) : !stats ? (
                <SkeletonLines rows={3} />
            ) : (
                <div className="flex gap-2">
                    <input
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && runSearch()}
                        placeholder={t(
                            "图检索：输入实体或问题…",
                            "Graph search: entity or question…",
                        )}
                        className="min-w-0 flex-1 rounded-md border border-border bg-bg px-2.5 py-1.5 text-xs text-text-primary outline-none placeholder:text-text-muted focus:border-brand"
                    />
                    <button
                        type="button"
                        disabled={!query.trim() || searching}
                        onClick={runSearch}
                        className="shrink-0 rounded-md bg-brand px-3 py-1.5 text-xs text-white disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        {t("搜索", "Search")}
                    </button>
                </div>
            )}
            {searching ? (
                <SkeletonLines rows={3} />
            ) : hits === null ? null : hits.length === 0 ? (
                <EmptyState
                    title={t("无图谱命中", "No graph hits")}
                    description={t(
                        "先构建图谱，或换个查询词",
                        "Build the graph first, or rephrase",
                    )}
                />
            ) : (
                <ul className="space-y-2 text-xs">
                    {hits.map((h, i) => (
                        <li
                            key={h.chunk_id || i}
                            className="rounded-card border border-border p-2.5"
                        >
                            <div className="mb-1 text-text-secondary">
                                {h.doc_id || h.chunk_id}
                            </div>
                            <p className="leading-relaxed text-text-primary">
                                {String(h.text ?? "").slice(0, 300)}
                            </p>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
