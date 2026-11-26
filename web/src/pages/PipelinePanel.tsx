import { useEffect, useState } from "react";
import { type PipelineStats, fetchPipeline } from "../api";
import { t } from "../i18n";
import { sdk } from "../sdk";

// Issue #163: 检索管线 zone from prototype 04-knowledge-qa — stage-by-stage
// pipeline view with live counters (docs/chunks/queries) and per-stage
// latency where the backend instruments it. Stages without instrumentation
// render with no numbers (null from the API), never invented ones.
export function PipelinePanel({ kbId }: { kbId: string }) {
    const [stats, setStats] = useState<PipelineStats | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        setStats(null);
        setError(null);
        let alive = true;
        fetchPipeline(kbId)
            .then((s) => {
                if (alive) setStats(s);
            })
            .catch((e) => {
                if (alive) {
                    const msg = e instanceof Error ? e.message : String(e);
                    setError(msg);
                    sdk.toast.show({
                        level: "error",
                        message: t(`管线加载失败: ${msg}`, `Pipeline load failed: ${msg}`),
                    });
                }
            });
        return () => {
            alive = false;
        };
    }, [kbId]);

    return (
        <section className="mt-4 rounded-card border border-border">
            <div className="flex flex-wrap items-center gap-x-3 border-b border-border px-3 py-2 text-xs font-medium text-text-secondary">
                <span>{t("检索管线", "Retrieval pipeline")}</span>
                {stats && (
                    <span className="text-text-muted">
                        {t(
                            `文档 ${stats.docs} · 分块 ${stats.chunks} · 查询 ${stats.queries}`,
                            `docs ${stats.docs} · chunks ${stats.chunks} · queries ${stats.queries}`,
                        )}
                        {stats.ask_latency != null && (
                            <>
                                {" · "}
                                {t(`/ask 均延迟 ${stats.ask_latency} ms`, `/ask avg ${stats.ask_latency} ms`)}
                            </>
                        )}
                    </span>
                )}
            </div>
            <div className="space-y-1.5 p-3 text-xs">
                {error && (
                    <div className="py-2 text-center text-text-muted">
                        {t("管线统计不可用", "Pipeline stats unavailable")} ({error})
                    </div>
                )}
                {!error && !stats && (
                    <div className="py-2 text-center text-text-muted">
                        {t("加载中…", "Loading…")}
                    </div>
                )}
                {stats &&
                    stats.stages.map((s, i) => (
                        <div key={s.id} className="flex items-center gap-2">
                            {i > 0 && (
                                <span className="text-text-muted" aria-hidden>
                                    →
                                </span>
                            )}
                            <div className="flex min-w-0 flex-1 items-center justify-between gap-2 rounded-lg border border-border/70 px-2.5 py-1.5">
                                <span className="truncate text-text-primary">
                                    {t(s.label_zh, s.label_en)}
                                </span>
                                <span className="shrink-0 tabular-nums text-text-muted">
                                    {s.calls != null
                                        ? t(
                                              `${s.calls} 次 · 均 ${s.avg_ms} ms`,
                                              `${s.calls} calls · avg ${s.avg_ms} ms`,
                                          )
                                        : t("—", "—")}
                                </span>
                            </div>
                        </div>
                    ))}
            </div>
        </section>
    );
}
