import { useEffect, useRef, useState } from "react";
import { type Citation, askKb } from "../api";
import { t } from "../i18n";
import { sdk } from "../sdk";

// Issue: console chat tab — RAG Q&A over /ask. Answers carry per-chunk
// citations (doc/page/score, low-confidence flagged). No markdown lib —
// plain text, whitespace-pre-wrap; this is an evidence tester, not a chat
// product.
interface ChatMsg {
    role: "user" | "assistant";
    text: string;
    citations?: Citation[];
    hasLowConfidence?: boolean;
}

export function ChatPanel({ kbId }: { kbId: string }) {
    const [msgs, setMsgs] = useState<ChatMsg[]>([]);
    const [input, setInput] = useState("");
    const [busy, setBusy] = useState(false);
    const listRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        setMsgs([]);
        setInput("");
    }, [kbId]);

    useEffect(() => {
        listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
    }, [msgs, busy]);

    const send = async () => {
        const q = input.trim();
        if (!q || busy) return;
        setInput("");
        setMsgs((m) => [...m, { role: "user", text: q }]);
        setBusy(true);
        try {
            const r = await askKb(kbId, q);
            setMsgs((m) => [
                ...m,
                {
                    role: "assistant",
                    text: r.answer,
                    citations: r.citations || [],
                    hasLowConfidence: r.has_low_confidence,
                },
            ]);
        } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            setMsgs((m) => [...m, { role: "assistant", text: msg }]);
            sdk.toast.show({
                level: "error",
                message: t(`问答失败: ${msg}`, `Ask failed: ${msg}`),
            });
        } finally {
            setBusy(false);
        }
    };

    return (
        <section className="rounded-card border border-border">
            <div className="border-b border-border px-3 py-2 text-xs font-medium text-text-secondary">
                {t("问答", "Chat")}
                <span className="ml-2 text-text-muted">
                    {t("调用 /ask：检索 + 生成，答案带引用", "calls /ask: retrieval + generation with citations")}
                </span>
            </div>
            <div ref={listRef} className="max-h-[48rem] min-h-80 space-y-3 overflow-y-auto p-3 text-xs">
                {msgs.length === 0 && (
                    <div className="py-6 text-center text-text-muted">
                        {t("问一个关于此知识库的问题", "Ask a question about this KB")}
                    </div>
                )}
                {msgs.map((m, i) =>
                    m.role === "user" ? (
                        <div key={i} className="flex justify-end">
                            <div className="max-w-[85%] whitespace-pre-wrap rounded-xl rounded-br-sm bg-brand px-3 py-2 text-white">
                                {m.text}
                            </div>
                        </div>
                    ) : (
                        <div key={i} className="flex justify-start">
                            <div className="max-w-[85%] space-y-2">
                                <div className="whitespace-pre-wrap rounded-xl rounded-bl-sm border border-border bg-bg-muted px-3 py-2 text-text-primary">
                                    {m.text}
                                </div>
                                {m.hasLowConfidence && (
                                    <div className="text-[10px] text-error">
                                        {t("⚠ 部分引用置信度低（OCR）", "⚠ Some citations have low OCR confidence")}
                                    </div>
                                )}
                                {m.citations && m.citations.length > 0 && (
                                    <div className="flex flex-wrap gap-1.5">
                                        {m.citations.map((c, j) => (
                                            <span
                                                key={`${c.chunk_id || j}`}
                                                title={c.doc_id || ""}
                                                className="rounded-md border border-border bg-bg px-2 py-0.5 text-[10px] text-text-secondary"
                                            >
                                                [{j + 1}] {c.doc_name}
                                                {c.page ? ` · p${c.page}` : ""}
                                                {typeof c.score === "number"
                                                    ? ` · ${c.score.toFixed(3)}`
                                                    : ""}
                                                {c.low_confidence ? " · ⚠" : ""}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    ),
                )}
                {busy && (
                    <div className="flex justify-start">
                        <div className="rounded-xl rounded-bl-sm border border-border bg-bg-muted px-3 py-2 text-text-muted">
                            {t("检索并生成中…", "Retrieving and generating…")}
                        </div>
                    </div>
                )}
            </div>
            <div className="flex gap-2 border-t border-border p-3">
                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                        // Enter sends, Shift+Enter inserts a newline; IME
                        // composition guarded so Chinese typing is safe.
                        if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                            e.preventDefault();
                            send();
                        }
                    }}
                    rows={2}
                    placeholder={t("输入问题…（Enter 发送，Shift+Enter 换行）", "Type a question… (Enter to send, Shift+Enter for newline)")}
                    disabled={busy}
                    className="min-w-0 flex-1 resize-none rounded-md border border-border bg-bg px-2.5 py-2 text-xs leading-relaxed text-text-primary outline-none placeholder:text-text-muted focus:border-brand disabled:opacity-50"
                />
                <button
                    type="button"
                    disabled={!input.trim() || busy}
                    onClick={send}
                    className="shrink-0 self-end rounded-md bg-brand px-3 py-1.5 text-xs text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {t("发送", "Send")}
                </button>
            </div>
        </section>
    );
}
