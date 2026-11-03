import { useEffect, useState } from "react";

import { sdk } from "./sdk";

// Union shell i18n rule (spec S9): pick copy by active language, zh default.
// Language authority is <html lang>: embedded, the shell pushes lang through
// the SDK bridge (host -> module) and the SDK writes documentElement.lang;
// standalone, we seed html lang from localStorage (same key the shell uses,
// `fusion.shell.lang`) or the navigator. A MutationObserver on html lang
// re-renders subscribers when the shell toggle fires.

export type Lang = "zh" | "en";

const LANG_KEY = "fusion.shell.lang";

function initialLang(): Lang {
    // Standalone: localStorage (the shell's key, so a user toggling in the
    // shell keeps their choice here) > navigator > html lang. Embedded, the
    // shell push overwrites html lang and the subscription takes over.
    try {
        const saved = localStorage.getItem(LANG_KEY);
        if (saved === "zh" || saved === "en") return saved;
    } catch {
        // storage unavailable
    }
    try {
        if (navigator.language?.toLowerCase().startsWith("en")) return "en";
    } catch {
        // navigator unavailable
    }
    return sdk.i18n.lang() === "en" ? "en" : "zh";
}

let current: Lang = initialLang();
const listeners = new Set<(l: Lang) => void>();

// Keep <html lang> authoritative at boot (standalone: nothing else does it).
if (typeof document !== "undefined") {
    document.documentElement.lang = current === "en" ? "en" : "zh-CN";
}

sdk.i18n.subscribe((l) => {
    if (l === current) return;
    current = l;
    for (const fn of listeners) fn(l);
});

export function t(zh: string, en: string): string {
    return current === "en" ? en : zh;
}

export function useLang(): Lang {
    const [lang, setLang] = useState<Lang>(current);
    useEffect(() => {
        listeners.add(setLang);
        return () => {
            listeners.delete(setLang);
        };
    }, []);
    return lang;
}

export function currentLang(): Lang {
    return current;
}

export function setLang(lang: Lang): void {
    // Standalone convenience; embedded the shell owns the toggle and this is
    // only reachable from its own UI (none today) — html lang drives t().
    if (typeof document !== "undefined") {
        document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
    }
    try {
        localStorage.setItem(LANG_KEY, lang);
    } catch {
        // storage unavailable — session-only
    }
}
