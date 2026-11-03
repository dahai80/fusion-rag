import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { sdk } from "./sdk";
import "@fusion/web-ui/tokens.css";
import "./index.css";

// Shell theme rule: html.light switches the token palette. Seed from the SDK
// (host pull / document default) and follow host pushes — without this the
// embedded console stays dark when the shell is toggled to light.
document.documentElement.classList.toggle("light", sdk.theme.get() === "light");
sdk.theme.subscribe((mode) => {
    document.documentElement.classList.toggle("light", mode === "light");
});

const rootEl = document.getElementById("root");
if (rootEl) {
    createRoot(rootEl).render(
        <StrictMode>
            <App />
        </StrictMode>,
    );
} else {
    console.error("[rag-web] #root element missing");
}
