import { useEffect, useState } from "react";
import { AdminPage } from "./pages/AdminPage";
import { ConsolePage } from "./pages/ConsolePage";

// Hash routing, two views: #/ = console, #/admin = admin surface (issue
// #113). No router dep — same pattern as the shell modules.
type Route = "console" | "admin";

function parseHash(): Route {
    const h = window.location.hash.replace(/^#\/?/, "");
    return h.startsWith("admin") ? "admin" : "console";
}

export function App() {
    const [route, setRoute] = useState<Route>(parseHash());

    useEffect(() => {
        const onHash = () => setRoute(parseHash());
        window.addEventListener("hashchange", onHash);
        return () => window.removeEventListener("hashchange", onHash);
    }, []);

    return route === "admin" ? <AdminPage /> : <ConsolePage />;
}
