import { useEffect, useState } from "react";
import type { LangId } from "./lib/models";
import { LANGUAGES } from "./lib/models";
import { loadHandLandmarker } from "./lib/hands";
import { SignToSign } from "./pages/SignToSign";
import { SignToText } from "./pages/SignToText";
import { Teach } from "./pages/Teach";
import { ToSign } from "./pages/ToSign";
import { ACCENT } from "./theme";

type Tab = "sign-text" | "to-sign" | "sign-sign" | "teach";
const TABS: { id: Tab; label: string }[] = [
  { id: "sign-text", label: "Sign → Text" },
  { id: "to-sign", label: "Speech → Sign" },
  { id: "sign-sign", label: "Sign → Sign" },
  { id: "teach", label: "Teach" },
];

const read = <T extends string>(k: string, ok: readonly T[], d: T): T => {
  try {
    const v = localStorage.getItem(k) as T;
    return ok.includes(v) ? v : d;
  } catch {
    return d;
  }
};
const write = (k: string, v: string) => {
  try {
    localStorage.setItem(k, v);
  } catch {
    /* ignore */
  }
};

export function App() {
  const ids = Object.keys(LANGUAGES) as LangId[];
  const [langId, setLangId] = useState<LangId>(() => read("signbridge.lang", ids, "ASL"));
  const [tab, setTab] = useState<Tab>(() => read("signbridge.tab", TABS.map((t) => t.id), "sign-text"));
  const lang = LANGUAGES[langId];
  const accent = ACCENT[langId];

  useEffect(() => write("signbridge.lang", langId), [langId]);
  useEffect(() => write("signbridge.tab", tab), [tab]);
  useEffect(() => {
    // warm up the hand model while the user reads the page
    if (tab !== "to-sign") loadHandLandmarker().catch(() => {});
  }, [tab]);

  return (
    <div className="app" style={{ ["--accent" as string]: accent }}>
      <header className="top">
        <div className="brand">
          <Logo />
          <span>SignBridge</span>
        </div>
        <div className="seg seg--lang" role="radiogroup" aria-label="Sign language">
          {ids.map((id) => (
            <button key={id} role="radio" aria-checked={id === langId} data-on={id === langId} onClick={() => setLangId(id)} title={LANGUAGES[id].name}>
              {id}
            </button>
          ))}
        </div>
      </header>

      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={t.id === tab} data-on={t.id === tab} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      <p className="langnote">
        <strong>{lang.name}.</strong> {lang.note}
      </p>

      <main key={`${langId}-${tab}`}>
        {tab === "sign-text" && <SignToText lang={lang} accent={accent} />}
        {tab === "to-sign" && <ToSign lang={lang} accent={accent} />}
        {tab === "sign-sign" && <SignToSign lang={lang} accent={accent} />}
        {tab === "teach" && <Teach lang={lang} accent={accent} />}
      </main>

      <footer className="foot">
        Runs entirely in your browser with MediaPipe hand tracking. Video never leaves your device. Fingerspelling only: full signed languages also use
        movement, facial grammar and space, which this doesn't read yet.{" "}
        <a href="https://github.com/pawar17/SignBridge">Source and accuracy numbers</a>
      </footer>
    </div>
  );
}

function Logo() {
  return (
    <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden>
      <rect width="26" height="26" rx="7" fill="var(--accent)" />
      <g stroke="#fff" strokeWidth="1.8" strokeLinecap="round" fill="none">
        <path d="M8 20 L8 12 M8 20 L12 17 L13 8 M12 17 L16 16 L17 9 M16 16 L19 14 L20 10" />
      </g>
    </svg>
  );
}
