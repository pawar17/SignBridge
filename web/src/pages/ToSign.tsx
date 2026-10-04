import { useEffect, useRef, useState } from "react";
import { SpellPlayer } from "../components/SpellPlayer";
import type { Language } from "../lib/models";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const SR: any = typeof window !== "undefined" ? (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition : null;

export function ToSign({ lang, accent }: { lang: Language; accent: string }) {
  const [draft, setDraft] = useState("");
  const [text, setText] = useState("");
  const [listening, setListening] = useState(false);
  const [micError, setMicError] = useState("");
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const rec = useRef<any>(null);

  useEffect(() => () => rec.current?.abort(), []);

  const listen = () => {
    if (listening) {
      rec.current?.stop();
      return;
    }
    setMicError("");
    const r = new SR();
    r.lang = lang.speechLang;
    r.interimResults = true;
    r.continuous = false;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    r.onresult = (e: any) => {
      let said = "";
      for (let k = 0; k < e.results.length; k++) said += e.results[k][0].transcript;
      setDraft(said);
      if (e.results[e.results.length - 1].isFinal) setText(said);
    };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    r.onerror = (e: any) => setMicError(e.error === "not-allowed" ? "Microphone access was blocked. Allow it for this site, or type instead." : "Didn't catch that. Try again or type it.");
    r.onend = () => setListening(false);
    rec.current = r;
    r.start();
    setListening(true);
  };

  return (
    <div className="split split--even">
      <div className="split__side">
        <form
          className="card compose"
          onSubmit={(e) => {
            e.preventDefault();
            setText(draft);
          }}
        >
          <label htmlFor="say">Say or type a word or name</label>
          <textarea id="say" rows={3} value={draft} placeholder="e.g. hello, or your name" onChange={(e) => setDraft(e.target.value)} />
          <div className="row">
            {SR ? (
              <button type="button" className={`btn ${listening ? "btn--rec" : ""}`} onClick={listen}>
                <span className="dot" /> {listening ? "Listening… tap to stop" : "Speak"}
              </button>
            ) : (
              <span className="muted small">Voice input needs Chrome, Edge or Safari.</span>
            )}
            <span className="grow" />
            <button type="submit" className="btn btn--primary" disabled={!draft.trim()}>
              Fingerspell in {lang.id}
            </button>
          </div>
          {micError && <p className="note note--warn">{micError}</p>}
        </form>
        <p className="muted small explain">
          Fingerspelling is how signers spell names and words that have no set sign.{" "}
          {lang.hasBaseModel
            ? `Each handshape is a real one from public ${lang.id} landmark data, or yours if you've taught that letter.`
            : "BSL handshapes come from the letters you teach in the Teach tab."}
        </p>
      </div>
      <div className="split__main">
        <SpellPlayer text={text} lang={lang.id} accent={accent} />
      </div>
    </div>
  );
}
