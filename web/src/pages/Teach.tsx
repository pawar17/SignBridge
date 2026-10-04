import { useEffect, useMemo, useRef, useState } from "react";
import { CameraView } from "../components/CameraView";
import { HandPose } from "../components/HandPose";
import type { Hand } from "../lib/features";
import type { Frame } from "../lib/hands";
import type { Language } from "../lib/models";
import type { PersonalSet, Prediction } from "../lib/recognizer";
import { Recognizer, addSamples, clearLetter, loadPersonal, poseFor, savePersonal } from "../lib/recognizer";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const RECORD_MS = 2000;
const TARGET = 30;
const MIN_FRAMES = 10;
const MAX_MS = 5000;

type Phase = { kind: "idle" } | { kind: "count"; n: number } | { kind: "rec"; got: number } | { kind: "done"; got: number };

export function Teach({ lang, accent }: { lang: Language; accent: string }) {
  const [set, setSet] = useState<PersonalSet>(() => loadPersonal(lang.id));
  const [letter, setLetter] = useState("A");
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [ref, setRef] = useState<Hand[] | null>(null);
  const [check, setCheck] = useState<Prediction | null>(null);
  const frames = useRef<Hand[][]>([]);
  const recUntil = useRef(0);
  const tick = useRef(0);
  const lastCheck = useRef(0);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => setSet(loadPersonal(lang.id)), [lang.id]);
  useEffect(() => {
    let alive = true;
    poseFor(lang.id, letter).then((p) => alive && setRef(p));
    return () => {
      alive = false;
    };
  }, [letter, lang.id, set]);

  const recognizer = useMemo(() => new Recognizer(lang, set), [lang, set]);

  const onFrame = (f: Frame) => {
    const now = performance.now();
    if (recUntil.current) {
      // sample frames at most every 60 ms; on slow devices keep going (up to 5 s) until there are enough
      if (f.hands.length && now - tick.current >= 60) {
        tick.current = now;
        frames.current.push(lang.hands === "one" ? [f.hands[0]] : f.hands.slice(0, 2));
        setPhase({ kind: "rec", got: frames.current.length });
      }
      const got = frames.current.length;
      const elapsed = now - (recUntil.current - RECORD_MS);
      if (got >= TARGET || (elapsed > RECORD_MS && got >= MIN_FRAMES) || elapsed > MAX_MS) {
        recUntil.current = 0;
        if (got >= 3) setSet(addSamples(lang, set, letter, frames.current));
        setPhase({ kind: "done", got });
      }
      return;
    }
    if (now - lastCheck.current > 120) {
      lastCheck.current = now;
      setCheck(f.hands.length ? recognizer.predict(f.hands) : null);
    }
  };

  const record = () => {
    frames.current = [];
    let n = 3;
    setPhase({ kind: "count", n });
    const t = setInterval(() => {
      n -= 1;
      if (n > 0) setPhase({ kind: "count", n });
      else {
        clearInterval(t);
        tick.current = 0;
        recUntil.current = performance.now() + RECORD_MS;
        setPhase({ kind: "rec", got: 0 });
      }
    }, 700);
  };

  const exportSet = () => {
    const blob = new Blob([JSON.stringify({ language: lang.id, ...set })], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `signbridge-${lang.id.toLowerCase()}-signs.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const importSet = async (file: File) => {
    try {
      const data = JSON.parse(await file.text());
      if (data.language && data.language !== lang.id) {
        alert(`That file has ${data.language} signs. Switch to ${data.language} first.`);
        return;
      }
      const next: PersonalSet = { samples: data.samples ?? {}, poses: data.poses ?? {} };
      savePersonal(lang.id, next);
      setSet(next);
    } catch {
      alert("That file isn't a SignBridge signs file.");
    }
  };

  const count = (l: string) => set.samples[l]?.length ?? 0;
  const taught = LETTERS.filter((l) => count(l) > 0).length;
  const busy = phase.kind === "count" || phase.kind === "rec";

  return (
    <div className="split">
      <div className="split__main">
        <CameraView
          onFrame={onFrame}
          accent={accent}
          badge={
            phase.kind === "count"
              ? { text: String(phase.n), sub: `Get ready to sign ${letter}` }
              : phase.kind === "rec"
                ? { text: letter, sub: `Recording ${phase.got}/${TARGET}` }
                : check?.letter && check.confidence >= 0.5
                  ? { text: check.letter, sub: "reads as" }
                  : null
          }
        />
        <p className="muted small">
          Teaching adapts SignBridge to your hands, camera and signing style. Signs are saved in this browser only. {lang.hasBaseModel ? `They blend with the ${lang.id} model.` : "BSL has no public dataset yet, so it learns entirely from you: teach at least two letters to start."}
        </p>
      </div>
      <div className="split__side">
        <div className="card">
          <div className="teach__head">
            <HandPose hands={ref} accent={accent} size={132} label={`Reference for ${letter}`} />
            <div className="teach__info">
              <div className="teach__letter" style={{ color: accent }}>
                {letter}
              </div>
              <p className="small muted">{count(letter) ? `${count(letter)} samples taught` : lang.hasBaseModel ? "Using the base model" : "Not taught yet"}</p>
              <div className="row">
                <button className="btn btn--primary" onClick={record} disabled={busy}>
                  {busy ? "Recording…" : count(letter) ? "Add more" : `Teach ${letter}`}
                </button>
                {count(letter) > 0 && (
                  <button className="btn" disabled={busy} onClick={() => setSet(clearLetter(lang.id, set, letter))}>
                    Reset
                  </button>
                )}
              </div>
              {phase.kind === "done" && <p className="small">{phase.got >= 3 ? `Saved ${phase.got} frames for ${letter}.` : "No hand seen. Turn on the camera and keep your hand in view."}</p>}
            </div>
          </div>
          <p className="small muted">Press Teach, wait for the countdown, then hold {letter} for 2 seconds. Move your hand slightly while holding so it learns a few angles.</p>
        </div>

        <div className="card">
          <div className="row">
            <h3>
              {taught}/26 letters taught
            </h3>
            <span className="grow" />
            <button className="btn btn--small" onClick={exportSet} disabled={!taught}>
              Export
            </button>
            <button className="btn btn--small" onClick={() => fileRef.current?.click()}>
              Import
            </button>
            <input ref={fileRef} type="file" accept="application/json" hidden onChange={(e) => e.target.files?.[0] && importSet(e.target.files[0])} />
          </div>
          <div className="letters">
            {LETTERS.map((l) => (
              <button key={l} className="letters__cell" data-on={l === letter} data-taught={count(l) > 0} onClick={() => !busy && setLetter(l)} style={l === letter ? { borderColor: accent } : undefined}>
                {l}
                {count(l) > 0 && <i style={{ background: accent }} />}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
