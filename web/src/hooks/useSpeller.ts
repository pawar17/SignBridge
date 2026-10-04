import { useEffect, useMemo, useRef, useState } from "react";
import type { Frame } from "../lib/hands";
import type { Language } from "../lib/models";
import type { Prediction } from "../lib/recognizer";
import { Recognizer, loadPersonal, taughtLetters } from "../lib/recognizer";

const HOLD_MS = 650; // same letter held this long -> commit
const REPEAT_MS = 2200; // keep holding -> type it again (for double letters)
const SPACE_MS = 1200; // hands down this long -> word break
const MIN_CONF = 0.6;
const WINDOW = 8;

export interface Live {
  letter: string | null;
  confidence: number;
  progress: number; // 0..1 toward committing
  source: Prediction["source"];
}

const EMPTY: Live = { letter: null, confidence: 0, progress: 0, source: "none" };

/**
 * Turns per-frame predictions into typed text: smooths over a short window,
 * commits a letter once it's held steady, and adds a space when the hands drop.
 */
export function useSpeller(lang: Language) {
  const recognizer = useMemo(() => new Recognizer(lang, loadPersonal(lang.id)), [lang]);
  const [ready, setReady] = useState(false);
  const [modelError, setModelError] = useState("");
  const [text, setText] = useState("");
  const [live, setLive] = useState<Live>(EMPTY);

  const s = useRef({
    hist: [] as Prediction[],
    cand: null as string | null,
    since: 0,
    committedAt: 0,
    locked: null as string | null,
    lastHand: 0,
    lastUi: 0,
  });

  useEffect(() => {
    setReady(false);
    setModelError("");
    recognizer.ready.then(
      () => setReady(true),
      () => setModelError("Couldn't load the recognition model. Check your connection and reload."),
    );
    s.current.hist = [];
  }, [recognizer]);

  const commit = (letter: string) => setText((t) => t + letter);

  const handle = (f: Frame) => {
    const st = s.current;
    const now = performance.now();
    if (!f.hands.length) {
      st.hist = [];
      st.cand = null;
      st.locked = null;
      if (st.lastHand && now - st.lastHand > SPACE_MS) {
        st.lastHand = 0;
        setText((t) => (t && !t.endsWith(" ") ? t + " " : t));
      }
      if (now - st.lastUi > 120) {
        st.lastUi = now;
        setLive(EMPTY);
      }
      return;
    }
    st.lastHand = now;
    if (!ready) return;

    const p = recognizer.predict(f.hands);
    st.hist.push(p);
    if (st.hist.length > WINDOW) st.hist.shift();
    const score = new Map<string, number>();
    st.hist.forEach((h) => h.letter && score.set(h.letter, (score.get(h.letter) ?? 0) + h.confidence));
    let top: string | null = null;
    let best = 0;
    score.forEach((v, k) => v > best && ((best = v), (top = k)));
    const conf = best / st.hist.length;

    if (top !== st.cand) {
      st.cand = top;
      st.since = now;
    }
    if (top !== st.locked) st.locked = st.locked && top === null ? st.locked : null;

    let progress = 0;
    if (top && conf >= MIN_CONF) {
      if (st.locked === top) {
        const held = now - st.committedAt;
        progress = Math.min(1, held / REPEAT_MS);
        if (held >= REPEAT_MS) {
          commit(top);
          st.committedAt = now;
        }
      } else {
        const held = now - st.since;
        progress = Math.min(1, held / HOLD_MS);
        if (held >= HOLD_MS) {
          commit(top);
          st.locked = top;
          st.committedAt = now;
        }
      }
    }
    if (now - st.lastUi > 80) {
      st.lastUi = now;
      setLive({ letter: top, confidence: conf, progress: st.locked === top ? 0 : progress, source: p.source });
    }
  };

  const needsTeaching = !lang.hasBaseModel && taughtLetters(recognizer.personal).length < 2;
  return {
    ready,
    modelError: needsTeaching ? `${lang.id} learns from you. Teach at least two letters in the Teach tab first.` : modelError,
    text,
    setText,
    live,
    handle,
    addSpace: () => setText((t) => (t && !t.endsWith(" ") ? t + " " : t)),
    backspace: () => setText((t) => t.slice(0, -1)),
    clear: () => setText(""),
  };
}

export function speak(text: string, langTag: string) {
  if (!("speechSynthesis" in window) || !text.trim()) return false;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text.trim());
  u.lang = langTag;
  const voice = window.speechSynthesis.getVoices().find((v) => v.lang === langTag) ?? window.speechSynthesis.getVoices().find((v) => v.lang.startsWith("en"));
  if (voice) u.voice = voice;
  window.speechSynthesis.speak(u);
  return true;
}
