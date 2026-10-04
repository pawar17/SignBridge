import { useEffect, useMemo, useState } from "react";
import type { Hand } from "../lib/features";
import type { LangId } from "../lib/models";
import { poseFor } from "../lib/recognizer";
import { HandPose } from "./HandPose";

interface Props {
  text: string;
  lang: LangId;
  accent: string;
  autoplay?: boolean;
}

const SPEEDS = [
  { label: "Slow", ms: 1400 },
  { label: "Normal", ms: 900 },
  { label: "Fast", ms: 550 },
];
const MOTION: Partial<Record<LangId, string[]>> = { ASL: ["J", "Z"] };

/** Plays text as a fingerspelling sequence, one handshape per letter. */
export function SpellPlayer({ text, lang, accent, autoplay = true }: Props) {
  const chars = useMemo(() => text.toUpperCase().replace(/[^A-Z ]/g, "").replace(/\s+/g, " ").trim().split(""), [text]);
  const [poses, setPoses] = useState<Record<string, Hand[] | null>>({});
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(autoplay);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    let alive = true;
    const letters = [...new Set(chars.filter((c) => c !== " "))];
    Promise.all(letters.map(async (l) => [l, await poseFor(lang, l)] as const)).then((r) => alive && setPoses(Object.fromEntries(r)));
    return () => {
      alive = false;
    };
  }, [chars, lang]);

  useEffect(() => {
    setI(0);
    setPlaying(autoplay && chars.length > 0);
  }, [chars, autoplay]);

  useEffect(() => {
    if (!playing || !chars.length) return;
    const ms = chars[i] === " " ? SPEEDS[speed].ms * 0.6 : SPEEDS[speed].ms;
    const t = setTimeout(() => {
      if (i + 1 >= chars.length) setPlaying(false);
      else setI(i + 1);
    }, ms);
    return () => clearTimeout(t);
  }, [playing, i, chars, speed]);

  if (!chars.length) {
    return (
      <div className="player player--empty">
        <p>Type or say something and it will be fingerspelled here.</p>
      </div>
    );
  }

  const c = chars[i] ?? " ";
  const missing = [...new Set(chars.filter((ch) => ch !== " " && poses[ch] === null))];

  return (
    <div className="player">
      <div className="player__stage">
        {c === " " ? (
          <div className="pose pose--space">
            <span>space</span>
          </div>
        ) : (
          <HandPose hands={poses[c] ?? null} accent={accent} size={260} label={`${lang} handshape for ${c}`} />
        )}
        <div className="player__letter" style={{ color: accent }}>
          {c === " " ? "␣" : c}
        </div>
        {MOTION[lang]?.includes(c) && <p className="player__hint">{c} moves: trace it in the air from this shape.</p>}
      </div>

      <div className="player__strip" aria-label="Letters">
        {chars.map((ch, k) => (
          <button key={k} className="player__cell" data-on={k === i} data-space={ch === " "} onClick={() => (setI(k), setPlaying(false))}>
            {ch === " " ? "" : ch}
          </button>
        ))}
      </div>

      <div className="player__controls">
        <button
          className="btn btn--primary"
          onClick={() => {
            if (!playing && i >= chars.length - 1) setI(0);
            setPlaying(!playing);
          }}
        >
          {playing ? "Pause" : i >= chars.length - 1 ? "Replay" : "Play"}
        </button>
        <button className="btn" onClick={() => (setI(Math.max(0, i - 1)), setPlaying(false))} aria-label="Previous letter">
          ←
        </button>
        <button className="btn" onClick={() => (setI(Math.min(chars.length - 1, i + 1)), setPlaying(false))} aria-label="Next letter">
          →
        </button>
        <div className="seg seg--small" role="group" aria-label="Speed">
          {SPEEDS.map((s, k) => (
            <button key={s.label} data-on={k === speed} onClick={() => setSpeed(k)}>
              {s.label}
            </button>
          ))}
        </div>
      </div>
      {missing.length > 0 && (
        <p className="note">
          No {lang} handshape yet for {missing.join(", ")}. Teach {missing.length === 1 ? "it" : "them"} in the Teach tab and {missing.length === 1 ? "it" : "they"}'ll show here.
        </p>
      )}
    </div>
  );
}
