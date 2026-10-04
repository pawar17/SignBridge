import { useEffect, useState } from "react";
import { CameraView } from "../components/CameraView";
import { HandPose } from "../components/HandPose";
import { SpellPlayer } from "../components/SpellPlayer";
import { Transcript } from "../components/Transcript";
import { useSpeller } from "../hooks/useSpeller";
import type { Hand } from "../lib/features";
import type { LangId, Language } from "../lib/models";
import { LANGUAGES } from "../lib/models";
import { poseFor } from "../lib/recognizer";
import { ACCENT } from "../theme";

export function SignToSign({ lang, accent }: { lang: Language; accent: string }) {
  const others = (Object.keys(LANGUAGES) as LangId[]).filter((l) => l !== lang.id);
  const [to, setTo] = useState<LangId>(others[0]);
  useEffect(() => {
    if (to === lang.id) setTo(others[0]);
  }, [lang.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const target = to === lang.id ? others[0] : to;
  const sp = useSpeller(lang);
  const [pose, setPose] = useState<Hand[] | null>(null);
  const shown = sp.live.letter && sp.live.confidence >= 0.6 ? sp.live.letter : sp.text.trim().slice(-1) || null;

  useEffect(() => {
    let alive = true;
    if (shown) poseFor(target, shown).then((p) => alive && setPose(p));
    else setPose(null);
    return () => {
      alive = false;
    };
  }, [shown, target]);

  return (
    <div className="split">
      <div className="split__main">
        <CameraView onFrame={sp.handle} accent={accent} badge={sp.live.letter && sp.live.confidence >= 0.6 ? { text: sp.live.letter, sub: lang.id } : null} />
        <Transcript
          text={sp.text}
          live={sp.live}
          accent={accent}
          ready={sp.ready}
          error={sp.modelError}
          onSpace={sp.addSpace}
          onBack={sp.backspace}
          onClear={sp.clear}
        />
      </div>
      <div className="split__side">
        <div className="card">
          <div className="bridge">
            <span className="tag" style={{ background: accent }}>
              {lang.id}
            </span>
            <span className="bridge__arrow">→</span>
            <div className="seg" role="group" aria-label="Translate into">
              {others.map((l) => (
                <button key={l} data-on={l === target} onClick={() => setTo(l)} style={l === target ? { background: ACCENT[l], borderColor: ACCENT[l] } : undefined}>
                  {l}
                </button>
              ))}
            </div>
          </div>
          <div className="bridge__now">
            <HandPose hands={shown ? pose : null} accent={ACCENT[target]} size={200} label={shown ? `${target} handshape for ${shown}` : undefined} />
            <div>
              <div className="bridge__letter" style={{ color: ACCENT[target] }}>
                {shown ?? "–"}
              </div>
              <p className="muted small">{shown ? `${shown} in ${LANGUAGES[target].name}` : `Sign in ${lang.id} and the matching ${target} handshape appears here.`}</p>
            </div>
          </div>
        </div>
        <div className="card">
          <h3>Whole message in {target}</h3>
          <SpellPlayer text={sp.text} lang={target} accent={ACCENT[target]} autoplay={false} />
        </div>
      </div>
    </div>
  );
}
