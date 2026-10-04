import { useEffect, useState } from "react";
import { CameraView } from "../components/CameraView";
import { HandPose } from "../components/HandPose";
import { Transcript } from "../components/Transcript";
import { speak, useSpeller } from "../hooks/useSpeller";
import type { Hand } from "../lib/features";
import type { Language } from "../lib/models";
import { poseFor } from "../lib/recognizer";

export function SignToText({ lang, accent }: { lang: Language; accent: string }) {
  const sp = useSpeller(lang);
  const [ref, setRef] = useState<Hand[] | null>(null);

  useEffect(() => {
    let alive = true;
    if (sp.live.letter) poseFor(lang.id, sp.live.letter).then((p) => alive && setRef(p));
    else setRef(null);
    return () => {
      alive = false;
    };
  }, [sp.live.letter, lang.id]);

  return (
    <div className="split">
      <div className="split__main">
        <CameraView onFrame={sp.handle} accent={accent} badge={sp.live.letter && sp.live.confidence >= 0.6 ? { text: sp.live.letter } : null} />
        <Tips lang={lang} />
      </div>
      <div className="split__side">
        <Transcript
          text={sp.text}
          live={sp.live}
          accent={accent}
          ready={sp.ready}
          error={sp.modelError}
          onSpace={sp.addSpace}
          onBack={sp.backspace}
          onClear={sp.clear}
          extra={
            <button className="btn btn--primary" disabled={!sp.text.trim()} onClick={() => speak(sp.text, lang.speechLang)}>
              Speak
            </button>
          }
        />
        {sp.live.letter && (
          <div className="card compare">
            <HandPose hands={ref} accent={accent} size={120} />
            <p>
              Reads as <strong>{sp.live.letter}</strong>. This is the reference {lang.id} handshape. If it isn't what you're signing, teach it your way in
              the Teach tab.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function Tips({ lang }: { lang: Language }) {
  return (
    <ul className="tips">
      <li>Keep your whole hand{lang.hands === "two" ? "s" : ""} in frame, about an arm's length away.</li>
      <li>Hold each letter still for a moment. The bar under the letter fills, then it types.</li>
      <li>To double a letter, keep holding it.</li>
    </ul>
  );
}
