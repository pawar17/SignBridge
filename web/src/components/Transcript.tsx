import type { Live } from "../hooks/useSpeller";

interface Props {
  text: string;
  live: Live;
  accent: string;
  ready: boolean;
  error: string;
  onSpace: () => void;
  onBack: () => void;
  onClear: () => void;
  extra?: React.ReactNode;
}

const SOURCE: Record<Live["source"], string> = {
  model: "base model",
  personal: "your taught signs",
  blend: "model + your signs",
  none: "",
};

export function Transcript({ text, live, accent, ready, error, onSpace, onBack, onClear, extra }: Props) {
  return (
    <div className="transcript">
      <div className="reading">
        <div className="reading__letter" style={{ color: live.letter ? accent : undefined }}>
          {live.letter ?? "–"}
        </div>
        <div className="reading__meta">
          <div className="meter" aria-label="Confidence">
            <div className="meter__fill" style={{ width: `${Math.round(live.confidence * 100)}%`, background: live.confidence >= 0.6 ? accent : "var(--muted)" }} />
          </div>
          <div className="reading__line">
            <span>{live.letter ? `${Math.round(live.confidence * 100)}% sure` : error || (ready ? "Waiting for a handshape" : "Loading model…")}</span>
            {live.letter && <span className="muted">{SOURCE[live.source]}</span>}
          </div>
          <div className="hold" aria-hidden>
            <div className="hold__fill" style={{ width: `${live.progress * 100}%`, background: accent }} />
          </div>
        </div>
      </div>

      <div className="output" aria-live="polite">
        {text ? (
          <>
            {text}
            <span className="caret" style={{ background: accent }} />
          </>
        ) : (
          <span className="muted">Hold a letter steady to type it. Drop your hand for a space.</span>
        )}
      </div>

      <div className="row">
        <button className="btn" onClick={onSpace}>
          Space
        </button>
        <button className="btn" onClick={onBack} disabled={!text}>
          Delete
        </button>
        <button className="btn" onClick={onClear} disabled={!text}>
          Clear
        </button>
        <span className="grow" />
        {extra}
      </div>
    </div>
  );
}
