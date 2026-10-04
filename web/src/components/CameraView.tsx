import { useEffect, useRef, useState } from "react";
import type { Frame } from "../lib/hands";
import { HAND_CONNECTIONS, HandCamera, cameraErrorMessage } from "../lib/hands";

type Status = "idle" | "starting" | "live" | "error";

interface Props {
  onFrame: (f: Frame) => void;
  /** Big letter shown in the corner of the video while signing */
  badge?: { text: string; sub?: string } | null;
  accent: string;
}

const FINGER_OF: number[] = [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5];

/**
 * Mirrored selfie camera with the MediaPipe hand skeleton drawn on top,
 * so the signer can see exactly what SignBridge is reading.
 */
export function CameraView({ onFrame, badge, accent }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const camRef = useRef<HandCamera | null>(null);
  const cb = useRef(onFrame);
  cb.current = onFrame;
  const accentRef = useRef(accent);
  accentRef.current = accent;
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const [handCount, setHandCount] = useState(0);

  const start = async () => {
    if (!videoRef.current) return;
    setStatus("starting");
    setError("");
    const cam = new HandCamera(videoRef.current, (f) => {
      draw(f);
      setHandCount((n) => (n === f.hands.length ? n : f.hands.length));
      cb.current(f);
    });
    camRef.current = cam;
    try {
      await cam.start();
      setStatus("live");
    } catch (e) {
      cam.stop();
      setError(cameraErrorMessage(e));
      setStatus("error");
    }
  };

  const stop = () => {
    camRef.current?.stop();
    camRef.current = null;
    const c = canvasRef.current;
    c?.getContext("2d")?.clearRect(0, 0, c.width, c.height);
    setStatus("idle");
    setHandCount(0);
    cb.current({ hands: [], raw: [], width: 0, height: 0 });
  };

  useEffect(() => () => camRef.current?.stop(), []);

  const draw = (f: Frame) => {
    const c = canvasRef.current;
    if (!c) return;
    if (c.width !== f.width || c.height !== f.height) {
      c.width = f.width;
      c.height = f.height;
    }
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, c.width, c.height);
    const scale = Math.max(1.5, c.width / 480);
    f.raw.forEach((lm, hi) => {
      const col = hi === 0 ? accentRef.current : "#ffffff";
      const P = lm.map((p) => [p.x * c.width, p.y * c.height]);
      ctx.lineCap = "round";
      // soft halo so lines read on any background
      ctx.strokeStyle = "rgba(0,0,0,0.35)";
      ctx.lineWidth = 6 * scale;
      ctx.beginPath();
      HAND_CONNECTIONS.forEach(([a, b]) => {
        ctx.moveTo(P[a][0], P[a][1]);
        ctx.lineTo(P[b][0], P[b][1]);
      });
      ctx.stroke();
      ctx.strokeStyle = col;
      ctx.lineWidth = 3 * scale;
      ctx.beginPath();
      HAND_CONNECTIONS.forEach(([a, b]) => {
        ctx.moveTo(P[a][0], P[a][1]);
        ctx.lineTo(P[b][0], P[b][1]);
      });
      ctx.stroke();
      P.forEach(([x, y], i) => {
        const tip = i > 0 && i % 4 === 0;
        ctx.beginPath();
        ctx.arc(x, y, (tip ? 5 : 3.5) * scale, 0, Math.PI * 2);
        ctx.fillStyle = tip ? col : "#ffffff";
        ctx.fill();
        ctx.lineWidth = 1.5 * scale;
        ctx.strokeStyle = FINGER_OF[i] === 0 ? "rgba(0,0,0,.6)" : "rgba(0,0,0,.45)";
        ctx.stroke();
      });
    });
  };

  return (
    <div className={`camera camera--${status}`}>
      <video ref={videoRef} className="camera__video" playsInline muted />
      <canvas ref={canvasRef} className="camera__overlay" aria-hidden />
      {status !== "live" && (
        <div className="camera__placeholder">
          <HandGlyph />
          {status === "error" ? <p className="camera__error">{error}</p> : <p>Your camera stays on this device. Nothing is uploaded.</p>}
          <button className="btn btn--primary" onClick={start} disabled={status === "starting"}>
            {status === "starting" ? "Starting hand tracking…" : status === "error" ? "Try again" : "Turn on camera"}
          </button>
        </div>
      )}
      {status === "live" && (
        <>
          <div className="camera__chip" data-on={handCount > 0}>
            <span className="dot" />
            {handCount === 0 ? "Show your hand" : handCount === 1 ? "1 hand" : "2 hands"}
          </div>
          {badge && (
            <div className="camera__badge">
              <span>{badge.text}</span>
              {badge.sub && <small>{badge.sub}</small>}
            </div>
          )}
          <button className="camera__stop" onClick={stop}>
            Stop camera
          </button>
        </>
      )}
    </div>
  );
}

function HandGlyph() {
  return (
    <svg width="56" height="56" viewBox="0 0 56 56" fill="none" aria-hidden>
      <g stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
        <path d="M28 50 L20 41 M28 50 L28 30 M28 50 L36 41" opacity=".35" />
        <path d="M20 41 L13 33 L9 27" />
        <path d="M24 33 L22 20 L21 12" />
        <path d="M28 30 L28 16 L28 7" />
        <path d="M32 33 L34 20 L35 13" />
        <path d="M36 41 L41 30 L44 24" />
      </g>
    </svg>
  );
}
