import type { Hand } from "../lib/features";
import { HAND_CONNECTIONS } from "../lib/hands";

interface Props {
  hands: Hand[] | null;
  accent: string;
  size?: number;
  label?: string;
}

const PALM = [0, 1, 5, 9, 13, 17];

/** A reference handshape drawn from real landmark data (palm-normalized points). */
export function HandPose({ hands, accent, size = 220, label }: Props) {
  if (!hands || !hands.length) {
    return (
      <div className="pose pose--empty" style={{ width: size, height: size }}>
        <span>Not taught yet</span>
      </div>
    );
  }
  const pts = hands.flat();
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY, 1e-3);
  const pad = span * 0.14;
  const vb = [minX - pad - (span - (maxX - minX)) / 2, minY - pad - (span - (maxY - minY)) / 2, span + pad * 2, span + pad * 2];
  const sw = (span + pad * 2) / 40;

  return (
    <svg className="pose" width={size} height={size} viewBox={vb.join(" ")} role="img" aria-label={label ?? "Handshape"}>
      {hands.map((P, hi) => (
        <g key={hi}>
          <polygon points={PALM.map((i) => P[i].join(",")).join(" ")} fill={accent} opacity={0.12} />
          {HAND_CONNECTIONS.map(([a, b], i) => (
            <line
              key={i}
              x1={P[a][0]}
              y1={P[a][1]}
              x2={P[b][0]}
              y2={P[b][1]}
              stroke="var(--ink)"
              strokeWidth={sw * 1.15}
              strokeLinecap="round"
              opacity={a === 0 || (a === 13 && b === 17) || (a === 5 && b === 9) || (a === 9 && b === 13) ? 0.35 : 0.9}
            />
          ))}
          {P.map(([x, y], i) => {
            const tip = i > 0 && i % 4 === 0;
            return <circle key={i} cx={x} cy={y} r={tip ? sw * 1.5 : sw * 0.9} fill={tip ? accent : "var(--paper)"} stroke="var(--ink)" strokeWidth={sw * 0.5} />;
          })}
        </g>
      ))}
    </svg>
  );
}
