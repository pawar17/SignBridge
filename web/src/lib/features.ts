/**
 * Handshape features. Must stay identical to hand_features / two_hand_features
 * in scripts/training/train_browser_models.py, or the trained models see
 * different numbers live than they saw in training.
 */
export type Pt = [number, number];
export type Hand = Pt[]; // 21 points, square units (x and y on the same scale)

const CHAINS = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8],
  [0, 9, 10, 11, 12],
  [0, 13, 14, 15, 16],
  [0, 17, 18, 19, 20],
];
const TIPS = [4, 8, 12, 16, 20];

const sub = (a: Pt, b: Pt): Pt => [a[0] - b[0], a[1] - b[1]];
const norm = (a: Pt) => Math.hypot(a[0], a[1]);
const dot = (a: Pt, b: Pt) => a[0] * b[0] + a[1] * b[1];

export function handFeatures(P: Hand): number[] {
  const rel = P.map((p) => sub(p, P[0]));
  const palm = norm(rel[9]) + 1e-6;
  const f: number[] = [];
  for (const ch of CHAINS) {
    for (let i = 1; i < 4; i++) {
      const a = sub(P[ch[i - 1]], P[ch[i]]);
      const b = sub(P[ch[i + 1]], P[ch[i]]);
      f.push(dot(a, b) / (norm(a) * norm(b) + 1e-6));
    }
  }
  for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) f.push(norm(sub(P[TIPS[i]], P[TIPS[j]])) / palm);
  for (const t of TIPS) f.push(norm(sub(P[t], P[0])) / palm);
  const ang = Math.atan2(rel[9][0], -rel[9][1]);
  f.push(Math.sin(ang), Math.cos(ang));
  const c = Math.cos(-ang);
  const s = Math.sin(-ang);
  for (const t of TIPS) {
    const [x, y] = rel[t];
    f.push((c * x - s * y) / palm, (s * x + c * y) / palm);
  }
  return f;
}

export function twoHandFeatures(hands: Hand[]): number[] {
  const hs = [...hands].sort((a, b) => a[0][0] - b[0][0]).slice(0, 2);
  const h1 = handFeatures(hs[0]);
  if (hs.length === 1) return [...h1, ...new Array(42).fill(0), 0, 0, 0];
  const h2 = handFeatures(hs[1]);
  const palm = (norm(sub(hs[0][9], hs[0][0])) + norm(sub(hs[1][9], hs[1][0]))) / 2 + 1e-6;
  const clip = (v: number) => Math.max(-6, Math.min(6, v));
  return [...h1, ...h2, clip((hs[1][0][0] - hs[0][0][0]) / palm), clip((hs[1][0][1] - hs[0][0][1]) / palm), 1];
}

/** Wrist-relative, palm-scaled points for drawing a reference handshape. */
export function poseFromHands(hands: Hand[]): Hand[] {
  if (hands.length === 1) {
    const P = hands[0];
    const palm = norm(sub(P[9], P[0])) + 1e-6;
    return [P.map((p) => [(p[0] - P[0][0]) / palm, (p[1] - P[0][1]) / palm] as Pt)];
  }
  const hs = [...hands].sort((a, b) => a[0][0] - b[0][0]).slice(0, 2);
  const palm = hs.reduce((s, h) => s + norm(sub(h[9], h[0])), 0) / hs.length + 1e-6;
  const cx = hs.reduce((s, h) => s + h[0][0], 0) / hs.length;
  const cy = hs.reduce((s, h) => s + h[0][1], 0) / hs.length;
  return hs.map((h) => h.map((p) => [(p[0] - cx) / palm, (p[1] - cy) / palm] as Pt));
}
