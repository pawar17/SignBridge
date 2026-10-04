import type { Hand } from "./features";
import { poseFromHands } from "./features";
import type { BaseModel, LangId, Language } from "./models";
import { featuresFor, loadBaseModel, predictBase } from "./models";

/**
 * Signs a user teaches SignBridge, kept in this browser. A nearest-neighbour
 * classifier over these samples adapts recognition to the person signing,
 * and is the whole model for languages with no public dataset (BSL).
 */
export interface PersonalSet {
  samples: Record<string, number[][]>; // letter -> feature vectors
  poses: Record<string, Hand[]>; // letter -> a reference handshape for display
}

const KEY = (lang: LangId) => `signbridge.personal.${lang}`;
const MAX_PER_LETTER = 60;

export function loadPersonal(lang: LangId): PersonalSet {
  try {
    const raw = localStorage.getItem(KEY(lang));
    if (raw) return JSON.parse(raw);
  } catch {
    /* storage unavailable */
  }
  return { samples: {}, poses: {} };
}

export function savePersonal(lang: LangId, set: PersonalSet) {
  try {
    localStorage.setItem(KEY(lang), JSON.stringify(set));
  } catch {
    /* storage full or unavailable: keeps working for this session */
  }
}

export function addSamples(lang: Language, set: PersonalSet, letter: string, frames: Hand[][]): PersonalSet {
  const feats = frames.map((h) => featuresFor(lang, h).map((v) => Math.round(v * 1e4) / 1e4));
  const next: PersonalSet = {
    samples: { ...set.samples, [letter]: [...(set.samples[letter] ?? []), ...feats].slice(-MAX_PER_LETTER) },
    poses: { ...set.poses, [letter]: poseFromHands(frames[Math.floor(frames.length / 2)]) },
  };
  savePersonal(lang.id, next);
  return next;
}

export function clearLetter(lang: LangId, set: PersonalSet, letter: string): PersonalSet {
  const samples = { ...set.samples };
  const poses = { ...set.poses };
  delete samples[letter];
  delete poses[letter];
  const next = { samples, poses };
  savePersonal(lang, next);
  return next;
}

export const taughtLetters = (set: PersonalSet) => Object.keys(set.samples).filter((k) => set.samples[k].length > 0).sort();

/** Per-feature scale for distances: the base model's if there is one, else from the taught samples. */
function scaler(set: PersonalSet, base: BaseModel | null) {
  if (base) return { mean: base.mean, scale: base.scale };
  const all = Object.values(set.samples).flat();
  const d = all[0]?.length ?? 0;
  const mean = new Array(d).fill(0);
  const sq = new Array(d).fill(0);
  all.forEach((v) => v.forEach((x, i) => ((mean[i] += x), (sq[i] += x * x))));
  return {
    mean: mean.map((m) => m / (all.length || 1)),
    scale: sq.map((s, i) => Math.sqrt(Math.max(s / (all.length || 1) - (mean[i] / (all.length || 1)) ** 2, 1e-4))),
  };
}

function knn(set: PersonalSet, x: number[], sc: { mean: number[]; scale: number[] }, k = 7) {
  const z = (v: number[]) => v.map((t, i) => (t - sc.mean[i]) / (sc.scale[i] || 1));
  const zx = z(x);
  const dists: { label: string; d: number }[] = [];
  for (const [label, vecs] of Object.entries(set.samples)) {
    for (const v of vecs) {
      const zv = z(v);
      let d = 0;
      for (let i = 0; i < zx.length; i++) d += (zx[i] - zv[i]) ** 2;
      dists.push({ label, d: Math.sqrt(d / zx.length) });
    }
  }
  dists.sort((a, b) => a.d - b.d);
  const top = dists.slice(0, k);
  const votes = new Map<string, number>();
  top.forEach(({ label, d }) => votes.set(label, (votes.get(label) ?? 0) + 1 / (d + 0.05)));
  const total = [...votes.values()].reduce((a, b) => a + b, 0) || 1;
  votes.forEach((v, key) => votes.set(key, v / total));
  return { probs: votes, nearest: top[0]?.d ?? Infinity };
}

export interface Prediction {
  letter: string | null;
  confidence: number;
  source: "model" | "personal" | "blend" | "none";
}

export class Recognizer {
  private base: BaseModel | null = null;
  ready: Promise<void>;

  constructor(
    public lang: Language,
    public personal: PersonalSet,
  ) {
    this.ready = lang.hasBaseModel
      ? loadBaseModel(lang.id).then((m) => {
          this.base = m;
        })
      : Promise.resolve();
  }

  predict(hands: Hand[]): Prediction {
    if (!hands.length) return { letter: null, confidence: 0, source: "none" };
    if (this.lang.hands === "one" && hands.length > 1) {
      // Two hands in view for a one-handed alphabet: use whichever reads more confidently
      const a = this.predict([hands[0]]);
      const b = this.predict([hands[1]]);
      return a.confidence >= b.confidence ? a : b;
    }
    const x = featuresFor(this.lang, hands);
    const taught = taughtLetters(this.personal);
    const pm = this.base ? predictBase(this.base, x) : null;
    const kn = taught.length >= 2 ? knn(this.personal, x, scaler(this.personal, this.base)) : null;

    let probs: Map<string, number>;
    let source: Prediction["source"];
    if (pm && kn && kn.nearest < 1.1) {
      // Close to something the user taught: trust their samples most
      probs = new Map(pm);
      probs.forEach((v, key) => probs.set(key, 0.3 * v + 0.7 * (kn.probs.get(key) ?? 0)));
      kn.probs.forEach((v, key) => !probs.has(key) && probs.set(key, 0.7 * v));
      source = "blend";
    } else if (pm) {
      probs = pm;
      source = "model";
    } else if (kn && kn.nearest < 1.6) {
      probs = kn.probs;
      source = "personal";
    } else {
      return { letter: null, confidence: 0, source: "none" };
    }
    let best: string | null = null;
    let bp = 0;
    probs.forEach((v, key) => v > bp && ((bp = v), (best = key)));
    return { letter: best, confidence: bp, source };
  }
}

/** Reference handshapes: the user's own taught pose first, then the dataset's. */
let posesPromise: Promise<Record<string, Record<string, { hands: Hand[] }>>> | null = null;
export const loadPoses = () =>
  (posesPromise ??= fetch(`${import.meta.env.BASE_URL}models/poses.json`).then((r) => r.json()));

export async function poseFor(lang: LangId, letter: string): Promise<Hand[] | null> {
  const mine = loadPersonal(lang).poses[letter];
  if (mine) return mine;
  const all = await loadPoses();
  return all[lang]?.[letter]?.hands ?? null;
}
