import type { Hand } from "./features";
import { handFeatures, twoHandFeatures } from "./features";

export type LangId = "ASL" | "ISL" | "BSL";
export type FeatureKind = "one-hand-42" | "two-hand-87";

export interface Language {
  id: LangId;
  name: string;
  hands: "one" | "two";
  features: FeatureKind;
  speechLang: string;
  hasBaseModel: boolean;
  note: string;
}

export const LANGUAGES: Record<LangId, Language> = {
  ASL: {
    id: "ASL",
    name: "American Sign Language",
    hands: "one",
    features: "one-hand-42",
    speechLang: "en-US",
    hasBaseModel: true,
    note: "One-handed alphabet. Trained on 44,735 samples from two public datasets.",
  },
  ISL: {
    id: "ISL",
    name: "Indian Sign Language",
    hands: "two",
    features: "two-hand-87",
    speechLang: "en-IN",
    hasBaseModel: true,
    note: "Two-handed alphabet. Beta: the public data is small, so teaching your own signs helps a lot.",
  },
  BSL: {
    id: "BSL",
    name: "British Sign Language",
    hands: "two",
    features: "two-hand-87",
    speechLang: "en-GB",
    hasBaseModel: false,
    note: "Two-handed alphabet. No usable public dataset yet, so BSL works from signs you teach it.",
  },
};

export const featuresFor = (lang: Language, hands: Hand[]): number[] =>
  lang.features === "one-hand-42" ? handFeatures(hands[0]) : twoHandFeatures(hands);

interface Layer {
  W: number[][];
  b: number[];
}

export interface BaseModel {
  language: string;
  labels: string[];
  mean: number[];
  scale: number[];
  layers: Layer[];
  notes: Record<string, number>;
}

const cache = new Map<string, Promise<BaseModel>>();

export const loadBaseModel = (lang: LangId): Promise<BaseModel> => {
  if (!cache.has(lang)) {
    cache.set(
      lang,
      fetch(`${import.meta.env.BASE_URL}models/${lang.toLowerCase()}.json`).then((r) => {
        if (!r.ok) throw new Error(`Couldn't load the ${lang} model`);
        return r.json();
      }),
    );
  }
  return cache.get(lang)!;
};

export const standardize = (m: Pick<BaseModel, "mean" | "scale">, x: number[]) => x.map((v, i) => (v - m.mean[i]) / (m.scale[i] || 1));

/** Forward pass of the exported MLP: ReLU hidden layers, softmax output. */
export function predictBase(m: BaseModel, x: number[]): Map<string, number> {
  let h = standardize(m, x);
  m.layers.forEach((layer, li) => {
    const out = layer.b.slice();
    for (let o = 0; o < out.length; o++) {
      const w = layer.W[o];
      let s = out[o];
      for (let i = 0; i < h.length; i++) s += w[i] * h[i];
      out[o] = li < m.layers.length - 1 ? Math.max(0, s) : s;
    }
    h = out;
  });
  const mx = Math.max(...h);
  const ex = h.map((v) => Math.exp(v - mx));
  const sum = ex.reduce((a, b) => a + b, 0);
  return new Map(m.labels.map((l, i) => [l, ex[i] / sum]));
}
