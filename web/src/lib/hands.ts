import type { HandLandmarker as HandLandmarkerT } from "@mediapipe/tasks-vision";
import type { Hand } from "./features";

// WASM ships with the app (copied from the npm package) so nothing depends on a CDN
const WASM = `${import.meta.env.BASE_URL}wasm`;

let landmarkerPromise: Promise<HandLandmarkerT> | null = null;

/** MediaPipe Hand Landmarker, the same model the training data was extracted with. */
export const loadHandLandmarker = () => {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const { HandLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
      const files = await FilesetResolver.forVisionTasks(WASM);
      const opts = (delegate: "GPU" | "CPU") => ({
        baseOptions: { modelAssetPath: `${import.meta.env.BASE_URL}models/hand_landmarker.task`, delegate },
        runningMode: "VIDEO" as const,
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      try {
        return await HandLandmarker.createFromOptions(files, opts("GPU"));
      } catch {
        return HandLandmarker.createFromOptions(files, opts("CPU"));
      }
    })();
    landmarkerPromise.catch(() => {
      landmarkerPromise = null;
    });
  }
  return landmarkerPromise;
};

export interface Frame {
  /** Points in square units (pixels of the camera frame), for features */
  hands: Hand[];
  /** Normalized 0..1 points, for drawing on the video */
  raw: { x: number; y: number }[][];
  width: number;
  height: number;
}

export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export class HandCamera {
  private stream: MediaStream | null = null;
  private raf = 0;
  private stopped = false;

  constructor(
    private video: HTMLVideoElement,
    private onFrame: (f: Frame) => void,
  ) {}

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } },
      audio: false,
    });
    if (this.stopped) return this.stop();
    this.video.srcObject = this.stream;
    this.video.muted = true;
    this.video.playsInline = true;
    await this.video.play();
    const landmarker = await loadHandLandmarker();
    if (this.stopped) return;
    let last = -1;
    const loop = () => {
      if (this.stopped) return;
      const v = this.video;
      if (v.readyState >= 2 && v.currentTime !== last) {
        last = v.currentTime;
        const res = landmarker.detectForVideo(v, performance.now());
        const w = v.videoWidth;
        const h = v.videoHeight;
        const raw = (res.landmarks ?? []).map((lm) => lm.map((p) => ({ x: p.x, y: p.y })));
        this.onFrame({ hands: raw.map((lm) => lm.map((p) => [p.x * w, p.y * h] as [number, number])), raw, width: w, height: h });
      }
      this.raf = requestAnimationFrame(loop);
    };
    loop();
  }

  stop() {
    this.stopped = true;
    cancelAnimationFrame(this.raf);
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }
}

export const cameraErrorMessage = (err: unknown) => {
  const name = err instanceof DOMException ? err.name : "";
  if (name === "NotAllowedError") return "Camera access was blocked. Allow the camera for this site, then try again.";
  if (name === "NotFoundError") return "No camera was found on this device.";
  return "Couldn't start hand tracking. Check your connection and try again.";
};
