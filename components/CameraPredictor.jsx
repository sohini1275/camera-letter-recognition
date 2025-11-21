import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

export default function CameraPredictor() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [pred, setPred] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        // model.json should be at /model.json in public/
        const m = await tf.loadLayersModel("/model.json");
        if (!cancelled) {
          setModel(m);
          setLoading(false);
        }
      } catch (err) {
        console.error("Failed to load model:", err);
        setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      } catch (err) {
        console.error("Camera error:", err);
      }
    }
    startCamera();
  }, []);

  useEffect(() => {
    let raf;
    const loop = async () => {
      if (!model || !videoRef.current || videoRef.current.readyState < 2) {
        raf = requestAnimationFrame(loop);
        return;
      }

      const W = 28, H = 28;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      // draw frame scaled to 28x28 (this is the same size your model expects)
      ctx.drawImage(videoRef.current, 0, 0, W, H);

      const tensor = tf.tidy(() => {
        let t = tf.browser.fromPixels(canvas); // [H,W,3]
        t = t.mean(2).toFloat();               // grayscale [H,W]
        t = t.div(255.0);                      // normalize to [0,1]
        // If your training inverted colors, use: t = tf.sub(1, t);
        return t.expandDims(0).expandDims(-1); // [1,28,28,1]
      });

      const logits = model.predict(tensor); // [1,26]
      const probs = Array.from(await logits.data());
      const top = probs.indexOf(Math.max(...probs));
      const confidence = probs[top];
      setPred({ letter: String.fromCharCode(65 + top), confidence: (confidence).toFixed(3) });

      tf.dispose([tensor, logits]);

      raf = requestAnimationFrame(loop);
    };

    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [model]);

  return (
    <div>
      <video ref={videoRef} style={{ display: "none" }} playsInline muted />
      <canvas ref={canvasRef} width={28} height={28} style={{ display: "none" }} />
      <div style={{ marginTop: 12 }}>
        <strong>Status:</strong> {loading ? "Loading model..." : "Model loaded."}
        <div style={{ marginTop: 8 }}>
          <h2>Prediction: {pred ? `${pred.letter} (${pred.confidence})` : "—"}</h2>
          <p style={{ fontSize: 14, color: "#555" }}>If predictions are wrong, add a Feedback button to save image + label for retraining.</p>
        </div>
      </div>
    </div>
  );
}
