// components/CameraPredictor.jsx
import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

export default function CameraPredictor() {
  const videoRef = useRef(null);
  const previewRef = useRef(null);
  const smallRef = useRef(null);
  const rafRef = useRef(null);

  const [model, setModel] = useState(null);
  const [running, setRunning] = useState(false);
  const [preds, setPreds] = useState([]);
  const [errMsg, setErrMsg] = useState(null);
  const [stream, setStream] = useState(null);
  const [invertColors, setInvertColors] = useState(true);

  // Load model once at mount
  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
      setErrMsg(null);
      try {
        await tf.ready();
        console.log("TFJS version:", tf.version.tfjs);
        const m = await tf.loadLayersModel("/model.json");
        console.log("Model loaded successfully:", m);
        if (!cancelled) setModel(m);
      } catch (err) {
        console.error("Model load failed:", err);
        setErrMsg(err.message || String(err));
      }
    }
    loadModel();

    return () => {
      cancelled = true;
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
      if (videoRef.current) {
        try {
          videoRef.current.pause();
          videoRef.current.srcObject = null;
        } catch (e) { /* ignore */ }
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Start camera - works independently of model
  async function startCamera() {
    if (running) return;
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      if (!videoRef.current) return;
      videoRef.current.srcObject = s;
      setStream(s);
      await videoRef.current.play();
      setRunning(true);
      if (!rafRef.current) rafRef.current = requestAnimationFrame(predictLoop);
    } catch (e) {
      console.error("Camera start error:", e);
      setErrMsg(String(e));
    }
  }

  // Stop camera
  function stopCamera() {
    setRunning(false);
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      setStream(null);
    }
    if (videoRef.current) {
      try {
        videoRef.current.pause();
        videoRef.current.srcObject = null;
      } catch (e) { /* ignore */ }
    }
  }

  // Prediction loop - works even without model (shows preview)
  async function predictLoop() {
    if (!running) {
      rafRef.current = null;
      return;
    }

    // Check video readiness
    if (!videoRef.current || videoRef.current.readyState < 2) {
      rafRef.current = requestAnimationFrame(predictLoop);
      return;
    }

    try {
      // Always draw preview (works without model)
      const small = smallRef.current;
      const preview = previewRef.current;

      if (!small || !preview) {
        rafRef.current = requestAnimationFrame(predictLoop);
        return;
      }

      const sctx = small.getContext("2d");
      const pctx = preview.getContext("2d");

      if (!sctx || !pctx) {
        rafRef.current = requestAnimationFrame(predictLoop);
        return;
      }

      // Draw to 28x28 canvas
      sctx.clearRect(0, 0, 28, 28);
      sctx.drawImage(videoRef.current, 0, 0, 28, 28);

      // Draw preview (scaled)
      pctx.imageSmoothingEnabled = false;
      pctx.clearRect(0, 0, preview.width, preview.height);
      pctx.drawImage(small, 0, 0, preview.width, preview.height);

      // Only predict if model is loaded
      if (model) {
        const tensor = tf.tidy(() => {
          let t = tf.browser.fromPixels(small);
          t = tf.image.rgbToGrayscale(t).squeeze();
          t = t.div(255.0);
          
          if (invertColors) {
            t = tf.sub(1.0, t);
          }
          
          return t.expandDims(0).expandDims(-1);
        });

        const out = model.predict(tensor);
        const probsArr = Array.from(await out.data());

        const top = probsArr
          .map((p, i) => ({ i, p }))
          .sort((a, b) => b.p - a.p)
          .slice(0, 3)
          .map((x) => ({
            index: x.i,
            letter: String.fromCharCode(65 + x.i),
            p: x.p,
          }));

        setPreds(top);
        tf.dispose([tensor, out]);
      }
    } catch (err) {
      console.error("Predict error:", err);
      setErrMsg(String(err));
    }

    rafRef.current = requestAnimationFrame(predictLoop);
  }

  // Retry loading model
  function retryLoad() {
    setModel(null);
    setErrMsg(null);
    (async () => {
      try {
        const m = await tf.loadLayersModel("/model.json");
        console.log("Model reloaded:", m);
        setModel(m);
      } catch (err) {
        console.error("Retry failed:", err);
        setErrMsg(err.message || String(err));
      }
    })();
  }

  // Send correction feedback
  async function sendCorrection(label) {
    const preview = previewRef.current;
    if (!preview) return;
    const dataUrl = preview.toDataURL("image/png");
    try {
      await fetch("/api/log-example", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-feedback-token": process.env.NEXT_PUBLIC_FEEDBACK_TOKEN || "",
        },
        body: JSON.stringify({ imageDataUrl: dataUrl, label }),
      });
      alert("Thanks — example submitted.");
    } catch (err) {
      console.error(err);
      alert("Failed to send example.");
    }
  }

  return (
    <div style={{ fontFamily: "sans-serif", maxWidth: 800, margin: "0 auto" }}>
      <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
        {/* Camera Section */}
        <div style={{ flex: "1 1 320px" }}>
          <video
            ref={videoRef}
            width={320}
            height={240}
            style={{ 
              borderRadius: 8, 
              border: "2px solid #ddd",
              width: "100%",
              maxWidth: 320,
              height: "auto"
            }}
            playsInline
            muted
          />
          
          {/* Control Buttons */}
          <div style={{ marginTop: 12, display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button
              onClick={startCamera}
              disabled={running}
              style={{
                padding: "8px 16px",
                background: running ? "#ccc" : "#2196F3",
                color: "white",
                border: "none",
                borderRadius: 6,
                cursor: running ? "not-allowed" : "pointer",
                fontWeight: 600,
                fontSize: 14
              }}
            >
              {running ? "Camera Running..." : "Start Camera"}
            </button>
            <button
              onClick={stopCamera}
              disabled={!running}
              style={{
                padding: "8px 16px",
                background: !running ? "#ccc" : "#f44336",
                color: "white",
                border: "none",
                borderRadius: 6,
                cursor: !running ? "not-allowed" : "pointer",
                fontWeight: 600,
                fontSize: 14
              }}
            >
              Stop
            </button>
          </div>
          
          {/* Inversion Toggle */}
          <div style={{ marginTop: 12 }}>
            <button
              onClick={() => setInvertColors(!invertColors)}
              style={{
                padding: "8px 16px",
                background: invertColors ? "#4CAF50" : "#ddd",
                color: invertColors ? "white" : "#333",
                border: "none",
                borderRadius: 6,
                cursor: "pointer",
                fontWeight: 600,
                fontSize: 14,
                width: "100%",
                maxWidth: 320
              }}
            >
              {invertColors ? "✓ Inverted (Black on White)" : "Normal (White on Black)"}
            </button>
            <div style={{ fontSize: 11, color: "#666", marginTop: 6, fontStyle: "italic" }}>
              Toggle for black pen on white paper vs. white on black
            </div>
          </div>

          {/* Correction Button */}
          {model && (
            <div style={{ marginTop: 12 }}>
              <button
                onClick={() => {
                  if (preds && preds[0]) sendCorrection(preds[0].letter);
                }}
                disabled={!preds[0]}
                style={{
                  padding: "6px 12px",
                  fontSize: 12,
                  background: !preds[0] ? "#eee" : "#FF9800",
                  color: !preds[0] ? "#999" : "white",
                  border: "none",
                  borderRadius: 4,
                  cursor: !preds[0] ? "not-allowed" : "pointer"
                }}
              >
                Send Correction Feedback
              </button>
            </div>
          )}
        </div>

        {/* Preview Section */}
        <div style={{ flex: "0 0 auto" }}>
          <div style={{ marginBottom: 8, fontWeight: 600, fontSize: 14 }}>
            Preview (28×28)
          </div>
          <canvas
            ref={previewRef}
            width={140}
            height={140}
            style={{
              width: 140,
              height: 140,
              imageRendering: "pixelated",
              border: "2px solid #ccc",
              borderRadius: 6,
              background: "#f5f5f5",
            }}
          />
          <div style={{ fontSize: 11, color: "#666", marginTop: 6 }}>
            What the model sees (scaled)
          </div>
        </div>

        {/* Raw Canvas + Predictions */}
        <div style={{ flex: "0 0 auto" }}>
          <div style={{ marginBottom: 8, fontWeight: 600, fontSize: 14 }}>
            Raw Input
          </div>
          <canvas
            ref={smallRef}
            width={28}
            height={28}
            style={{
              width: 56,
              height: 56,
              imageRendering: "pixelated",
              border: "1px solid #ddd",
              background: "#f5f5f5",
              borderRadius: 4
            }}
          />
          
          {/* Predictions */}
          <div style={{ marginTop: 16 }}>
            <strong style={{ fontSize: 14 }}>Top Predictions:</strong>
            <ol style={{ paddingLeft: 20, margin: "8px 0", fontSize: 14 }}>
              {model ? (
                preds.length ? (
                  preds.map((p) => (
                    <li key={p.index} style={{ marginBottom: 4 }}>
                      <strong style={{ fontSize: 18, color: "#2196F3" }}>{p.letter}</strong>
                      {" — "}
                      <span style={{ color: "#666" }}>{(p.p * 100).toFixed(1)}%</span>
                    </li>
                  ))
                ) : (
                  <li style={{ color: "#999" }}>Waiting for predictions...</li>
                )
              ) : (
                <li style={{ color: "#999" }}>Model loading...</li>
              )}
            </ol>
          </div>
        </div>
      </div>

      {/* Error Message */}
      {errMsg && (
        <div
          style={{
            color: "#d32f2f",
            marginTop: 16,
            padding: 12,
            background: "#ffebee",
            borderRadius: 6,
            border: "1px solid #ef5350"
          }}
        >
          <strong>Error:</strong> {errMsg}
          <button
            onClick={retryLoad}
            style={{
              marginLeft: 12,
              padding: "4px 12px",
              background: "#f44336",
              color: "white",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 12
            }}
          >
            Retry Load
          </button>
        </div>
      )}

      {/* Loading Message */}
      {!model && !errMsg && (
        <div
          style={{
            marginTop: 16,
            padding: 12,
            background: "#e3f2fd",
            borderRadius: 6,
            color: "#1976d2",
            fontSize: 13
          }}
        >
          📥 Loading model... Make sure <code style={{ background: "#fff", padding: "2px 6px", borderRadius: 3 }}>/public/model.json</code> and{" "}
          <code style={{ background: "#fff", padding: "2px 6px", borderRadius: 3 }}>/public/group1-shard1of1.bin</code> are present.
        </div>
      )}
    </div>
  );
}
