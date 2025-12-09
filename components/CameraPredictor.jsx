// components/CameraPredictor.jsx
import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

export default function CameraPredictor() {
  // Log TFJS version in the app (for debugging)
  useEffect(() => {
    (async () => {
      try {
        await tf.ready();
        console.log("TFJS version inside app:", tf.version.tfjs);
      } catch (e) {
        console.error("Error checking TFJS version:", e);
      }
    })();
  }, []);

  const videoRef = useRef(null);
  const previewRef = useRef(null);
  const smallRef = useRef(null);
  const rafRef = useRef(null);
  const drewOnceRef = useRef(false); // to log only once when we first draw

  const [model, setModel] = useState(null);
  const [running, setRunning] = useState(false);
  const [preds, setPreds] = useState([]);
  const [errMsg, setErrMsg] = useState(null);
  const [stream, setStream] = useState(null);

  // Try to load model (even if it fails, preview will still work)
  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
      setErrMsg(null);
      try {
        const m = await tf.loadLayersModel("/model.json");
        console.log("Model loaded (graph):", m);
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
        } catch (e) {
          /* ignore */
        }
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Start camera – independent of model
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
      drewOnceRef.current = false;
      if (!rafRef.current) {
        rafRef.current = requestAnimationFrame(previewLoop);
      }
    } catch (e) {
      console.error("Camera start error:", e);
      setErrMsg(String(e));
    }
  }

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
      } catch (e) {
        /* ignore */
      }
    }
  }

  // 🔁 Preview loop: just draws camera → 28×28 → scaled preview
  function previewLoop() {
    try {
      if (!running) {
        rafRef.current = null;
        return;
      }

      const video = videoRef.current;
      const small = smallRef.current;
      const preview = previewRef.current;

      if (!video || !small || !preview) {
        // Refs not attached yet, try again next frame
        rafRef.current = requestAnimationFrame(previewLoop);
        return;
      }

      if (video.readyState < 2) {
        // Video not ready yet
        rafRef.current = requestAnimationFrame(previewLoop);
        return;
      }

      const sctx = small.getContext("2d");
      const pctx = preview.getContext("2d");

      if (!sctx || !pctx) {
        console.error("Canvas context missing");
        rafRef.current = requestAnimationFrame(previewLoop);
        return;
      }

      // Draw to 28×28
      sctx.clearRect(0, 0, 28, 28);
      sctx.drawImage(video, 0, 0, 28, 28);

      // Draw scaled version
      pctx.imageSmoothingEnabled = false;
      pctx.clearRect(0, 0, preview.width, preview.height);
      pctx.drawImage(small, 0, 0, preview.width, preview.height);

      if (!drewOnceRef.current) {
        console.log("✅ First preview frame drawn");
        drewOnceRef.current = true;
      }
    } catch (err) {
      console.error("previewLoop error:", err);
    }

    rafRef.current = requestAnimationFrame(previewLoop);
  }

  // (Prediction-related functions left for later)
  function retryLoad() {
    setModel(null);
    setErrMsg(null);
    (async () => {
      try {
        const m = await tf.loadLayersModel("/model.json");
        console.log("Model reloaded:");
        setModel(m);
      } catch (err) {
        console.error("Retry failed:", err);
        setErrMsg(err.message || String(err));
      }
    })();
  }

  async function sendCorrection(label) {
    const preview = previewRef.current;
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
    <div style={{ fontFamily: "sans-serif", maxWidth: 720 }}>
      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        <div>
          <video
            ref={videoRef}
            width={320}
            height={240}
            style={{ borderRadius: 8, border: "1px solid #ddd" }}
            playsInline
            muted
          />
          <div style={{ marginTop: 8 }}>
            <button
              onClick={startCamera}
              disabled={running}
              style={{ marginRight: 8 }}
            >
              Start
            </button>
            <button onClick={stopCamera} disabled={!running}>
              Stop
            </button>
            <button
              onClick={() => {
                if (preds && preds[0]) sendCorrection(preds[0].letter);
              }}
              style={{ marginLeft: 8 }}
            >
              Send top-1 as correction
            </button>
          </div>
          {!model && (
            <div style={{ marginTop: 8, fontSize: 12, color: "#666" }}>
              Model is not loaded in the browser yet (TFJS/Keras compatibility
              issue). Camera preview still matches the 28×28 input used for
              training in Python.
            </div>
          )}
        </div>

        <div>
          <div style={{ marginBottom: 8 }}>Preview (what the model sees)</div>
          <canvas
            ref={previewRef}
            width={140}
            height={140}
            style={{
              width: 140,
              height: 140,
              imageRendering: "pixelated",
              border: "1px solid #ccc",
              borderRadius: 6,
              background: "#000",
            }}
          />
          <div style={{ fontSize: 12, color: "#666", marginTop: 6 }}>
            28×28 preview (scaled)
          </div>
        </div>

        <div>
          <div style={{ marginBottom: 8 }}>28×28 raw</div>
          <canvas
            ref={smallRef}
            width={28}
            height={28}
            style={{
              width: 56,
              height: 56,
              imageRendering: "pixelated",
              border: "1px solid #eee",
              background: "#000",
            }}
          />
          <div style={{ marginTop: 12 }}>
            <strong>Top predictions</strong>
            <ol>
              {preds.length ? (
                preds.map((p) => (
                  <li key={p.index}>
                    {p.letter} — {(p.p * 100).toFixed(1)}%
                  </li>
                ))
              ) : (
                <li>—</li>
              )}
            </ol>
          </div>
        </div>
      </div>

      {errMsg && (
        <div style={{ color: "red", marginTop: 12 }}>
          Error loading model in browser: {errMsg}
          <button onClick={retryLoad} style={{ marginLeft: 8 }}>
            Retry load
          </button>
        </div>
      )}
      {!model && !errMsg && (
        <div style={{ marginTop: 12 }}>
          Loading model... make sure /model.json is present in{" "}
          <code>/public</code>. (For the review, you can show predictions from
          your Python notebook.)
        </div>
      )}
    </div>
  );
}
