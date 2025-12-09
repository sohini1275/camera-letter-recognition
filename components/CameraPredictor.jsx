// components/CameraPredictor.jsx
import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";

export default function CameraPredictor() {
  // Log TFJS version in the app
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

  const [model, setModel] = useState(null);
  const [running, setRunning] = useState(false);
  const [preds, setPreds] = useState([]);
  const [errMsg, setErrMsg] = useState(null);
  const [stream, setStream] = useState(null);

  // Try to load model once at mount (if it fails, we still show preview)
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

    // cleanup on unmount
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

  // Start camera – allow even if model isn't loaded yet
  async function startCamera() {
    if (running) return; // only block if already running
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
      if (!rafRef.current) {
        rafRef.current = requestAnimationFrame(previewLoop);
      }
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
      } catch (e) {
        /* ignore */
      }
    }
  }

  // 🔁 Preview loop (NO TF / NO PREDICTION, just drawing)
  async function previewLoop() {
    if (!running) {
      rafRef.current = null;
      return;
    }

    const video = videoRef.current;
    const small = smallRef.current;
    const preview = previewRef.current;

    // Need all elements and video must be ready
    if (!video || !small || !preview || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(previewLoop);
      return;
    }

    // Draw camera frame to 28x28 canvas
    const sctx = small.getContext("2d");
    sctx.drawImage(video, 0, 0, 28, 28);

    // Draw scaled-up version to preview canvas
    const pctx = preview.getContext("2d");
    pctx.imageSmoothingEnabled = false;
    pctx.drawImage(small, 0, 0, preview.width, preview.height);

    // (Optional) Later you can re-add TF prediction here

    rafRef.current = requestAnimationFrame(previewLoop);
  }

  // Retry loading model (for after you fix compatibility)
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
            {/* Start button: only disabled while camera is already running */}
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
              issue). Camera preview below matches the 28×28 input used for
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
