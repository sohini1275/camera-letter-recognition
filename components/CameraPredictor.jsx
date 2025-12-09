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

  // Load model once at mount
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
      // stop camera if running
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
      videoRef.current.srcObject = s;
      setStream(s);
      await videoRef.current.play();
      setRunning(true);
      if (!rafRef.current) {
        rafRef.current = requestAnimationFrame(predictLoop);
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

  // Prediction loop
  async function predictLoop() {
    if (!running) {
      rafRef.current = null;
      return;
    }
    if (!model || !videoRef.current || videoRef.current.readyState < 2) {
      // camera can be running even if model isn't ready yet
      rafRef.current = requestAnimationFrame(predictLoop);
      return;
    }

    // draw to small canvas (28x28)
    const small = smallRef.current;
    const sctx = small.getContext("2d");
    sctx.drawImage(videoRef.current, 0, 0, 28, 28);

    // draw preview (scaled)
    const preview = previewRef.current;
    const pctx = preview.getContext("2d");
    pctx.imageSmoothingEnabled = false;
    pctx.drawImage(small, 0, 0, preview.width, preview.height);

    // make input tensor inside tidy
    const tensor = tf.tidy(() => {
      let t = tf.browser.fromPixels(small); // [28,28,3]
      t = tf.image.rgbToGrayscale(t).squeeze(); // [28,28]
      t = t.div(255.0); // [0,1]
      // If needed: invert or map to [-1,1]
      // t = tf.sub(1, t);   // invert
      // t = t.mul(2).sub(1); // to [-1,1]
      return t.expandDims(0).expandDims(-1); // [1,28,28,1]
    });

    let out = null;
    try {
      // model.predict can return Tensor, Array<Tensor>, or {outputName: Tensor}
      out = model.predict(tensor);

      let tensorOut;
      if (Array.isArray(out)) {
        tensorOut = out[0];
      } else if (out && typeof out === "object" && !("shape" in out)) {
        const keys = Object.keys(out);
        tensorOut = out[keys[0]];
      } else {
        tensorOut = out;
      }

      const probsArr = Array.from(await tensorOut.data());

      const top = probsArr
        .map((p, i) => ({ i, p }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 3)
        .map((x) => ({
          index: x.i,
          letter: String.fromCharCode(65 + x.i),
          p: x.p,
        }));

      console.log("DEBUG top before setPreds:", top);
      setPreds(() => [...top]);
      console.log("DEBUG setPreds after force-update");

      if (Array.isArray(out)) {
        out.forEach((t) => tf.dispose(t));
      } else if (out && typeof out === "object" && !("shape" in out)) {
        Object.values(out).forEach((t) => tf.dispose(t));
      } else {
        tf.dispose(out);
      }
    } catch (err) {
      console.error("predict error", err);
      setErrMsg(String(err));
      try {
        if (Array.isArray(out)) out.forEach((t) => tf.dispose(t));
        else if (out && typeof out === "object" && !("shape" in out))
          Object.values(out).forEach((t) => tf.dispose(t));
        else tf.dispose(out);
      } catch (e) {
        /* ignore */
      }
    } finally {
      try {
        tf.dispose(tensor);
      } catch (e) {
        /* ignore */
      }
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
          {!model && !errMsg && (
            <div style={{ marginTop: 8, fontSize: 12, color: "#666" }}>
              Model still loading... camera can start, predictions appear when
              the model is ready.
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
          Error: {errMsg}
          <button onClick={retryLoad} style={{ marginLeft: 8 }}>
            Retry load
          </button>
        </div>
      )}
      {!model && !errMsg && (
        <div style={{ marginTop: 12 }}>
          Loading model... make sure /model.json is present in{" "}
          <code>/public</code>.
        </div>
      )}
    </div>
  );
}
