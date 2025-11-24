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

  // Load model once at mount
  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
      setErrMsg(null);
      try {
        const m = await tf.loadLayersModel("/model.json"); // production: no cache-bust
        console.log("Model loaded:");
        if (m.summary) m.summary();
        if (!cancelled) setModel(m);
      } catch (err) {
        console.error("Model load failed:", err);
        setErrMsg(err.message || String(err));
      }
    }
    loadModel();
    return () => { cancelled = true; };
  }, []);

  // Start camera
  async function startCamera() {
    if (running || !model) return;
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
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
      stream.getTracks().forEach(t => t.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }
  }

  // Prediction loop
  async function predictLoop() {
    if (!running) {
      rafRef.current = null;
      return;
    }
    if (!model || !videoRef.current || videoRef.current.readyState < 2) {
      rafRef.current = requestAnimationFrame(predictLoop);
      return;
    }

    const small = smallRef.current;
    const sctx = small.getContext("2d");
    sctx.drawImage(videoRef.current, 0, 0, 28, 28);

    const preview = previewRef.current;
    const pctx = preview.getContext("2d");
    pctx.imageSmoothingEnabled = false;
    pctx.drawImage(small, 0, 0, preview.width, preview.height);

    const tensor = tf.tidy(() => {
      let t = tf.browser.fromPixels(small); // [28,28,3]
      t = t.mean(2).toFloat();              // [28,28]
      t = t.div(255.0);                     // [0,1]
      // If needed: invert or map to [-1,1] (uncomment one)
      // t = tf.sub(1, t);   // invert
      // t = t.mul(2).sub(1); // to [-1,1]
      return t.expandDims(0).expandDims(-1); // [1,28,28,1]
    });

    try {
      const logits = model.predict(tensor);
      const probs = Array.from(await logits.data());
      const top = probs
        .map((p,i)=>({i,p}))
        .sort((a,b)=>b.p-a.p)
        .slice(0,3)
        .map(x=>({ index: x.i, letter: String.fromCharCode(65 + x.i), p: x.p }));
      setPreds(top);
      tf.dispose([logits]);
    } catch (err) {
      console.error("predict error", err);
      setErrMsg(String(err));
    } finally {
      tf.dispose(tensor);
    }

    rafRef.current = requestAnimationFrame(predictLoop);
  }

  // Retry loading model
  function retryLoad() {
    setModel(null);
    setErrMsg(null);
    // re-run the effect by manually loading
    (async () => {
      try {
        const m = await tf.loadLayersModel("/model.json");
        console.log("Model reloaded:");
        if (m.summary) m.summary();
        setModel(m);
      } catch (err) {
        console.error("Retry failed:", err);
        setErrMsg(err.message || String(err));
      }
    })();
  }

  // Capture & send labeled correction (image + label)
  async function sendCorrection(label) {
    const preview = previewRef.current;
    const dataUrl = preview.toDataURL("image/png");
    try {
      await fetch('/api/log-example', {
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'x-feedback-token': process.env.NEXT_PUBLIC_FEEDBACK_TOKEN || ''},
        body: JSON.stringify({ imageDataUrl: dataUrl, label })
      });
      alert('Thanks — example submitted.');
    } catch (err) {
      console.error(err);
      alert('Failed to send example.');
    }
  }

  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: 720 }}>
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
        <div>
          <video ref={videoRef} width={320} height={240} style={{ borderRadius: 8, border: '1px solid #ddd' }} playsInline muted />
          <div style={{ marginTop: 8 }}>
            <button onClick={startCamera} disabled={running || !model} style={{ marginRight: 8 }}>Start</button>
            <button onClick={stopCamera} disabled={!running}>Stop</button>
            <button onClick={() => { if (preds && preds[0]) sendCorrection(preds[0].letter); }} style={{ marginLeft: 8 }}>Send top-1 as correction</button>
          </div>
        </div>

        <div>
          <div style={{ marginBottom: 8 }}>Preview (what the model sees)</div>
          <canvas ref={previewRef} width={140} height={140} style={{ width: 140, height: 140, imageRendering: 'pixelated', border: '1px solid #ccc', borderRadius: 6 }} />
          <div style={{ fontSize: 12, color: '#666', marginTop: 6 }}>28×28 preview (scaled)</div>
        </div>

        <div>
          <div style={{ marginBottom: 8 }}>28×28 raw</div>
          <canvas ref={smallRef} width={28} height={28} style={{ width: 56, height: 56, imageRendering: 'pixelated', border: '1px solid #eee' }} />
          <div style={{ marginTop: 12 }}>
            <strong>Top predictions</strong>
            <ol>
              {preds.length ? preds.map(p => (
                <li key={p.index}>{p.letter} — {(p.p*100).toFixed(1)}%</li>
              )) : <li>—</li>}
            </ol>
          </div>
        </div>
      </div>

      {errMsg && (
        <div style={{ color: 'red', marginTop: 12 }}>
          Error: {errMsg} <button onClick={retryLoad} style={{ marginLeft: 8 }}>Retry load</button>
        </div>
      )}
      {!model && !errMsg && <div style={{ marginTop: 12 }}>Loading model... make sure /model.json is present in <code>/public</code>.</div>}
    </div>
  );
}
