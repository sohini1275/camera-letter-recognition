import dynamic from "next/dynamic";
import Head from "next/head";

const CameraPredictor = dynamic(() => import("../components/CameraPredictor"), { ssr: false });

export default function Home() {
  return (
    <>
      <Head>
        <title>Camera Letter Recognition</title>
        <meta name="viewport" content="width=device-width,initial-scale=1" />
      </Head>
      <main style={{ padding: 20, fontFamily: "sans-serif" }}>
        <h1>Camera Letter Recognition</h1>
        <p>Allow camera access when prompted. Predictions are done client-side with TensorFlow.js.</p>
        <CameraPredictor />
      </main>
    </>
  );
}
