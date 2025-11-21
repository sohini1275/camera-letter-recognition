// NOTE: in the free Vercel serverless environment, persistent disk is not recommended.
// Use this endpoint to receive labeled examples; forward them to a cloud storage or DB.
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Only POST" });
  try {
    const body = req.body; // expects { imageDataUrl: "...", label: "A" }
    // TODO: forward body to storage (S3, Firebase) or GitHub repo via API
    console.log("Received example:", body?.label ?? "(no label)");
    return res.status(200).json({ ok: true });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "server error" });
  }
}
