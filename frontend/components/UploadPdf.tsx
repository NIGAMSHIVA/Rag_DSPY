"use client";

import { useState } from "react";
import { uploadPdf } from "@/lib/api";

export default function UploadPdf({ onUploaded }: { onUploaded: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState("");

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setMsg("");
    try {
      await uploadPdf(file);
      setMsg("✅ PDF uploaded. RAG mode is now available.");
      onUploaded();
    } catch {
      setMsg("❌ Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-4 rounded-xl shadow w-full">
      <h2 className="font-semibold mb-2">📄 Optional: Upload PDF for RAG</h2>

      <input
        type="file"
        accept="application/pdf"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />

      <button
        onClick={handleUpload}
        disabled={!file || loading}
        className="mt-3 w-full bg-black text-white py-2 rounded"
      >
        {loading ? "Uploading..." : "Upload PDF"}
      </button>

      {msg && <p className="text-sm mt-2">{msg}</p>}
    </div>
  );
}
