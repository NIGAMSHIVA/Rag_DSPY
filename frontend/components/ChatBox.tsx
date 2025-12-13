"use client";

import { useState } from "react";
import { queryAssistant } from "@/lib/api";

export default function ChatBox({ ragReady }: { ragReady: boolean }) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("auto"); // auto | rag | web | email
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runQuery = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    // If user selected RAG but no pdf uploaded
    if (mode === "rag" && !ragReady) {
      setError("Upload a PDF first to use RAG mode.");
      setLoading(false);
      return;
    }

    try {
      const res = await queryAssistant(query, mode, ragReady);
      setResult(res);
    } catch {
      setError("Backend not reachable / request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-4 rounded-xl shadow w-full">
      <div className="flex items-center justify-between mb-2">
        <h2 className="font-semibold">🤖 AI Assistant</h2>

        <select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          className="border rounded px-2 py-1 text-sm"
        >
          <option value="auto">Auto (Orchestrator)</option>
          <option value="web">Web</option>
          <option value="email">Email</option>
          <option value="rag">PDF (RAG)</option>
        </select>
      </div>

      <textarea
        className="w-full border rounded p-2"
        rows={3}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask anything..."
      />

      <button
        onClick={runQuery}
        disabled={loading || !query.trim()}
        className="mt-3 w-full bg-black text-white py-2 rounded"
      >
        {loading ? "Running..." : "Run"}
      </button>

      {ragReady && (
        <p className="text-xs text-green-600 mt-2">
          ✅ RAG is enabled (PDF uploaded)
        </p>
      )}

      {error && <p className="text-red-500 mt-2">❌ {error}</p>}

      {result && (
        <div className="mt-4 border-t pt-3 text-sm">
          <p><b>Intent:</b> {result.intent}</p>
          <p className="mt-2 whitespace-pre-wrap">{result.answer}</p>
        </div>
      )}
    </div>
  );
}
