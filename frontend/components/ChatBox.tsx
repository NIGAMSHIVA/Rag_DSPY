"use client";

import { useState } from "react";
import { sendQuery } from "@/lib/api";
import ResponseCard from "./ResponseCard";
import Loader from "./Loader";

export default function ChatBox() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const data = await sendQuery(query);
      setResult(data);
    } catch (e) {
      setResult({ answer: "❌ Backend not reachable" });
    }

    setLoading(false);
  };

  return (
    <div className="bg-white shadow-xl rounded-xl p-6 w-[520px]">
      <h1 className="text-2xl font-bold mb-4 text-center">
        🤖 AI Assistant
      </h1>

      <textarea
        className="w-full border rounded-lg p-3 mb-3"
        rows={4}
        placeholder="Ask anything or write an email..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <button
        onClick={handleSubmit}
        className="w-full bg-black text-white py-2 rounded-lg hover:bg-gray-800"
      >
        Run
      </button>

      {loading && <Loader />}

      {result && <ResponseCard data={result} />}
    </div>
  );
}
