"use client";

import { useState } from "react";
import UploadPdf from "@/components/UploadPdf";
import ChatBox from "@/components/ChatBox";

export default function Home() {
  const [ragReady, setRagReady] = useState(false);

  return (
    <main className="min-h-screen bg-gray-100 flex items-center justify-center p-6">
      <div className="w-full max-w-xl space-y-4">
        <UploadPdf onUploaded={() => setRagReady(true)} />
        <ChatBox ragReady={ragReady} />
      </div>
    </main>
  );
}
