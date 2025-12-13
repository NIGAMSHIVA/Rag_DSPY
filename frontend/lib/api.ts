const BASE_URL = "http://localhost:8000";

export async function uploadPdf(file: File) {
  const formData = new FormData();
  formData.append("pdf", file);

  const res = await fetch(`${BASE_URL}/upload-pdf`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error("PDF upload failed");
  return res.json();
}

export async function queryAssistant(query: string, mode: string, ragEnabled: boolean) {
  const res = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      mode,         // auto | rag | web | email
      rag_enabled: ragEnabled,
    }),
  });

  if (!res.ok) throw new Error("Query failed");
  return res.json();
}
