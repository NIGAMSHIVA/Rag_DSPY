import "./global.css";

export const metadata = {
  title: "AI Orchestrator",
  description: "RAG | Web | Email Assistant",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-100 text-gray-900">
        {children}
      </body>
    </html>
  );
}
