export default function ResponseCard({ data }: { data: any }) {
  return (
    <div className="mt-4 bg-gray-50 border rounded-lg p-4">
      {data.intent && (
        <p className="text-sm text-gray-500 mb-1">
          Intent: <b>{data.intent.toUpperCase()}</b>
        </p>
      )}

      <pre className="whitespace-pre-wrap text-sm">
        {data.answer}
      </pre>
    </div>
  );
}
