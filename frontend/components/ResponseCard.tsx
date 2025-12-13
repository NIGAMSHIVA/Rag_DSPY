export default function ResponseCard({ data }: any) {
  return (
    <div className="mt-4 bg-gray-50 p-4 rounded">
      <p className="text-sm text-gray-500 mb-1">
        Intent: <b>{data.intent}</b>
      </p>

      <p className="whitespace-pre-wrap">{data.answer}</p>
    </div>
  );
}
