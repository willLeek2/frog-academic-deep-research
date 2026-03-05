export default function ReportViewer({ report }: { report: string }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Generated Report</h3>
      <div className="prose prose-sm max-w-none">
        <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono leading-relaxed">
          {report}
        </pre>
      </div>
    </div>
  );
}
