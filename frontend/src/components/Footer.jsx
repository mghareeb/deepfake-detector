const METRICS = [
  { label: "AUC", value: "0.94" },
  { label: "F1 Score", value: "0.91" },
  { label: "Accuracy", value: "93.2%" },
];

export default function Footer() {
  return (
    <footer className="border-t border-white/5 bg-white/[0.02]">
      <div className="max-w-6xl mx-auto px-4 py-8 flex flex-col items-center gap-6">
        {/* Stat pills */}
        <div className="flex flex-wrap justify-center gap-4">
          {METRICS.map((m) => (
            <div
              key={m.label}
              className="flex items-center gap-3 rounded-xl bg-surface border border-white/5 px-5 py-3"
            >
              <span className="text-2xl font-bold text-white">{m.value}</span>
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                {m.label}
              </span>
            </div>
          ))}
        </div>

        <p className="text-xs text-gray-600 text-center max-w-md">
          Benchmark targets on FaceForensics++. Model currently uses ImageNet
          pretrained weights — fine-tune on a deepfake dataset for production use.
        </p>

        <p className="text-xs text-gray-700">
          Deepfake Detector &middot; EfficientNet-B4 + FFT Ensemble
        </p>
      </div>
    </footer>
  );
}
