export default function ImageComparison({ imagePreview, heatmapBase64, faceDetected }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-4xl mx-auto">
        {/* Original */}
        <div className="rounded-2xl overflow-hidden bg-surface border border-white/5">
          <div className="px-4 py-2 bg-white/[0.03] border-b border-white/5 flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-brand" />
            <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">Original</span>
          </div>
          <img src={imagePreview} alt="Original" className="w-full object-contain max-h-[360px]" />
        </div>

        {/* Heatmap */}
        <div className="rounded-2xl overflow-hidden bg-surface border border-white/5">
          <div className="px-4 py-2 bg-white/[0.03] border-b border-white/5 flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-fake" />
            <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">Grad-CAM Heatmap</span>
          </div>
          <img
            src={`data:image/png;base64,${heatmapBase64}`}
            alt="Heatmap"
            className="w-full object-contain max-h-[360px]"
          />
        </div>
      </div>

      {/* Face badge */}
      <div className="flex justify-center">
        <span className={[
          "text-xs font-medium px-3 py-1 rounded-full border",
          faceDetected
            ? "border-real/30 text-real bg-real/10"
            : "border-white/10 text-gray-500 bg-white/[0.03]",
        ].join(" ")}>
          {faceDetected ? "Face detected" : "No face detected — full image analysed"}
        </span>
      </div>
    </div>
  );
}
