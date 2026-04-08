import { useState, useCallback } from "react";
import DropZone from "./components/DropZone.jsx";
import ScoreMeter from "./components/ScoreMeter.jsx";
import ImageComparison from "./components/ImageComparison.jsx";
import ConfidenceChart from "./components/ConfidenceChart.jsx";
import Footer from "./components/Footer.jsx";
import { predictImage } from "./api/predict.js";

export default function App() {
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB

  const handleImageSelected = useCallback(async (file) => {
    // Handle rejected files from DropZone
    if (file._rejected) {
      setError(file.message);
      return;
    }
    if (file.size > MAX_FILE_SIZE) {
      setError(`Image is too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum size is 10 MB.`);
      return;
    }
    setImagePreview(URL.createObjectURL(file));
    setPrediction(null);
    setError(null);
    setLoading(true);
    try {
      const result = await predictImage(file);
      setPrediction(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleClear = useCallback(() => {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImagePreview(null);
    setPrediction(null);
    setError(null);
  }, [imagePreview]);

  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Hero ── */}
      <header className="pt-20 pb-12 text-center px-4">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-brand-dim text-brand text-sm font-medium mb-6">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
          EfficientNet-B4 + FFT Analysis
        </div>
        <h1 className="text-5xl sm:text-6xl font-extrabold tracking-tight bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">
          Is this image real?
        </h1>
        <p className="mt-4 text-lg text-gray-400 max-w-xl mx-auto">
          Drop an image below and our AI will analyse pixel patterns and frequency
          artifacts to determine if it's been manipulated.
        </p>
      </header>

      {/* ── Main content ── */}
      <main className="flex-1 w-full max-w-6xl mx-auto px-4 pb-16 space-y-12">
        {/* Upload zone */}
        <section className="max-w-2xl mx-auto">
          <DropZone
            onImageSelected={handleImageSelected}
            imagePreview={imagePreview}
            loading={loading}
            onClear={handleClear}
          />

          {loading && (
            <div className="mt-6 flex items-center justify-center gap-3 text-gray-400">
              <div className="h-5 w-5 rounded-full border-2 border-brand/30 border-t-brand animate-[spin-slow_0.8s_linear_infinite]" />
              Analyzing image...
            </div>
          )}

          {error && (
            <div className="mt-4 flex items-center justify-between rounded-xl bg-red-500/10 border border-red-500/30 px-4 py-3 text-sm text-red-400">
              <span>Error: {error}</span>
              <button onClick={() => setError(null)} className="underline opacity-70 hover:opacity-100">
                Dismiss
              </button>
            </div>
          )}
        </section>

        {/* Results */}
        {prediction && (
          <div className="space-y-12 animate-[pulse-ring_0.4s_ease-out_1]">
            {/* Score meter */}
            <section className="flex justify-center">
              <ScoreMeter score={prediction.score} />
            </section>

            {/* Side-by-side: original vs heatmap */}
            {imagePreview && (
              <section>
                <h2 className="text-center text-lg font-semibold text-gray-300 mb-6">
                  Original vs Grad-CAM Heatmap
                </h2>
                <ImageComparison
                  imagePreview={imagePreview}
                  heatmapBase64={prediction.heatmap_base64}
                  faceDetected={prediction.face_detected}
                />
              </section>
            )}

            {/* Bar chart */}
            <section className="max-w-2xl mx-auto">
              <ConfidenceChart
                pixelScore={prediction.pixel_score}
                freqScore={prediction.freq_score}
                ensembleScore={prediction.score}
              />
            </section>
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}
