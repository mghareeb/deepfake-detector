import { useCallback } from "react";
import { useDropzone } from "react-dropzone";

const ACCEPT = { "image/jpeg": [], "image/png": [], "image/webp": [] };
const MAX_SIZE = 10 * 1024 * 1024;

export default function DropZone({ onImageSelected, imagePreview, loading, onClear }) {
  const onDrop = useCallback(
    (accepted) => { if (accepted.length > 0) onImageSelected(accepted[0]); },
    [onImageSelected],
  );

  const onDropRejected = useCallback(
    (rejections) => {
      if (rejections.length > 0) {
        const err = rejections[0].errors[0];
        if (err.code === "file-too-large") {
          const sizeMB = (rejections[0].file.size / 1024 / 1024).toFixed(1);
          onImageSelected({ _rejected: true, message: `Image is too large (${sizeMB} MB). Maximum size is 10 MB.` });
        } else if (err.code === "file-invalid-type") {
          onImageSelected({ _rejected: true, message: "Invalid file type. Please upload JPEG, PNG, or WebP." });
        }
      }
    },
    [onImageSelected],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: ACCEPT,
    maxSize: MAX_SIZE,
    multiple: false,
    disabled: loading,
  });

  return (
    <div className="flex flex-col items-center gap-3">
      <div
        {...getRootProps()}
        className={[
          "w-full rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer overflow-hidden",
          "flex items-center justify-center",
          isDragActive
            ? "border-brand bg-brand/10 scale-[1.01]"
            : imagePreview
              ? "border-white/15 bg-surface"
              : "border-white/15 bg-surface hover:border-brand/50 hover:bg-surface-hover",
          loading && "opacity-50 pointer-events-none",
          !imagePreview && "min-h-[260px]",
        ].join(" ")}
      >
        <input {...getInputProps()} />

        {imagePreview ? (
          <img src={imagePreview} alt="Uploaded" className="w-full max-h-[520px] object-contain p-2" />
        ) : (
          <div className="flex flex-col items-center gap-3 text-gray-500 py-10 px-6 text-center">
            <svg className="w-12 h-12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <p className="text-base">
              {isDragActive ? "Drop your image here" : "Drag & drop an image, or click to select"}
            </p>
            <span className="text-xs opacity-60">JPEG, PNG, WebP up to 10 MB</span>
          </div>
        )}
      </div>

      {imagePreview && !loading && (
        <button
          onClick={(e) => { e.stopPropagation(); onClear(); }}
          className="text-sm text-gray-400 border border-white/15 rounded-lg px-4 py-1.5 hover:border-red-500 hover:text-red-400 transition"
        >
          Clear image
        </button>
      )}
    </div>
  );
}
