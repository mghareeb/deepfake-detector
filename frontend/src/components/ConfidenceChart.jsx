import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

export default function ConfidenceChart({ pixelScore, freqScore, ensembleScore }) {
  const data = {
    labels: ["Pixel (CNN)", "Frequency (FFT)", "Ensemble"],
    datasets: [
      {
        data: [
          Math.round(pixelScore * 100),
          Math.round(freqScore * 100),
          Math.round(ensembleScore * 100),
        ],
        backgroundColor: [
          "rgba(0, 212, 255, 0.65)",
          "rgba(162, 115, 255, 0.65)",
          "rgba(255, 71, 87, 0.65)",
        ],
        borderColor: ["#00d4ff", "#a273ff", "#ff4757"],
        borderWidth: 1,
        borderRadius: 6,
        barPercentage: 0.55,
      },
    ],
  };

  const options = {
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: (ctx) => `${ctx.raw}% fake probability` } },
    },
    scales: {
      x: {
        min: 0,
        max: 100,
        ticks: { color: "rgba(255,255,255,0.4)", callback: (v) => `${v}%` },
        grid: { color: "rgba(255,255,255,0.04)" },
      },
      y: {
        ticks: { color: "rgba(255,255,255,0.7)", font: { size: 13, weight: 600 } },
        grid: { display: false },
      },
    },
  };

  return (
    <div className="rounded-2xl bg-surface border border-white/5 p-6">
      <h3 className="text-sm font-semibold text-gray-300 mb-1">Score Breakdown</h3>
      <p className="text-xs text-gray-500 mb-5">
        Ensemble = 70 % pixel + 30 % frequency
      </p>
      <div className="h-[140px]">
        <Bar data={data} options={options} />
      </div>
    </div>
  );
}
