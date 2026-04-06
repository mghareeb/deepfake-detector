import { motion, useMotionValue, useTransform, animate } from "framer-motion";
import { useEffect } from "react";

const RADIUS = 80;
const STROKE = 10;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function scoreColor(pct) {
  if (pct < 25) return "#2ed573";
  if (pct < 50) return "#7bed9f";
  if (pct < 65) return "#ffa502";
  if (pct < 80) return "#ff6348";
  return "#ff4757";
}

export default function ScoreMeter({ score }) {
  const pct = Math.round(score * 100);
  const mv = useMotionValue(0);
  const displayed = useTransform(mv, (v) => Math.round(v));
  const color = scoreColor(pct);

  useEffect(() => {
    mv.set(0);
    const ctrl = animate(mv, pct, { duration: 1.6, ease: "easeOut" });
    return ctrl.stop;
  }, [pct, mv]);

  const dashOffset = CIRCUMFERENCE - (score * CIRCUMFERENCE);

  return (
    <div className="relative flex flex-col items-center">
      <svg viewBox="0 0 200 200" className="w-56 h-56 -rotate-90">
        {/* track */}
        <circle cx="100" cy="100" r={RADIUS} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={STROKE} />
        {/* arc */}
        <motion.circle
          cx="100" cy="100" r={RADIUS}
          fill="none"
          stroke={color}
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          initial={{ strokeDashoffset: CIRCUMFERENCE }}
          animate={{ strokeDashoffset: dashOffset }}
          transition={{ duration: 1.6, ease: "easeOut" }}
        />
      </svg>

      {/* center label */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className="flex items-baseline" style={{ color }}>
          <motion.span className="text-5xl font-extrabold leading-none">
            {displayed}
          </motion.span>
          <span className="text-xl font-bold ml-0.5">%</span>
        </div>
        <span className="mt-1 text-xs font-semibold tracking-widest uppercase" style={{ color }}>
          {pct >= 50 ? "Likely Fake" : "Likely Real"}
        </span>
      </div>
    </div>
  );
}
