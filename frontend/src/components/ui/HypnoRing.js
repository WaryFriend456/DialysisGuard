'use client';
import { cn } from '@/lib/utils';
import { Loader2 } from 'lucide-react';

export default function HypnoRing({ risk = 0, level = 'low', size = 240 }) {
    const radius = size * 0.35;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (risk * circumference); // 0 to 1

    const colors = {
        low: 'var(--color-risk-low)',
        moderate: 'var(--color-risk-moderate)',
        high: 'var(--color-risk-high)',
        critical: 'var(--color-risk-critical)',
    };

    const color = colors[level.toLowerCase()] || colors.low;

    return (
        <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
            {/* Outer Glow Ring */}
            <div className="absolute inset-0 rounded-full animate-pulse opacity-20"
                style={{ background: `radial-gradient(circle, ${color} 0%, transparent 70%)` }} />

            {/* SVG Ring */}
            <svg width={size} height={size} className="relative z-10 rotate-[-90deg]">
                {/* Track */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke="var(--color-bg-tertiary)"
                    strokeWidth="12"
                    fill="transparent"
                />
                {/* Progress */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke={color}
                    strokeWidth="12"
                    fill="transparent"
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    strokeLinecap="round"
                    className="transition-all duration-1000 ease-out"
                    style={{ filter: `drop-shadow(0 0 8px ${color})` }}
                />
            </svg>

            {/* Inner Data */}
            <div className="absolute z-20 flex flex-col items-center text-center">
                <span className="text-[10px] uppercase tracking-[0.2em] text-text-muted mb-1">Risk Probability</span>
                <div className="flex items-baseline gap-1">
                    <span className="text-5xl font-bold font-mono tracking-tighter text-glow" style={{ color }}>
                        {(risk * 100).toFixed(0)}
                    </span>
                    <span className="text-xl text-text-muted font-mono">%</span>
                </div>
                <span className={cn(
                    "mt-2 text-xs font-bold uppercase tracking-widest px-2 py-0.5 rounded border backdrop-blur-md",
                    `text-[${color}] border-[${color}] bg-[${color}]/10`
                )} style={{ color, borderColor: color, backgroundColor: `${color}1A` }}>
                    {level}
                </span>
            </div>
        </div>
    );
}
