'use client';
import { useState, useEffect, useRef, useCallback, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import PageShell from '@/components/PageShell';
import { useAuth } from '@/contexts/AuthContext';
import { sessions, patients as patientsApi, connectMonitor, alerts as alertsApi } from '@/lib/api';
import { cn } from '@/lib/utils';
import {
    Play, Square, RefreshCw, Loader2, Zap, AlertTriangle,
    Heart, Droplets, Brain, Eye, TrendingUp, TrendingDown,
    Minus, CheckCircle2, ArrowRight, MonitorDot, Users,
} from 'lucide-react';

/* ── risk-level helpers ── */
const RISK_COLOR = {
    low: 'text-risk-low', LOW: 'text-risk-low',
    moderate: 'text-risk-moderate', MODERATE: 'text-risk-moderate',
    high: 'text-risk-high', HIGH: 'text-risk-high',
    critical: 'text-risk-critical', CRITICAL: 'text-risk-critical',
};
const RISK_BG = {
    low: 'badge-risk-low', LOW: 'badge-risk-low',
    moderate: 'badge-risk-moderate', MODERATE: 'badge-risk-moderate',
    high: 'badge-risk-high', HIGH: 'badge-risk-high',
    critical: 'badge-risk-critical', CRITICAL: 'badge-risk-critical',
};
const riskBarColor = (r) =>
    r >= 0.75 ? 'var(--color-risk-critical)' :
        r >= 0.50 ? 'var(--color-risk-high)' :
            r >= 0.30 ? 'var(--color-risk-moderate)' : 'var(--color-risk-low)';

/* ── Mini sparkline ── */
function Sparkline({ data, accessor, danger, dangerTest, color = 'var(--color-accent)', height = 36 }) {
    const last = data.slice(-20);
    if (!last.length) return null;
    const vals = last.map(accessor);
    const min = Math.min(...vals) - 5;
    const max = Math.max(...vals) + 5;
    const range = max - min || 1;
    return (
        <div className="flex items-end gap-[2px]" style={{ height }}>
            {last.map((_, i) => {
                const v = vals[i];
                const h = Math.max(3, ((v - min) / range) * height);
                const isDanger = dangerTest?.(v);
                return (
                    <div key={i} className="rounded-sm transition-all" style={{
                        width: 4, height: h,
                        background: isDanger ? danger : color,
                        opacity: 0.3 + (i / last.length) * 0.7,
                    }} />
                );
            })}
        </div>
    );
}

/* ──────────────── Main Content ──────────────── */
function MonitorContent() {
    const { user, loading } = useAuth();
    const router = useRouter();
    const searchParams = useSearchParams();
    const patientId = searchParams.get('patient');

    const [patient, setPatient] = useState(null);
    const [patientLoading, setPatientLoading] = useState(false);
    const [sessionId, setSessionId] = useState(null);
    const [connected, setConnected] = useState(false);
    const [starting, setStarting] = useState(false);
    const [currentStep, setCurrentStep] = useState(0);
    const [totalSteps, setTotalSteps] = useState(30);
    const [riskProfile, setRiskProfile] = useState('');
    const [latestData, setLatestData] = useState(null);
    const [vitalsHistory, setVitalsHistory] = useState([]);
    const [riskHistory, setRiskHistory] = useState([]);
    const [activeAlerts, setActiveAlerts] = useState([]);
    const [sessionComplete, setSessionComplete] = useState(false);
    const [screenFlash, setScreenFlash] = useState(false);
    const [error, setError] = useState(null);
    const wsRef = useRef(null);
    const startingRef = useRef(false); // Prevent double-start on re-renders

    // Auth guard
    useEffect(() => {
        if (!loading && !user) router.push('/login');
    }, [user, loading, router]);

    // Load patient data
    useEffect(() => {
        if (!user || !patientId) return;
        let cancelled = false;
        setPatientLoading(true);
        patientsApi.get(patientId)
            .then((data) => { if (!cancelled) setPatient(data); })
            .catch(() => { if (!cancelled) router.push('/patients'); })
            .finally(() => { if (!cancelled) setPatientLoading(false); });
        return () => { cancelled = true; };
    }, [user, patientId, router]);

    // Cleanup WebSocket on unmount or patientId change
    useEffect(() => {
        return () => {
            if (wsRef.current) {
                try { wsRef.current.close(); } catch { /* ignore */ }
                wsRef.current = null;
            }
        };
    }, [patientId]);

    /* ── WebSocket handler ── */
    const handleMessage = useCallback((data) => {
        if (!data || !data.type) return;

        switch (data.type) {
            case 'session_start':
                setRiskProfile(data.risk_profile || '');
                setTotalSteps(data.total_steps || 30);
                break;

            case 'session_complete':
                setSessionComplete(true);
                setConnected(false);
                break;

            case 'error':
                setError(data.message || 'Unknown error');
                break;

            case 'monitoring_data': {
                setLatestData(data);
                setCurrentStep(data.step ?? 0);

                // vitals history
                if (data.vitals) {
                    setVitalsHistory((prev) => [...prev, {
                        step: data.step,
                        time: data.time_minutes,
                        bp: data.vitals.bp ?? 0,
                        hr: data.vitals.hr ?? 0,
                    }]);
                }

                // risk history
                if (data.prediction) {
                    setRiskHistory((prev) => [...prev, {
                        step: data.step,
                        risk: data.prediction.risk_probability ?? 0,
                        level: data.prediction.risk_level ?? 'low',
                    }]);
                }

                // alerts
                if (data.alert) {
                    setActiveAlerts((prev) => [data.alert, ...prev]);
                    if (data.alert.severity === 'CRITICAL') {
                        setScreenFlash(true);
                        setTimeout(() => setScreenFlash(false), 500);
                    }
                }
                if (data.escalation_alerts?.length > 0) {
                    setActiveAlerts((prev) => [...data.escalation_alerts, ...prev]);
                }
                break;
            }
            default:
                break;
        }
    }, []);

    /* ── Session controls ── */
    const startSession = useCallback(async (profile) => {
        if (startingRef.current || connected) return;
        startingRef.current = true;
        setStarting(true);
        setError(null);

        try {
            // Close existing WebSocket if any
            if (wsRef.current) {
                try { wsRef.current.close(); } catch { /* ignore */ }
                wsRef.current = null;
            }

            const session = await sessions.create({
                patient_id: patientId,
                risk_profile: profile || undefined,
            });

            if (!session?.id) {
                throw new Error('Invalid session response');
            }

            setSessionId(session.id);
            setRiskProfile(profile || '');
            setSessionComplete(false);
            setLatestData(null);
            setVitalsHistory([]);
            setRiskHistory([]);
            setActiveAlerts([]);
            setCurrentStep(0);

            const ws = connectMonitor(
                session.id,
                handleMessage,
                (e) => {
                    console.error('WS error:', e);
                    setError('WebSocket connection error');
                },
                () => {
                    setConnected(false);
                },
            );
            wsRef.current = ws;
            setConnected(true);
        } catch (e) {
            setError('Failed to start session: ' + (e.message || 'Unknown error'));
        } finally {
            setStarting(false);
            startingRef.current = false;
        }
    }, [patientId, connected, handleMessage]);

    const stopSession = useCallback(async () => {
        if (wsRef.current) {
            try { wsRef.current.close(); } catch { /* ignore */ }
            wsRef.current = null;
        }
        setConnected(false);
        setSessionComplete(true);
        if (sessionId) {
            try { await sessions.stop(sessionId); } catch { /* ignore */ }
        }
    }, [sessionId]);

    const resetSession = useCallback(() => {
        if (wsRef.current) {
            try { wsRef.current.close(); } catch { /* ignore */ }
            wsRef.current = null;
        }
        setSessionComplete(false);
        setConnected(false);
        setLatestData(null);
        setVitalsHistory([]);
        setRiskHistory([]);
        setActiveAlerts([]);
        setCurrentStep(0);
        setSessionId(null);
        setError(null);
    }, []);

    const acknowledgeAlert = useCallback(async (alertId) => {
        if (!alertId) return;
        try {
            await alertsApi.acknowledge(alertId);
            setActiveAlerts((p) => p.filter((a) => a.id !== alertId));
        } catch { /* ignore */ }
    }, []);

    /* ── Loading / auth guard ── */
    if (loading || !user) {
        return <div className="flex min-h-screen items-center justify-center bg-bg-primary"><Loader2 className="h-8 w-8 animate-spin text-accent" /></div>;
    }

    if (!patientId) {
        return (
            <PageShell>
                <div className="flex flex-col items-center justify-center py-32 text-center">
                    <MonitorDot className="mb-4 h-14 w-14 text-text-muted/30" />
                    <h2 className="text-xl font-bold text-text-primary">No Patient Selected</h2>
                    <p className="mt-2 max-w-sm text-sm text-text-muted">Go to the Patients page to select a patient for monitoring.</p>
                    <button onClick={() => router.push('/patients')} className="mt-6 flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-sm font-semibold text-bg-primary hover:bg-accent-hover cursor-pointer">
                        <Users className="h-4 w-4" /> Go to Patients
                    </button>
                </div>
            </PageShell>
        );
    }

    if (patientLoading) {
        return (
            <PageShell>
                <div className="flex flex-col items-center justify-center py-32">
                    <Loader2 className="h-8 w-8 animate-spin text-accent" />
                    <p className="mt-3 text-sm text-text-muted">Loading patient data…</p>
                </div>
            </PageShell>
        );
    }

    const risk = latestData?.prediction;
    const xai = latestData?.xai;
    const vitals = latestData?.vitals;
    const anomalies = latestData?.anomalies || [];

    return (
        <PageShell>
            {/* Critical flash overlay */}
            {screenFlash && <div className="pointer-events-none fixed inset-0 z-50 bg-risk-critical/15 animate-pulse" />}

            {/* ── Header ── */}
            <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
                <div>
                    <h1 className="flex items-center gap-3 text-2xl font-bold text-text-primary">
                        {connected && <span className="pulse-dot bg-risk-low" />}
                        Real-Time Monitor
                    </h1>
                    <p className="mt-1 text-sm text-text-muted">
                        {patient?.name || `Patient #${patientId?.slice(0, 6)}`}
                        {riskProfile && ` · Profile: ${riskProfile}`}
                        {connected && ` · Step ${currentStep + 1}/${totalSteps}`}
                    </p>
                </div>
                <div className="flex gap-2">
                    {!connected && !sessionComplete && !starting && (
                        <>
                            <button onClick={() => startSession(null)} className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-bg-primary hover:bg-accent-hover cursor-pointer">
                                <Play className="h-4 w-4" /> Start
                            </button>
                            <button onClick={() => startSession('high')} className="flex items-center gap-2 rounded-lg border border-risk-high/40 px-3 py-2 text-xs font-medium text-risk-high hover:bg-risk-high-bg cursor-pointer">
                                <Zap className="h-3.5 w-3.5" /> High Risk
                            </button>
                            <button onClick={() => startSession('critical')} className="flex items-center gap-2 rounded-lg border border-risk-critical/40 px-3 py-2 text-xs font-medium text-risk-critical hover:bg-risk-critical-bg cursor-pointer">
                                <AlertTriangle className="h-3.5 w-3.5" /> Critical
                            </button>
                        </>
                    )}
                    {starting && (
                        <button disabled className="flex items-center gap-2 rounded-lg bg-accent/50 px-4 py-2 text-sm font-semibold text-bg-primary cursor-not-allowed">
                            <Loader2 className="h-4 w-4 animate-spin" /> Starting…
                        </button>
                    )}
                    {connected && (
                        <button onClick={stopSession} className="flex items-center gap-2 rounded-lg bg-error/15 px-4 py-2 text-sm font-semibold text-error hover:bg-error/25 cursor-pointer">
                            <Square className="h-4 w-4" /> Stop
                        </button>
                    )}
                    {sessionComplete && (
                        <button onClick={resetSession} className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-bg-primary hover:bg-accent-hover cursor-pointer">
                            <RefreshCw className="h-4 w-4" /> New Session
                        </button>
                    )}
                </div>
            </div>

            {/* ── Error Banner ── */}
            {error && (
                <div className="mb-4 flex items-center gap-3 rounded-lg border border-risk-critical/30 bg-risk-critical-bg px-4 py-3 text-sm text-risk-critical animate-fade-in">
                    <AlertTriangle className="h-4 w-4 flex-shrink-0" />
                    <span className="flex-1">{error}</span>
                    <button onClick={() => setError(null)} className="text-xs font-medium hover:underline cursor-pointer">Dismiss</button>
                </div>
            )}

            {/* ── Active Alerts ── */}
            {activeAlerts.length > 0 && (
                <div className="mb-4 space-y-2">
                    {activeAlerts.slice(0, 3).map((a, i) => (
                        <div key={a.id || i} className={cn(
                            'flex items-center gap-3 rounded-lg border px-4 py-2.5 text-sm animate-fade-in',
                            a.severity === 'CRITICAL' ? 'border-risk-critical/30 bg-risk-critical-bg' :
                                a.severity === 'HIGH' ? 'border-risk-high/30 bg-risk-high-bg' :
                                    'border-risk-moderate/30 bg-risk-moderate-bg'
                        )}>
                            <span className={cn('rounded-md px-2 py-0.5 text-[11px] font-bold', RISK_BG[a.severity] || RISK_BG.HIGH)}>
                                {a.escalation_level > 0 ? '⬆ ' : ''}{a.severity}
                            </span>
                            <span className="flex-1">{a.message}</span>
                            {a.id && (
                                <button onClick={() => acknowledgeAlert(a.id)} className="rounded-md px-2 py-1 text-xs font-medium text-text-secondary hover:bg-surface-hover cursor-pointer">
                                    <CheckCircle2 className="h-4 w-4" />
                                </button>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {/* ── Ready state ── */}
            {!connected && !latestData && !sessionComplete && !starting && (
                <div className="card flex flex-col items-center py-20 text-center">
                    <Heart className="mb-4 h-14 w-14 text-accent/30" />
                    <h2 className="text-xl font-bold text-text-primary">Ready to Monitor</h2>
                    <p className="mt-2 max-w-lg text-sm text-text-muted">
                        Start a session to begin real-time AI monitoring. The system will simulate realistic vital signs and predict instability using the trained BiGRU model.
                    </p>
                </div>
            )}

            {/* ──────── Monitoring Data ──────── */}
            {latestData && (
                <>
                    {/* Row 1: Risk Gauge | BP | HR */}
                    <div className="mb-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* Risk Gauge */}
                        <div className="card flex flex-col items-center p-5">
                            <p className="mb-3 text-[11px] font-semibold tracking-widest uppercase text-text-muted">Instability Risk</p>
                            <p className={cn('text-5xl font-extrabold tabular-nums', RISK_COLOR[risk?.risk_level] || 'text-accent')}>
                                {risk?.risk_probability != null ? (risk.risk_probability * 100).toFixed(0) : '—'}%
                            </p>
                            <span className={cn('mt-2 rounded-md px-2.5 py-0.5 text-xs font-bold', RISK_BG[risk?.risk_level] || RISK_BG.low)}>
                                {risk?.risk_level?.toUpperCase() || 'UNKNOWN'}
                            </span>
                            {risk?.confidence && (
                                <p className="mt-3 text-[11px] text-text-muted">
                                    95% CI: {(risk.confidence.lower * 100).toFixed(0)}–{(risk.confidence.upper * 100).toFixed(0)}%
                                </p>
                            )}
                            {xai?.risk_trend && (
                                <p className={cn('mt-2 flex items-center gap-1 text-xs font-medium',
                                    xai.risk_trend === 'increasing' ? 'text-risk-high' :
                                        xai.risk_trend === 'decreasing' ? 'text-risk-low' : 'text-text-muted'
                                )}>
                                    {xai.risk_trend === 'increasing' ? <TrendingUp className="h-3.5 w-3.5" /> :
                                        xai.risk_trend === 'decreasing' ? <TrendingDown className="h-3.5 w-3.5" /> :
                                            <Minus className="h-3.5 w-3.5" />}
                                    {xai.risk_trend === 'increasing' ? 'Rising' : xai.risk_trend === 'decreasing' ? 'Falling' : 'Stable'}
                                </p>
                            )}
                        </div>

                        {/* Blood Pressure */}
                        <div className="card p-5">
                            <p className="mb-2 text-[11px] font-semibold tracking-widest uppercase text-text-muted">Blood Pressure</p>
                            <p className={cn('text-4xl font-extrabold tabular-nums', (vitals?.bp ?? 120) < 80 ? 'text-risk-critical' : 'text-accent')}>
                                {vitals?.bp?.toFixed(0) ?? '—'}
                                <span className="ml-1.5 text-sm font-normal text-text-muted">mmHg</span>
                            </p>
                            <p className={cn('mt-2 text-xs', (vitals?.bp_change ?? 0) < -5 ? 'text-risk-critical' : 'text-text-muted')}>
                                Δ {vitals?.bp_change > 0 ? '+' : ''}{vitals?.bp_change?.toFixed(1) ?? '0.0'}
                            </p>
                            <div className="mt-3">
                                <Sparkline data={vitalsHistory} accessor={(v) => v.bp}
                                    color="var(--color-accent)" danger="var(--color-risk-critical)"
                                    dangerTest={(v) => v < 80} />
                            </div>
                        </div>

                        {/* Heart Rate */}
                        <div className="card p-5">
                            <p className="mb-2 text-[11px] font-semibold tracking-widest uppercase text-text-muted">Heart Rate</p>
                            <p className={cn('text-4xl font-extrabold tabular-nums', (vitals?.hr ?? 70) > 110 ? 'text-risk-critical' : 'text-chart-5')}>
                                {vitals?.hr?.toFixed(0) ?? '—'}
                                <span className="ml-1.5 text-sm font-normal text-text-muted">bpm</span>
                            </p>
                            <p className={cn('mt-2 text-xs', (vitals?.hr_change ?? 0) > 5 ? 'text-risk-high' : 'text-text-muted')}>
                                Δ {vitals?.hr_change > 0 ? '+' : ''}{vitals?.hr_change?.toFixed(1) ?? '0.0'}
                            </p>
                            <div className="mt-3">
                                <Sparkline data={vitalsHistory} accessor={(v) => v.hr}
                                    color="var(--color-chart-5)" danger="var(--color-risk-critical)"
                                    dangerTest={(v) => v > 110} />
                            </div>
                        </div>
                    </div>

                    {/* Anomaly badges */}
                    {anomalies.length > 0 && (
                        <div className="mb-4 flex flex-wrap gap-2">
                            {anomalies.map((a, i) => (
                                <span key={i} className="flex items-center gap-1.5 rounded-lg border border-risk-high/30 bg-risk-high-bg px-3 py-1 text-xs font-medium text-risk-high">
                                    <AlertTriangle className="h-3 w-3" />
                                    {a.type?.replace(/_/g, ' ')} ({a.feature?.replace('Current_', '')})
                                </span>
                            ))}
                        </div>
                    )}

                    {/* Row 2: AI Explanation | Risk Timeline */}
                    <div className="mb-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* NL Explanation */}
                        <div className="card p-5">
                            <h4 className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-text-muted">
                                <Brain className="h-4 w-4" /> AI Explanation
                            </h4>
                            <p className="text-sm leading-relaxed text-text-secondary">
                                {xai?.nl_explanation || 'Waiting for AI analysis…'}
                            </p>
                        </div>

                        {/* Risk Timeline */}
                        <div className="card p-5">
                            <h4 className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-text-muted">
                                <TrendingUp className="h-4 w-4" /> Risk Timeline
                            </h4>
                            <div className="flex items-end gap-[2px]" style={{ height: 80 }}>
                                {riskHistory.map((r, i) => {
                                    const h = Math.max(4, r.risk * 80);
                                    return <div key={i} className="rounded-sm transition-all" style={{
                                        flex: 1, height: h, background: riskBarColor(r.risk),
                                        opacity: 0.4 + (i / riskHistory.length) * 0.6, transition: 'height 0.3s',
                                    }} />;
                                })}
                            </div>
                            <div className="mt-2 flex justify-between text-[10px] text-text-muted">
                                <span>0 min</span>
                                <span>{latestData?.time_minutes?.toFixed?.(0) ?? latestData?.time_minutes ?? 0} min</span>
                            </div>

                            {xai?.risk_forecast_5step?.length > 0 && (
                                <div className="mt-4">
                                    <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-text-muted">5-Step Forecast</p>
                                    <div className="flex gap-1.5">
                                        {xai.risk_forecast_5step.map((f, i) => (
                                            <div key={i} className="flex-1 rounded-md py-1 text-center text-xs font-bold tabular-nums" style={{
                                                background: f >= 0.5 ? 'var(--color-risk-high-bg)' : 'var(--color-risk-low-bg)',
                                                color: riskBarColor(f),
                                            }}>
                                                {(f * 100).toFixed(0)}%
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Row 3: SHAP | Attention */}
                    <div className="mb-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* SHAP */}
                        <div className="card p-5">
                            <h4 className="mb-4 flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-text-muted">
                                <Droplets className="h-4 w-4" /> Feature Attribution (SHAP)
                            </h4>
                            {xai?.top_features?.length > 0 ? xai.top_features.map((f, i) => {
                                const maxVal = Math.max(...xai.top_features.map((x) => x.abs_value || 0.01));
                                const pct = maxVal > 0 ? (f.abs_value / maxVal) * 100 : 0;
                                const isPos = f.direction === 'risk_increasing';
                                return (
                                    <div key={i} className="mb-2 flex items-center gap-3 text-xs">
                                        <span className="w-28 truncate text-text-secondary">{f.name?.replace(/_/g, ' ')}</span>
                                        <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface">
                                            <div className={cn('h-full rounded-full transition-all', isPos ? 'bg-risk-critical/60' : 'bg-risk-low/60')}
                                                style={{ width: `${pct}%` }} />
                                        </div>
                                        <span className={cn('w-14 text-right font-mono tabular-nums', isPos ? 'text-risk-critical' : 'text-risk-low')}>
                                            {isPos ? '+' : ''}{f.value?.toFixed(3)}
                                        </span>
                                    </div>
                                );
                            }) : (
                                <p className="text-xs text-text-muted italic">Computing feature attributions…</p>
                            )}
                        </div>

                        {/* Attention */}
                        <div className="card p-5">
                            <h4 className="mb-4 flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-text-muted">
                                <Eye className="h-4 w-4" /> Temporal Attention
                            </h4>
                            {xai?.attention_weights?.length > 0 ? (
                                <>
                                    <div className="flex items-end gap-[2px]" style={{ height: 64 }}>
                                        {xai.attention_weights.map((w, i) => {
                                            const maxW = Math.max(...xai.attention_weights) || 1;
                                            const h = Math.max(3, (w / maxW) * 64);
                                            return <div key={i} className="rounded-sm" style={{
                                                flex: 1, height: h,
                                                background: `oklch(0.72 0.14 195 / ${0.25 + (w / maxW) * 0.75})`,
                                                transition: 'height 0.3s',
                                            }} />;
                                        })}
                                    </div>
                                    <div className="mt-2 flex justify-between text-[10px] text-text-muted">
                                        <span>Oldest</span>
                                        <span>Peak: step {xai.attention_weights.indexOf(Math.max(...xai.attention_weights))}</span>
                                        <span>Latest</span>
                                    </div>
                                </>
                            ) : (
                                <p className="text-xs text-text-muted italic">Waiting for attention weights…</p>
                            )}
                        </div>
                    </div>

                    {/* Progress bar */}
                    <div className="mt-2">
                        <div className="h-1 overflow-hidden rounded-full bg-surface">
                            <div className="h-full rounded-full bg-gradient-to-r from-accent to-chart-5 transition-all" style={{ width: `${((currentStep + 1) / totalSteps) * 100}%` }} />
                        </div>
                        <div className="mt-1.5 flex justify-between text-[11px] text-text-muted">
                            <span>Step {currentStep + 1} / {totalSteps}</span>
                            <span>{latestData?.time_minutes?.toFixed?.(0) ?? latestData?.time_minutes ?? 0} min</span>
                        </div>
                    </div>
                </>
            )}

            {/* ── Session complete ── */}
            {sessionComplete && !connected && (
                <div className="card mt-8 flex flex-col items-center py-12 text-center">
                    <CheckCircle2 className="mb-4 h-14 w-14 text-risk-low" />
                    <h2 className="text-xl font-bold text-text-primary">Session Complete</h2>
                    <p className="mt-2 text-sm text-text-muted">
                        {vitalsHistory.length} vitals recorded · {activeAlerts.length} alerts generated
                    </p>
                    <div className="mt-6 flex gap-3">
                        <button onClick={resetSession} className="flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-sm font-semibold text-bg-primary hover:bg-accent-hover cursor-pointer">
                            <RefreshCw className="h-4 w-4" /> New Session
                        </button>
                        <button onClick={() => router.push('/patients')} className="flex items-center gap-2 rounded-lg border border-border-subtle px-5 py-2.5 text-sm font-medium text-text-secondary hover:bg-surface-hover cursor-pointer">
                            <ArrowRight className="h-4 w-4" /> Patients
                        </button>
                    </div>
                </div>
            )}
        </PageShell>
    );
}

export default function MonitorPage() {
    return (
        <Suspense fallback={<div className="flex min-h-screen items-center justify-center bg-bg-primary"><Loader2 className="h-8 w-8 animate-spin text-accent" /></div>}>
            <MonitorContent />
        </Suspense>
    );
}
