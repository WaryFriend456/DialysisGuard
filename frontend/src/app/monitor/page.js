'use client';

import { Suspense, useEffect, useMemo, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
    AlertTriangle,
    Brain,
    CheckCircle2,
    Eye,
    Heart,
    Loader2,
    Minus,
    MonitorDot,
    Play,
    Square,
    TrendingDown,
    TrendingUp,
    Users,
    Zap,
} from 'lucide-react';
import PageShell from '@/components/PageShell';
import { useAuth } from '@/contexts/AuthContext';
import { useMonitoring } from '@/contexts/MonitoringContext';
import { alerts as alertsApi, patients as patientsApi } from '@/lib/api';
import { cn } from '@/lib/utils';

const RISK_COLOR = {
    low: 'text-risk-low',
    moderate: 'text-risk-moderate',
    high: 'text-risk-high',
    critical: 'text-risk-critical',
};

const RISK_BG = {
    low: 'badge-risk-low',
    moderate: 'badge-risk-moderate',
    high: 'badge-risk-high',
    critical: 'badge-risk-critical',
    LOW: 'badge-risk-low',
    MODERATE: 'badge-risk-moderate',
    HIGH: 'badge-risk-high',
    CRITICAL: 'badge-risk-critical',
};

function Sparkline({ data, accessor, dangerTest, baseColor, dangerColor }) {
    const values = data.slice(-20).map(accessor);
    if (!values.length) return null;
    const min = Math.min(...values) - 4;
    const max = Math.max(...values) + 4;
    const range = max - min || 1;

    return (
        <div className="flex h-14 items-end gap-[3px]">
            {values.map((value, index) => {
                const height = Math.max(6, ((value - min) / range) * 56);
                const color = dangerTest?.(value) ? dangerColor : baseColor;
                return (
                    <div
                        key={`${value}-${index}`}
                        className="flex-1 rounded-full"
                        style={{ height, background: color, opacity: 0.35 + (index / values.length) * 0.65 }}
                    />
                );
            })}
        </div>
    );
}

function MonitorContent() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const patientIdFromQuery = searchParams.get('patient');
    const { user, loading } = useAuth();
    const {
        session,
        connected,
        data,
        history,
        alerts,
        error,
        status,
        patientSummary,
        hasActiveSession,
        startSession,
        stopSession,
        ensurePatientSession,
    } = useMonitoring();

    const [patient, setPatient] = useState(null);
    const [pageError, setPageError] = useState(null);
    const activePatientId = patientIdFromQuery || session?.patient_id || null;
    const patientLoading = Boolean(activePatientId) && patient?.id !== activePatientId;

    useEffect(() => {
        if (!loading && !user) {
            router.push('/login');
        }
    }, [loading, router, user]);

    useEffect(() => {
        if (!user || !patientIdFromQuery) return;
        ensurePatientSession(patientIdFromQuery, { autoStart: false });
    }, [ensurePatientSession, patientIdFromQuery, user]);

    useEffect(() => {
        if (!activePatientId || !user) return;

        let cancelled = false;
        patientsApi.get(activePatientId)
            .then((record) => {
                if (!cancelled) {
                    setPatient(record);
                    setPageError(null);
                }
            })
            .catch(() => {
                if (!cancelled) {
                    setPageError('Unable to load the selected patient.');
                }
            });

        return () => {
            cancelled = true;
        };
    }, [activePatientId, user]);
    const screenFlash = alerts[0]?.severity === 'CRITICAL';

    const latestPrediction = data?.prediction;
    const latestVitals = data?.vitals;
    const latestXai = data?.xai;
    const anomalies = data?.anomalies || [];
    const topAlerts = alerts.slice(0, 3);

    const riskTrend = latestXai?.risk_trend;
    const trendIndicator = useMemo(() => {
        if (riskTrend === 'increasing') {
            return { icon: TrendingUp, label: 'Rising', color: 'text-risk-high' };
        }
        if (riskTrend === 'decreasing') {
            return { icon: TrendingDown, label: 'Improving', color: 'text-risk-low' };
        }
        return { icon: Minus, label: 'Stable', color: 'text-text-muted' };
    }, [riskTrend]);
    const TrendIcon = trendIndicator.icon;

    const acknowledgeAlert = async (alertId) => {
        if (!alertId) return;
        try {
            await alertsApi.acknowledge(alertId);
        } catch {
            setPageError('Unable to acknowledge the alert right now.');
        }
    };

    const handleStart = async (riskProfile = null) => {
        if (!activePatientId) return;
        await startSession(activePatientId, riskProfile);
    };

    const handleResume = async () => {
        if (!activePatientId) return;
        await ensurePatientSession(activePatientId, { autoStart: false });
    };

    if (loading || !user) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-bg-primary">
                <Loader2 className="h-8 w-8 animate-spin text-accent" />
            </div>
        );
    }

    return (
        <PageShell>
            {screenFlash && <div className="pointer-events-none fixed inset-0 z-50 bg-risk-critical/10" />}

            {!activePatientId && !hasActiveSession && (
                <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
                    <section className="card overflow-hidden">
                        <div className="border-b border-border-subtle px-6 py-5">
                            <p className="text-xs uppercase tracking-[0.2em] text-text-muted">No session selected</p>
                            <h2 className="mt-2 text-2xl font-semibold text-text-primary">Select a patient to begin or resume monitoring</h2>
                            <p className="mt-2 max-w-2xl text-sm text-text-secondary">
                                The monitor workspace stays available from the sidebar. If a session is already running, the header chip above returns you directly to it.
                            </p>
                        </div>
                        <div className="grid gap-4 px-6 py-6 sm:grid-cols-2">
                            <button
                                onClick={() => router.push('/patients')}
                                className="rounded-3xl bg-accent px-5 py-4 text-left text-sm font-medium text-bg-primary transition-transform hover:-translate-y-0.5"
                            >
                                <p className="flex items-center gap-2 text-base font-semibold">
                                    <Users className="h-4 w-4" />
                                    Open patient list
                                </p>
                                <p className="mt-2 text-bg-primary/80">Choose a patient profile and launch bedside monitoring.</p>
                            </button>
                            <button
                                onClick={() => router.push(user.role === 'doctor' ? '/dashboard/doctor' : '/dashboard/caregiver')}
                                className="rounded-3xl border border-border-subtle bg-surface px-5 py-4 text-left text-sm font-medium text-text-primary transition-transform hover:-translate-y-0.5"
                            >
                                <p className="text-base font-semibold">Return to dashboard</p>
                                <p className="mt-2 text-text-secondary">Review alerts, patient queue, and current operational summary.</p>
                            </button>
                        </div>
                    </section>

                    <section className="card p-6">
                        <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Monitoring behavior</p>
                        <div className="mt-4 space-y-3 text-sm text-text-secondary">
                            <p>Readings are delivered on a 3-5 second wall-clock cadence.</p>
                            <p>Leaving the monitor page no longer stops an active session.</p>
                            <p>Returning through the sidebar resumes the in-flight session instead of forcing a restart.</p>
                        </div>
                    </section>
                </div>
            )}

            {activePatientId && (
                <div className="space-y-6">
                    {(pageError || error) && (
                        <div className="flex items-center gap-3 rounded-3xl border border-risk-critical/30 bg-risk-critical-bg px-5 py-4 text-sm text-risk-critical">
                            <AlertTriangle className="h-4 w-4 shrink-0" />
                            <span>{pageError || error}</span>
                        </div>
                    )}

                    <section className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
                        <div className="card p-6">
                            <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Patient context</p>
                            <div className="mt-4 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                                <div>
                                    <h2 className="text-3xl font-semibold text-text-primary">
                                        {patient?.name || patientSummary?.name || 'Selected patient'}
                                    </h2>
                                    <p className="mt-2 text-sm text-text-secondary">
                                        {patientLoading
                                            ? 'Loading patient profile...'
                                            : patient
                                                ? `${patient.age} years · ${patient.gender} · ${patient.weight} kg · ${patient.disease_severity} severity`
                                                : 'Patient profile unavailable'}
                                    </p>
                                    <p className="mt-2 text-xs uppercase tracking-wide text-text-muted">
                                        Session state: {connected ? 'live' : hasActiveSession ? 'paused' : status}
                                        {session ? ` · Step ${session.current_step || 0}/${session.total_steps || 30}` : ''}
                                    </p>
                                </div>
                                <div className="flex flex-wrap gap-3">
                                    {!hasActiveSession && (
                                        <>
                                            <button
                                                onClick={() => handleStart(null)}
                                                className="flex items-center gap-2 rounded-2xl bg-accent px-4 py-3 text-sm font-semibold text-bg-primary"
                                            >
                                                <Play className="h-4 w-4" />
                                                Start monitoring
                                            </button>
                                            <button
                                                onClick={() => handleStart('high')}
                                                className="flex items-center gap-2 rounded-2xl border border-risk-high/30 bg-risk-high-bg px-4 py-3 text-sm font-medium text-risk-high"
                                            >
                                                <Zap className="h-4 w-4" />
                                                High-risk demo
                                            </button>
                                        </>
                                    )}
                                    {hasActiveSession && !connected && (
                                        <button
                                            onClick={handleResume}
                                            className="flex items-center gap-2 rounded-2xl bg-accent px-4 py-3 text-sm font-semibold text-bg-primary"
                                        >
                                            <MonitorDot className="h-4 w-4" />
                                            Resume session
                                        </button>
                                    )}
                                    {connected && (
                                        <button
                                            onClick={stopSession}
                                            className="flex items-center gap-2 rounded-2xl border border-risk-critical/30 bg-risk-critical-bg px-4 py-3 text-sm font-semibold text-risk-critical"
                                        >
                                            <Square className="h-4 w-4" />
                                            Stop session
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>

                        <div className="card p-6">
                            <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Operational status</p>
                            <div className="mt-4 grid gap-3 sm:grid-cols-3 xl:grid-cols-1">
                                <div className="rounded-3xl border border-border-subtle bg-surface px-4 py-4">
                                    <p className="text-xs uppercase tracking-wide text-text-muted">Connection</p>
                                    <p className="mt-2 text-xl font-semibold text-text-primary">{connected ? 'Live' : hasActiveSession ? 'Paused' : 'Idle'}</p>
                                </div>
                                <div className="rounded-3xl border border-border-subtle bg-surface px-4 py-4">
                                    <p className="text-xs uppercase tracking-wide text-text-muted">Elapsed</p>
                                    <p className="mt-2 text-xl font-semibold text-text-primary">{data?.time_minutes ?? 0} min</p>
                                </div>
                                <div className="rounded-3xl border border-border-subtle bg-surface px-4 py-4">
                                    <p className="text-xs uppercase tracking-wide text-text-muted">Alerts</p>
                                    <p className="mt-2 text-xl font-semibold text-text-primary">{alerts.length}</p>
                                </div>
                            </div>
                        </div>
                    </section>

                    {topAlerts.length > 0 && (
                        <section className="space-y-3">
                            {topAlerts.map((alert, index) => (
                                <div
                                    key={`${alert.id || alert.message}-${index}`}
                                    className="flex items-center gap-3 rounded-3xl border border-risk-high/25 bg-risk-high-bg px-5 py-4"
                                >
                                    <span className={cn('rounded-full px-3 py-1 text-xs font-semibold', RISK_BG[alert.severity] || RISK_BG.HIGH)}>
                                        {alert.severity}
                                    </span>
                                    <p className="flex-1 text-sm text-text-primary">{alert.message}</p>
                                    {alert.id && (
                                        <button
                                            onClick={() => acknowledgeAlert(alert.id)}
                                            className="rounded-2xl border border-border-subtle px-3 py-2 text-xs font-medium text-text-secondary hover:bg-surface-hover"
                                        >
                                            <CheckCircle2 className="h-4 w-4" />
                                        </button>
                                    )}
                                </div>
                            ))}
                        </section>
                    )}

                    <section className="grid gap-4 lg:grid-cols-3">
                        <div className="card p-6">
                            <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Instability risk</p>
                            <p className={cn('mt-4 text-5xl font-bold', RISK_COLOR[latestPrediction?.risk_level] || 'text-text-primary')}>
                                {latestPrediction?.risk_probability != null ? `${Math.round(latestPrediction.risk_probability * 100)}%` : '--'}
                            </p>
                            <div className="mt-3 flex items-center gap-3">
                                <span className={cn('rounded-full px-3 py-1 text-xs font-semibold', RISK_BG[latestPrediction?.risk_level] || RISK_BG.low)}>
                                    {latestPrediction?.risk_level?.toUpperCase() || 'WAITING'}
                                </span>
                                <span className="text-xs text-text-muted">
                                    CI {latestPrediction?.confidence?.lower != null ? `${Math.round(latestPrediction.confidence.lower * 100)}-${Math.round(latestPrediction.confidence.upper * 100)}%` : '--'}
                                </span>
                            </div>
                            <div className={cn('mt-4 inline-flex items-center gap-2 text-sm font-medium', trendIndicator.color)}>
                                <TrendIcon className="h-4 w-4" />
                                {trendIndicator.label}
                            </div>
                        </div>

                        <div className="card p-6">
                            <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Blood pressure</p>
                            <p className={cn('mt-4 text-4xl font-bold', (latestVitals?.bp ?? 120) < 80 ? 'text-risk-critical' : 'text-text-primary')}>
                                {latestVitals?.bp != null ? `${latestVitals.bp.toFixed(0)} mmHg` : '--'}
                            </p>
                            <p className="mt-2 text-sm text-text-secondary">Delta {latestVitals?.bp_change != null ? latestVitals.bp_change.toFixed(1) : '--'}</p>
                            <div className="mt-4">
                                <Sparkline
                                    data={history.vitals}
                                    accessor={(item) => item.bp}
                                    dangerTest={(value) => value < 80}
                                    baseColor="var(--color-accent)"
                                    dangerColor="var(--color-risk-critical)"
                                />
                            </div>
                        </div>

                        <div className="card p-6">
                            <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Heart rate</p>
                            <p className={cn('mt-4 text-4xl font-bold', (latestVitals?.hr ?? 70) > 110 ? 'text-risk-critical' : 'text-text-primary')}>
                                {latestVitals?.hr != null ? `${latestVitals.hr.toFixed(0)} bpm` : '--'}
                            </p>
                            <p className="mt-2 text-sm text-text-secondary">Delta {latestVitals?.hr_change != null ? latestVitals.hr_change.toFixed(1) : '--'}</p>
                            <div className="mt-4">
                                <Sparkline
                                    data={history.vitals}
                                    accessor={(item) => item.hr}
                                    dangerTest={(value) => value > 110}
                                    baseColor="var(--color-chart-5)"
                                    dangerColor="var(--color-risk-critical)"
                                />
                            </div>
                        </div>
                    </section>

                    <section className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
                        <div className="card p-6">
                            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-text-muted">
                                <Brain className="h-4 w-4" />
                                AI explanation
                            </div>
                            <p className="mt-4 text-sm leading-7 text-text-secondary">
                                {latestXai?.nl_explanation || 'The monitor will summarize why risk is changing as soon as live data arrives.'}
                            </p>

                            {latestXai?.risk_forecast_5step?.length > 0 && (
                                <div className="mt-6">
                                    <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Forecast</p>
                                    <div className="mt-3 grid grid-cols-5 gap-2">
                                        {latestXai.risk_forecast_5step.map((value, index) => (
                                            <div key={`${value}-${index}`} className="rounded-2xl border border-border-subtle bg-surface px-3 py-3 text-center">
                                                <p className="text-xs text-text-muted">+{index + 1}</p>
                                                <p className="mt-1 text-sm font-semibold text-text-primary">{Math.round(value * 100)}%</p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="card p-6">
                            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-text-muted">
                                <AlertTriangle className="h-4 w-4" />
                                Active findings
                            </div>
                            <div className="mt-4 space-y-3">
                                {anomalies.length === 0 && (
                                    <p className="text-sm text-text-secondary">No anomaly flags at the current step.</p>
                                )}
                                {anomalies.map((anomaly, index) => (
                                    <div key={`${anomaly.type}-${index}`} className="rounded-2xl border border-border-subtle bg-surface px-4 py-3">
                                        <p className="text-sm font-medium text-text-primary">{anomaly.type?.replace(/_/g, ' ')}</p>
                                        <p className="mt-1 text-xs text-text-muted">{anomaly.feature?.replace('Current_', '') || 'Physiology'} · {anomaly.severity}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </section>

                    <section className="grid gap-4 xl:grid-cols-2">
                        <div className="card p-6">
                            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-text-muted">
                                <Heart className="h-4 w-4" />
                                Realtime feature attribution
                            </div>
                            <div className="mt-5 space-y-3">
                                {(latestXai?.top_features || []).slice(0, 5).map((feature, index, items) => {
                                    const max = Math.max(...items.map((entry) => entry.abs_value || 0.01), 0.01);
                                    const width = `${(feature.abs_value / max) * 100}%`;
                                    return (
                                        <div key={`${feature.name}-${index}`}>
                                            <div className="mb-2 flex items-center justify-between gap-3 text-xs">
                                                <span className="truncate text-text-secondary">{feature.name?.replace(/_/g, ' ')}</span>
                                                <span className="font-mono text-text-primary">{feature.value?.toFixed?.(3) ?? feature.value}</span>
                                            </div>
                                            <div className="h-2 rounded-full bg-surface-hover">
                                                <div
                                                    className={cn('h-2 rounded-full', feature.direction === 'risk_increasing' ? 'bg-risk-critical' : 'bg-risk-low')}
                                                    style={{ width }}
                                                />
                                            </div>
                                        </div>
                                    );
                                })}
                                {!latestXai?.top_features?.length && (
                                    <p className="text-sm text-text-secondary">Feature attribution will populate after the first readings arrive.</p>
                                )}
                            </div>
                        </div>

                        <div className="card p-6">
                            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.2em] text-text-muted">
                                <Eye className="h-4 w-4" />
                                Temporal attention
                            </div>
                            <div className="mt-5">
                                {latestXai?.attention_weights?.length ? (
                                    <>
                                        <div className="flex h-32 items-end gap-[3px]">
                                            {latestXai.attention_weights.map((weight, index) => {
                                                const maxWeight = Math.max(...latestXai.attention_weights, 0.01);
                                                return (
                                                    <div
                                                        key={`${weight}-${index}`}
                                                        className="flex-1 rounded-t-2xl bg-accent/70"
                                                        style={{ height: `${Math.max(8, (weight / maxWeight) * 128)}px`, opacity: 0.3 + (weight / maxWeight) * 0.7 }}
                                                    />
                                                );
                                            })}
                                        </div>
                                        <div className="mt-3 flex items-center justify-between text-xs text-text-muted">
                                            <span>Older steps</span>
                                            <span>Most recent step</span>
                                        </div>
                                    </>
                                ) : (
                                    <p className="text-sm text-text-secondary">Attention weights will appear once the sequence has enough context.</p>
                                )}
                            </div>
                        </div>
                    </section>
                </div>
            )}
        </PageShell>
    );
}

export default function MonitorPage() {
    return (
        <Suspense
            fallback={
                <div className="flex min-h-screen items-center justify-center bg-bg-primary">
                    <Loader2 className="h-8 w-8 animate-spin text-accent" />
                </div>
            }
        >
            <MonitorContent />
        </Suspense>
    );
}
