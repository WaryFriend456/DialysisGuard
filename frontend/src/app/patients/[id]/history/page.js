'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { ChevronRight, Clock3, FileText, Loader2, ShieldAlert, TrendingUp } from 'lucide-react';
import PageShell from '@/components/PageShell';
import { useAuth } from '@/contexts/AuthContext';
import { patients as patientsApi, sessions as sessionsApi } from '@/lib/api';

export default function PatientHistoryPage() {
    const params = useParams();
    const router = useRouter();
    const { user, loading: authLoading } = useAuth();
    const [patient, setPatient] = useState(null);
    const [history, setHistory] = useState([]);
    const [selectedReport, setSelectedReport] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        if (!authLoading && !user) {
            router.push('/login');
        }
    }, [authLoading, router, user]);

    useEffect(() => {
        if (!user || !params?.id) return;
        let cancelled = false;

        async function load() {
            setLoading(true);
            try {
                const [patientData, sessionData] = await Promise.all([
                    patientsApi.get(params.id),
                    sessionsApi.forPatient(params.id),
                ]);
                if (!cancelled) {
                    setPatient(patientData);
                    setHistory(sessionData.sessions || []);
                }
            } catch (err) {
                if (!cancelled) {
                    setError(err.message || 'Unable to load patient history');
                }
            } finally {
                if (!cancelled) setLoading(false);
            }
        }

        load();
        return () => {
            cancelled = true;
        };
    }, [params?.id, user]);

    const loadReport = async (sessionId) => {
        try {
            const report = await sessionsApi.report(sessionId);
            setSelectedReport(report);
        } catch (err) {
            setError(err.message || 'Unable to load session report');
        }
    };

    if (authLoading || !user || loading) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-bg-primary">
                <Loader2 className="h-8 w-8 animate-spin text-accent" />
            </div>
        );
    }

    if (error && !patient) {
        return (
            <PageShell>
                <div className="card p-8">
                    <p className="text-lg font-semibold text-text-primary">History unavailable</p>
                    <p className="mt-2 text-sm text-text-secondary">{error}</p>
                    <button
                        onClick={() => router.push('/patients')}
                        className="mt-5 rounded-2xl bg-accent px-4 py-3 text-sm font-semibold text-bg-primary"
                    >
                        Return to patients
                    </button>
                </div>
            </PageShell>
        );
    }

    return (
        <PageShell>
            <section className="card p-6">
                <div className="flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
                    <div>
                        <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Patient history</p>
                        <h2 className="mt-2 text-3xl font-semibold text-text-primary">{patient?.name || 'Patient'} session timeline</h2>
                        <p className="mt-2 text-sm text-text-secondary">Review prior monitoring sessions, alert counts, and auto-generated reports before the demo.</p>
                    </div>
                    <div className="flex flex-wrap gap-3">
                        <button
                            onClick={() => router.push(`/patients/${params.id}`)}
                            className="rounded-2xl border border-border-subtle px-4 py-3 text-sm font-medium text-text-primary hover:bg-surface-hover"
                        >
                            Back to profile
                        </button>
                        <button
                            onClick={() => router.push(`/monitor?patient=${params.id}`)}
                            className="rounded-2xl bg-accent px-4 py-3 text-sm font-semibold text-bg-primary"
                        >
                            Open monitor
                        </button>
                    </div>
                </div>
            </section>

            {error && (
                <div className="mt-6 rounded-2xl border border-risk-critical/30 bg-risk-critical-bg px-4 py-3 text-sm text-risk-critical">
                    {error}
                </div>
            )}

            <section className="mt-6 grid gap-6 xl:grid-cols-[1fr_0.9fr]">
                <div className="card p-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Sessions</p>
                            <h3 className="mt-2 text-xl font-semibold text-text-primary">{history.length} recorded runs</h3>
                        </div>
                    </div>

                    <div className="mt-5 space-y-3">
                        {history.length === 0 && (
                            <p className="text-sm text-text-secondary">No monitoring history is available yet for this patient.</p>
                        )}
                        {history.map((entry) => (
                            <button
                                key={entry.id}
                                onClick={() => loadReport(entry.id)}
                                className="flex w-full items-center justify-between gap-4 rounded-2xl border border-border-subtle bg-surface px-4 py-4 text-left transition-colors hover:bg-surface-hover"
                            >
                                <div>
                                    <p className="text-sm font-semibold capitalize text-text-primary">{entry.status}</p>
                                    <p className="mt-1 text-xs text-text-muted">{new Date(entry.start_time).toLocaleString()}</p>
                                    <div className="mt-3 flex flex-wrap gap-2 text-xs text-text-secondary">
                                        <span className="rounded-full border border-border-subtle px-3 py-1">Steps {entry.current_step}/{entry.total_steps}</span>
                                        <span className="rounded-full border border-border-subtle px-3 py-1">Alerts {entry.alert_count}</span>
                                        <span className="rounded-full border border-border-subtle px-3 py-1">Predictions {entry.prediction_count}</span>
                                    </div>
                                </div>
                                <ChevronRight className="h-4 w-4 text-text-muted" />
                            </button>
                        ))}
                    </div>
                </div>

                <div className="card p-6">
                    <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Selected report</p>
                    {!selectedReport ? (
                        <div className="mt-5 rounded-2xl border border-border-subtle bg-surface px-4 py-5 text-sm text-text-secondary">
                            Select a session on the left to load the generated report summary.
                        </div>
                    ) : (
                        <div className="mt-5 space-y-4">
                            <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-4">
                                <p className="flex items-center gap-2 text-sm font-medium text-text-primary">
                                    <Clock3 className="h-4 w-4 text-accent" />
                                    Duration
                                </p>
                                <p className="mt-2 text-2xl font-semibold text-text-primary">{selectedReport.duration_steps} steps</p>
                            </div>
                            <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-4">
                                <p className="flex items-center gap-2 text-sm font-medium text-text-primary">
                                    <TrendingUp className="h-4 w-4 text-accent" />
                                    Risk summary
                                </p>
                                <div className="mt-3 space-y-1 text-sm text-text-secondary">
                                    <p>Average risk: {selectedReport.summary?.avg_risk}</p>
                                    <p>Peak risk: {selectedReport.summary?.max_risk}</p>
                                    <p>Peak step: {selectedReport.summary?.peak_risk_step}</p>
                                </div>
                            </div>
                            <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-4">
                                <p className="flex items-center gap-2 text-sm font-medium text-text-primary">
                                    <ShieldAlert className="h-4 w-4 text-accent" />
                                    Alert totals
                                </p>
                                <div className="mt-3 space-y-1 text-sm text-text-secondary">
                                    <p>Total alerts: {selectedReport.alerts?.total}</p>
                                    <p>Critical alerts: {selectedReport.alerts?.critical}</p>
                                    <p>Acknowledged: {selectedReport.alerts?.acknowledged}</p>
                                </div>
                            </div>
                            <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-4">
                                <p className="flex items-center gap-2 text-sm font-medium text-text-primary">
                                    <FileText className="h-4 w-4 text-accent" />
                                    Report metadata
                                </p>
                                <p className="mt-2 text-sm text-text-secondary">Generated at {new Date(selectedReport.generated_at).toLocaleString()}</p>
                            </div>
                        </div>
                    )}
                </div>
            </section>
        </PageShell>
    );
}
