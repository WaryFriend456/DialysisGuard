'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { Activity, Clock3, Droplets, Loader2, MonitorDot, ShieldAlert, TrendingUp } from 'lucide-react';
import PageShell from '@/components/PageShell';
import { useAuth } from '@/contexts/AuthContext';
import { patients as patientsApi, sessions as sessionsApi } from '@/lib/api';
import { cn } from '@/lib/utils';

const severityClass = {
    Mild: 'badge-risk-low',
    Moderate: 'badge-risk-moderate',
    Severe: 'badge-risk-high',
    Critical: 'badge-risk-critical',
};

export default function PatientDetailPage() {
    const params = useParams();
    const router = useRouter();
    const { user, loading: authLoading } = useAuth();
    const [patient, setPatient] = useState(null);
    const [sessions, setSessions] = useState([]);
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
                    setSessions(sessionData.sessions || []);
                }
            } catch (err) {
                if (!cancelled) {
                    setError(err.message || 'Unable to load patient profile');
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

    if (authLoading || !user || loading) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-bg-primary">
                <Loader2 className="h-8 w-8 animate-spin text-accent" />
            </div>
        );
    }

    if (error || !patient) {
        return (
            <PageShell>
                <div className="card p-8">
                    <p className="text-lg font-semibold text-text-primary">Patient profile unavailable</p>
                    <p className="mt-2 text-sm text-text-secondary">{error || 'The selected patient record could not be loaded.'}</p>
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
                <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
                    <div>
                        <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Patient profile</p>
                        <h2 className="mt-2 text-3xl font-semibold text-text-primary">{patient.name || 'Unnamed patient'}</h2>
                        <p className="mt-2 text-sm text-text-secondary">
                            {patient.age} years · {patient.gender} · {patient.weight} kg · Dry weight {patient.dry_weight} kg
                        </p>
                    </div>
                    <div className="flex flex-wrap gap-3">
                        <button
                            onClick={() => router.push(`/monitor?patient=${patient.id}`)}
                            className="inline-flex items-center gap-2 rounded-2xl bg-accent px-5 py-3 text-sm font-semibold text-bg-primary"
                        >
                            <MonitorDot className="h-4 w-4" />
                            Open monitor
                        </button>
                        <button
                            onClick={() => router.push(`/patients/${patient.id}/history`)}
                            className="rounded-2xl border border-border-subtle px-5 py-3 text-sm font-medium text-text-primary hover:bg-surface-hover"
                        >
                            View history
                        </button>
                    </div>
                </div>
            </section>

            <section className="mt-6 grid gap-4 xl:grid-cols-4">
                <div className="card p-5">
                    <p className="text-xs uppercase tracking-wide text-text-muted">Severity</p>
                    <span className={cn('mt-3 inline-flex rounded-full px-3 py-1 text-xs font-semibold', severityClass[patient.disease_severity] || severityClass.Moderate)}>
                        {patient.disease_severity}
                    </span>
                </div>
                <div className="card p-5">
                    <p className="text-xs uppercase tracking-wide text-text-muted">Dialysis duration</p>
                    <p className="mt-3 text-2xl font-semibold text-text-primary">{patient.dialysis_duration} h</p>
                </div>
                <div className="card p-5">
                    <p className="text-xs uppercase tracking-wide text-text-muted">Frequency</p>
                    <p className="mt-3 text-2xl font-semibold text-text-primary">{patient.dialysis_frequency}/week</p>
                </div>
                <div className="card p-5">
                    <p className="text-xs uppercase tracking-wide text-text-muted">Fluid removal</p>
                    <p className="mt-3 text-2xl font-semibold text-text-primary">{patient.fluid_removal_rate} ml/h</p>
                </div>
            </section>

            <section className="mt-6 grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
                <div className="card p-6">
                    <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Clinical baseline</p>
                    <div className="mt-5 grid gap-3 md:grid-cols-2">
                        {[
                            ['Creatinine', patient.creatinine],
                            ['Urea', patient.urea],
                            ['Potassium', patient.potassium],
                            ['Hemoglobin', patient.hemoglobin],
                            ['Hematocrit', patient.hematocrit],
                            ['Albumin', patient.albumin],
                            ['Vascular access', patient.vascular_access_type],
                            ['Dialyzer', patient.dialyzer_type],
                        ].map(([label, value]) => (
                            <div key={label} className="rounded-2xl border border-border-subtle bg-surface px-4 py-4">
                                <p className="text-xs uppercase tracking-wide text-text-muted">{label}</p>
                                <p className="mt-2 text-base font-semibold text-text-primary">{value}</p>
                            </div>
                        ))}
                    </div>
                </div>

                <div className="card p-6">
                    <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Monitoring summary</p>
                    <div className="mt-5 space-y-3">
                        <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-4">
                            <p className="flex items-center gap-2 text-sm font-medium text-text-primary">
                                <Clock3 className="h-4 w-4 text-accent" />
                                Recorded sessions
                            </p>
                            <p className="mt-2 text-3xl font-semibold text-text-primary">{sessions.length}</p>
                        </div>
                        <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-4">
                            <p className="flex items-center gap-2 text-sm font-medium text-text-primary">
                                <Activity className="h-4 w-4 text-accent" />
                                Latest status
                            </p>
                            <p className="mt-2 text-base font-semibold capitalize text-text-primary">{sessions[0]?.status || 'No sessions yet'}</p>
                        </div>
                        <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-4">
                            <p className="flex items-center gap-2 text-sm font-medium text-text-primary">
                                <ShieldAlert className="h-4 w-4 text-accent" />
                                Hypertension
                            </p>
                            <p className="mt-2 text-base font-semibold text-text-primary">{patient.hypertension ? 'Present' : 'Not reported'}</p>
                        </div>
                        <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-4">
                            <p className="flex items-center gap-2 text-sm font-medium text-text-primary">
                                <Droplets className="h-4 w-4 text-accent" />
                                Diabetes
                            </p>
                            <p className="mt-2 text-base font-semibold text-text-primary">{patient.diabetes ? 'Present' : 'Not reported'}</p>
                        </div>
                    </div>
                </div>
            </section>

            <section className="mt-6 card p-6">
                <div className="flex items-center justify-between">
                    <div>
                        <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Recent sessions</p>
                        <h3 className="mt-2 text-xl font-semibold text-text-primary">Monitoring history preview</h3>
                    </div>
                    <button
                        onClick={() => router.push(`/patients/${patient.id}/history`)}
                        className="rounded-2xl border border-border-subtle px-4 py-3 text-sm font-medium text-text-primary hover:bg-surface-hover"
                    >
                        Full history
                    </button>
                </div>

                <div className="mt-5 space-y-3">
                    {sessions.length === 0 && (
                        <p className="text-sm text-text-secondary">No prior sessions recorded for this patient.</p>
                    )}
                    {sessions.slice(0, 5).map((entry) => (
                        <div key={entry.id} className="rounded-2xl border border-border-subtle bg-surface px-4 py-4">
                            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                                <div>
                                    <p className="text-sm font-semibold capitalize text-text-primary">{entry.status}</p>
                                    <p className="mt-1 text-xs text-text-muted">{new Date(entry.start_time).toLocaleString()}</p>
                                </div>
                                <div className="flex flex-wrap gap-2 text-xs text-text-secondary">
                                    <span className="rounded-full border border-border-subtle px-3 py-1">Steps {entry.current_step}/{entry.total_steps}</span>
                                    <span className="rounded-full border border-border-subtle px-3 py-1">Predictions {entry.prediction_count}</span>
                                    <span className="rounded-full border border-border-subtle px-3 py-1">Alerts {entry.alert_count}</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </section>
        </PageShell>
    );
}
