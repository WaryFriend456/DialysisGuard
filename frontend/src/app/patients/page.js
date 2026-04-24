'use client';

import { Suspense, useCallback, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
    Activity,
    AlertCircle,
    Clock3,
    Loader2,
    MonitorDot,
    Plus,
    Search,
    UserRound,
    X,
} from 'lucide-react';
import PageShell from '@/components/PageShell';
import { patients as patientsApi } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';
import { cn } from '@/lib/utils';

const INPUT_CLS = 'w-full rounded-2xl border border-border-subtle bg-bg-secondary px-4 py-3 text-sm text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none';
const severityClass = {
    Mild: 'badge-risk-low',
    Moderate: 'badge-risk-moderate',
    Severe: 'badge-risk-high',
    Critical: 'badge-risk-critical',
};

const defaultForm = {
    name: '',
    age: 55,
    gender: 'Male',
    weight: 75,
    diabetes: false,
    hypertension: false,
    kidney_failure_cause: 'Other',
    creatinine: 5.0,
    urea: 50.0,
    potassium: 4.5,
    hemoglobin: 11.0,
    hematocrit: 33.0,
    albumin: 3.8,
    dialysis_duration: 4.0,
    dialysis_frequency: 3,
    dialysate_composition: 'Standard',
    vascular_access_type: 'Fistula',
    dialyzer_type: 'High-flux',
    urine_output: 500,
    dry_weight: 70.0,
    fluid_removal_rate: 350,
    disease_severity: 'Moderate',
};

function PatientsContent() {
    const { user, loading: authLoading } = useAuth();
    const router = useRouter();
    const searchParams = useSearchParams();
    const [patients, setPatients] = useState([]);
    const [search, setSearch] = useState('');
    const [loading, setLoading] = useState(true);
    const [showAddModal, setShowAddModal] = useState(false);
    const [error, setError] = useState('');
    const [formData, setFormData] = useState(defaultForm);

    const loadPatients = useCallback(async () => {
        setLoading(true);
        try {
            const res = await patientsApi.list(`search=${encodeURIComponent(search)}&limit=50`);
            setPatients(res.patients || []);
        } catch (err) {
            setError(err.message || 'Unable to load patients');
        } finally {
            setLoading(false);
        }
    }, [search]);

    useEffect(() => {
        if (!authLoading && !user) {
            router.push('/login');
        }
    }, [authLoading, router, user]);

    useEffect(() => {
        if (user) {
            loadPatients();
        }
    }, [loadPatients, user]);

    useEffect(() => {
        if (searchParams.get('action') === 'add') {
            setShowAddModal(true);
        }
    }, [searchParams]);

    const update = (key, value) => {
        setFormData((current) => ({ ...current, [key]: value }));
    };

    const handleAdd = async (event) => {
        event.preventDefault();
        setError('');
        try {
            await patientsApi.create(formData);
            setFormData(defaultForm);
            setShowAddModal(false);
            await loadPatients();
        } catch (err) {
            setError(err.message || 'Unable to create patient');
        }
    };

    if (authLoading || !user) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-bg-primary">
                <Loader2 className="h-8 w-8 animate-spin text-accent" />
            </div>
        );
    }

    return (
        <PageShell>
            <section className="card p-6">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                        <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Patient registry</p>
                        <h2 className="mt-2 text-3xl font-semibold text-text-primary">Clinical profiles and monitoring entry point</h2>
                        <p className="mt-2 max-w-2xl text-sm text-text-secondary">
                            Every patient card now routes to a real detail page, with history and monitoring available from there.
                        </p>
                    </div>
                    <button
                        onClick={() => setShowAddModal(true)}
                        className="inline-flex items-center gap-2 rounded-2xl bg-accent px-5 py-3 text-sm font-semibold text-bg-primary"
                    >
                        <Plus className="h-4 w-4" />
                        Add patient
                    </button>
                </div>

                <div className="mt-6 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <label className="relative block w-full max-w-xl">
                        <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted" />
                        <input
                            value={search}
                            onChange={(event) => setSearch(event.target.value)}
                            placeholder="Search by patient name or demographic"
                            className={cn(INPUT_CLS, 'pl-11')}
                        />
                    </label>
                    <p className="text-sm text-text-secondary">{patients.length} patient records loaded</p>
                </div>
            </section>

            <section className="mt-6">
                {loading ? (
                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                        {Array.from({ length: 6 }).map((_, index) => (
                            <div key={index} className="skeleton h-56 rounded-3xl" />
                        ))}
                    </div>
                ) : patients.length === 0 ? (
                    <div className="card flex flex-col items-center py-16 text-center">
                        <UserRound className="mb-4 h-12 w-12 text-text-muted/40" />
                        <p className="text-lg font-medium text-text-primary">No patients found</p>
                        <p className="mt-2 max-w-md text-sm text-text-secondary">Create the first patient profile to unlock monitoring, history, and alert flows.</p>
                    </div>
                ) : (
                    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                        {patients.map((patient) => (
                            <article
                                key={patient.id}
                                className="card card-hover cursor-pointer p-6"
                                onClick={() => router.push(`/patients/${patient.id}`)}
                            >
                                <div className="flex items-start justify-between gap-3">
                                    <div>
                                        <h3 className="text-lg font-semibold text-text-primary">{patient.name || `Patient ${patient.id?.slice(0, 6)}`}</h3>
                                        <p className="mt-1 text-sm text-text-secondary">{patient.age} years · {patient.gender} · {patient.weight} kg</p>
                                    </div>
                                    <span className={cn('rounded-full px-3 py-1 text-xs font-semibold', severityClass[patient.disease_severity] || severityClass.Moderate)}>
                                        {patient.disease_severity || 'Moderate'}
                                    </span>
                                </div>

                                <div className="mt-5 grid grid-cols-2 gap-3 text-sm text-text-secondary">
                                    <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-3">
                                        <p className="text-xs uppercase tracking-wide text-text-muted">Access</p>
                                        <p className="mt-1 font-medium text-text-primary">{patient.vascular_access_type}</p>
                                    </div>
                                    <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-3">
                                        <p className="text-xs uppercase tracking-wide text-text-muted">Dialyzer</p>
                                        <p className="mt-1 font-medium text-text-primary">{patient.dialyzer_type}</p>
                                    </div>
                                    <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-3">
                                        <p className="text-xs uppercase tracking-wide text-text-muted">Duration</p>
                                        <p className="mt-1 font-medium text-text-primary">{patient.dialysis_duration} h</p>
                                    </div>
                                    <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-3">
                                        <p className="text-xs uppercase tracking-wide text-text-muted">Frequency</p>
                                        <p className="mt-1 font-medium text-text-primary">{patient.dialysis_frequency}/week</p>
                                    </div>
                                </div>

                                <div className="mt-5 flex gap-2">
                                    <button
                                        onClick={(event) => {
                                            event.stopPropagation();
                                            router.push(`/patients/${patient.id}`);
                                        }}
                                        className="flex-1 rounded-2xl border border-border-subtle px-4 py-3 text-sm font-medium text-text-primary hover:bg-surface-hover"
                                    >
                                        View profile
                                    </button>
                                    <button
                                        onClick={(event) => {
                                            event.stopPropagation();
                                            router.push(`/monitor?patient=${patient.id}`);
                                        }}
                                        className="inline-flex items-center gap-2 rounded-2xl bg-accent px-4 py-3 text-sm font-semibold text-bg-primary"
                                    >
                                        <MonitorDot className="h-4 w-4" />
                                        Monitor
                                    </button>
                                </div>
                            </article>
                        ))}
                    </div>
                )}
            </section>

            {showAddModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 p-4 backdrop-blur-sm">
                    <div className="card max-h-[90vh] w-full max-w-3xl overflow-y-auto p-6">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs uppercase tracking-[0.2em] text-text-muted">New patient</p>
                                <h3 className="mt-2 text-2xl font-semibold text-text-primary">Create clinical baseline</h3>
                            </div>
                            <button
                                onClick={() => setShowAddModal(false)}
                                className="rounded-2xl p-2 text-text-muted hover:bg-surface-hover hover:text-text-primary"
                            >
                                <X className="h-5 w-5" />
                            </button>
                        </div>

                        {error && (
                            <div className="mt-4 flex items-center gap-3 rounded-2xl border border-error/30 bg-error/10 px-4 py-3 text-sm text-error">
                                <AlertCircle className="h-4 w-4" />
                                <span>{error}</span>
                            </div>
                        )}

                        <form onSubmit={handleAdd} className="mt-6 space-y-6">
                            <div className="grid gap-4 md:grid-cols-2">
                                <div>
                                    <label className="mb-2 block text-xs uppercase tracking-wide text-text-muted">Name</label>
                                    <input className={INPUT_CLS} value={formData.name} onChange={(event) => update('name', event.target.value)} required />
                                </div>
                                <div>
                                    <label className="mb-2 block text-xs uppercase tracking-wide text-text-muted">Age</label>
                                    <input className={INPUT_CLS} type="number" value={formData.age} onChange={(event) => update('age', Number(event.target.value))} required />
                                </div>
                                <div>
                                    <label className="mb-2 block text-xs uppercase tracking-wide text-text-muted">Gender</label>
                                    <select className={INPUT_CLS} value={formData.gender} onChange={(event) => update('gender', event.target.value)}>
                                        <option>Male</option>
                                        <option>Female</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="mb-2 block text-xs uppercase tracking-wide text-text-muted">Weight (kg)</label>
                                    <input className={INPUT_CLS} type="number" step="0.1" value={formData.weight} onChange={(event) => update('weight', Number(event.target.value))} />
                                </div>
                                <div>
                                    <label className="mb-2 block text-xs uppercase tracking-wide text-text-muted">Disease severity</label>
                                    <select className={INPUT_CLS} value={formData.disease_severity} onChange={(event) => update('disease_severity', event.target.value)}>
                                        <option>Mild</option>
                                        <option>Moderate</option>
                                        <option>Severe</option>
                                        <option>Critical</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="mb-2 block text-xs uppercase tracking-wide text-text-muted">Kidney failure cause</label>
                                    <select className={INPUT_CLS} value={formData.kidney_failure_cause} onChange={(event) => update('kidney_failure_cause', event.target.value)}>
                                        <option>Diabetes</option>
                                        <option>Hypertension</option>
                                        <option>Glomerulonephritis</option>
                                        <option>Polycystic</option>
                                        <option>Other</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="mb-2 block text-xs uppercase tracking-wide text-text-muted">Creatinine</label>
                                    <input className={INPUT_CLS} type="number" step="0.1" value={formData.creatinine} onChange={(event) => update('creatinine', Number(event.target.value))} />
                                </div>
                                <div>
                                    <label className="mb-2 block text-xs uppercase tracking-wide text-text-muted">Urea</label>
                                    <input className={INPUT_CLS} type="number" step="0.1" value={formData.urea} onChange={(event) => update('urea', Number(event.target.value))} />
                                </div>
                                <div>
                                    <label className="mb-2 block text-xs uppercase tracking-wide text-text-muted">Dialysis duration (hours)</label>
                                    <input className={INPUT_CLS} type="number" step="0.5" value={formData.dialysis_duration} onChange={(event) => update('dialysis_duration', Number(event.target.value))} />
                                </div>
                                <div>
                                    <label className="mb-2 block text-xs uppercase tracking-wide text-text-muted">Fluid removal rate (ml/hour)</label>
                                    <input className={INPUT_CLS} type="number" value={formData.fluid_removal_rate} onChange={(event) => update('fluid_removal_rate', Number(event.target.value))} />
                                </div>
                            </div>

                            <div className="grid gap-3 sm:grid-cols-2">
                                <label className="flex items-center gap-3 rounded-2xl border border-border-subtle bg-surface px-4 py-3 text-sm text-text-primary">
                                    <input type="checkbox" checked={formData.diabetes} onChange={(event) => update('diabetes', event.target.checked)} />
                                    Diabetes
                                </label>
                                <label className="flex items-center gap-3 rounded-2xl border border-border-subtle bg-surface px-4 py-3 text-sm text-text-primary">
                                    <input type="checkbox" checked={formData.hypertension} onChange={(event) => update('hypertension', event.target.checked)} />
                                    Hypertension
                                </label>
                            </div>

                            <div className="flex flex-col gap-3 sm:flex-row sm:justify-end">
                                <button
                                    type="button"
                                    onClick={() => setShowAddModal(false)}
                                    className="rounded-2xl border border-border-subtle px-5 py-3 text-sm font-medium text-text-secondary hover:bg-surface-hover"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    className="rounded-2xl bg-accent px-5 py-3 text-sm font-semibold text-bg-primary"
                                >
                                    Save patient
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </PageShell>
    );
}

export default function PatientsPage() {
    return (
        <Suspense
            fallback={
                <div className="flex min-h-screen items-center justify-center bg-bg-primary">
                    <Loader2 className="h-8 w-8 animate-spin text-accent" />
                </div>
            }
        >
            <PatientsContent />
        </Suspense>
    );
}
