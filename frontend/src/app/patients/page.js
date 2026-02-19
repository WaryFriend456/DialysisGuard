'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import PageShell from '@/components/PageShell';
import { useAuth } from '@/contexts/AuthContext';
import { patients as patientsApi } from '@/lib/api';
import { cn } from '@/lib/utils';
import {
    Search, UserPlus, Activity, Clock, BarChart3, X,
    Loader2, Users, AlertCircle, Monitor,
} from 'lucide-react';

const SEVERITY_STYLE = {
    Mild: 'badge-risk-low',
    Moderate: 'badge-risk-moderate',
    Severe: 'badge-risk-high',
    Critical: 'badge-risk-critical',
};

const INPUT_CLS = 'w-full rounded-lg border border-border-subtle bg-bg-secondary px-3 py-2 text-sm text-text-primary placeholder:text-text-muted transition-colors focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30';

export default function PatientsPage() {
    const { user, loading: authLoading } = useAuth();
    const router = useRouter();
    const [patients, setPatients] = useState([]);
    const [search, setSearch] = useState('');
    const [loading, setLoading] = useState(true);
    const [showAddModal, setShowAddModal] = useState(false);
    const [error, setError] = useState('');
    const [formData, setFormData] = useState({
        name: '', age: 55, gender: 'Male', weight: 75, diabetes: false, hypertension: false,
        kidney_failure_cause: 'Other', creatinine: 5.0, urea: 50.0, potassium: 4.5,
        hemoglobin: 11.0, hematocrit: 33.0, albumin: 3.8, dialysis_duration: 4.0,
        dialysis_frequency: 3, dialysate_composition: 'Standard', vascular_access_type: 'Fistula',
        dialyzer_type: 'High-flux', urine_output: 500, dry_weight: 70.0,
        fluid_removal_rate: 350, disease_severity: 'Moderate',
    });

    useEffect(() => {
        if (!authLoading && !user) router.push('/login');
    }, [user, authLoading, router]);

    useEffect(() => {
        if (user) loadPatients();
    }, [user, search]);

    const loadPatients = async () => {
        setLoading(true);
        try {
            const res = await patientsApi.list(`search=${search}&limit=50`);
            setPatients(res.patients || []);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const handleAdd = async (e) => {
        e.preventDefault();
        setError('');
        try {
            await patientsApi.create(formData);
            setShowAddModal(false);
            loadPatients();
        } catch (err) {
            setError(err.message);
        }
    };

    const update = (key, val) => setFormData((f) => ({ ...f, [key]: val }));

    if (authLoading || !user) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-bg-primary">
                <Loader2 className="h-8 w-8 animate-spin text-accent" />
            </div>
        );
    }

    return (
        <PageShell>
            {/* Header */}
            <div className="mb-6 flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-text-primary">Patients</h1>
                    <p className="mt-1 text-sm text-text-muted">
                        {patients.length} patient{patients.length !== 1 ? 's' : ''} registered
                    </p>
                </div>
                <button
                    onClick={() => setShowAddModal(true)}
                    className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-bg-primary transition-all hover:bg-accent-hover cursor-pointer"
                >
                    <UserPlus className="h-4 w-4" />
                    Add Patient
                </button>
            </div>

            {/* Search */}
            <div className="relative mb-6 max-w-md">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted" />
                <input
                    placeholder="Search patients…"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className={cn(INPUT_CLS, 'pl-9')}
                />
            </div>

            {/* Content */}
            {loading ? (
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                    {[...Array(6)].map((_, i) => (
                        <div key={i} className="skeleton h-44 rounded-xl" />
                    ))}
                </div>
            ) : patients.length === 0 ? (
                <div className="card flex flex-col items-center py-16 text-center">
                    <Users className="mb-3 h-12 w-12 text-text-muted/40" />
                    <p className="text-text-muted">No patients found. Add your first patient to get started.</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
                    {patients.map((p) => (
                        <div
                            key={p.id}
                            className="card card-hover cursor-pointer p-5"
                            onClick={() => router.push(`/monitor?patient=${p.id}`)}
                        >
                            <div className="flex items-start justify-between">
                                <div>
                                    <h3 className="text-sm font-semibold text-text-primary">
                                        {p.name || `Patient #${p.id?.slice(0, 6)}`}
                                    </h3>
                                    <p className="mt-0.5 text-xs text-text-muted">
                                        {p.age} yrs · {p.gender} · {p.weight} kg
                                    </p>
                                </div>
                                <span className={cn(
                                    'rounded-md px-2 py-0.5 text-[11px] font-semibold',
                                    SEVERITY_STYLE[p.disease_severity] || SEVERITY_STYLE.Moderate
                                )}>
                                    {p.disease_severity || 'Moderate'}
                                </span>
                            </div>

                            <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-text-secondary">
                                <span className="flex items-center gap-1.5">
                                    <Activity className="h-3 w-3 text-text-muted" />
                                    {p.vascular_access_type}
                                </span>
                                <span className="flex items-center gap-1.5">
                                    <BarChart3 className="h-3 w-3 text-text-muted" />
                                    {p.dialyzer_type}
                                </span>
                                <span className="flex items-center gap-1.5">
                                    <Clock className="h-3 w-3 text-text-muted" />
                                    {p.dialysis_duration}h
                                </span>
                                <span className="flex items-center gap-1.5">
                                    <span className="text-text-muted text-[10px]">×</span>
                                    {p.dialysis_frequency}x/wk
                                </span>
                            </div>

                            <div className="mt-4 flex gap-2">
                                <button
                                    className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-accent/10 px-3 py-1.5 text-xs font-medium text-accent transition-colors hover:bg-accent/20 cursor-pointer"
                                    onClick={(e) => { e.stopPropagation(); router.push(`/monitor?patient=${p.id}`); }}
                                >
                                    <Monitor className="h-3.5 w-3.5" />
                                    Monitor
                                </button>
                                <button
                                    className="flex items-center justify-center gap-1.5 rounded-lg border border-border-subtle px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:bg-surface-hover cursor-pointer"
                                    onClick={(e) => { e.stopPropagation(); router.push(`/patients/${p.id}/history`); }}
                                >
                                    <BarChart3 className="h-3.5 w-3.5" />
                                    History
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* ── Add Patient Modal ── */}
            {showAddModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
                    <div className="card w-full max-w-xl max-h-[85vh] overflow-y-auto p-8 animate-fade-in">
                        <div className="mb-6 flex items-center justify-between">
                            <h2 className="text-lg font-bold text-text-primary">Add New Patient</h2>
                            <button
                                onClick={() => setShowAddModal(false)}
                                className="rounded-lg p-1.5 text-text-muted transition-colors hover:bg-surface-hover hover:text-text-primary cursor-pointer"
                            >
                                <X className="h-5 w-5" />
                            </button>
                        </div>

                        {error && (
                            <div className="mb-4 flex items-center gap-2 rounded-lg border border-error/30 bg-error/10 px-4 py-2.5 text-sm text-error">
                                <AlertCircle className="h-4 w-4 shrink-0" /> {error}
                            </div>
                        )}

                        <form onSubmit={handleAdd}>
                            <div className="grid grid-cols-2 gap-4">
                                {/* Name */}
                                <div>
                                    <label className="mb-1 block text-xs font-medium text-text-secondary">Name</label>
                                    <input className={INPUT_CLS} value={formData.name} onChange={(e) => update('name', e.target.value)} required />
                                </div>
                                {/* Age */}
                                <div>
                                    <label className="mb-1 block text-xs font-medium text-text-secondary">Age</label>
                                    <input className={INPUT_CLS} type="number" value={formData.age} onChange={(e) => update('age', parseInt(e.target.value))} required />
                                </div>
                                {/* Gender */}
                                <div>
                                    <label className="mb-1 block text-xs font-medium text-text-secondary">Gender</label>
                                    <select className={cn(INPUT_CLS, 'appearance-none')} value={formData.gender} onChange={(e) => update('gender', e.target.value)}>
                                        <option>Male</option><option>Female</option>
                                    </select>
                                </div>
                                {/* Weight */}
                                <div>
                                    <label className="mb-1 block text-xs font-medium text-text-secondary">Weight (kg)</label>
                                    <input className={INPUT_CLS} type="number" step="0.1" value={formData.weight} onChange={(e) => update('weight', parseFloat(e.target.value))} />
                                </div>
                                {/* Severity */}
                                <div>
                                    <label className="mb-1 block text-xs font-medium text-text-secondary">Severity</label>
                                    <select className={cn(INPUT_CLS, 'appearance-none')} value={formData.disease_severity} onChange={(e) => update('disease_severity', e.target.value)}>
                                        <option>Mild</option><option>Moderate</option><option>Severe</option><option>Critical</option>
                                    </select>
                                </div>
                                {/* Kidney Failure Cause */}
                                <div>
                                    <label className="mb-1 block text-xs font-medium text-text-secondary">Kidney Failure Cause</label>
                                    <select className={cn(INPUT_CLS, 'appearance-none')} value={formData.kidney_failure_cause} onChange={(e) => update('kidney_failure_cause', e.target.value)}>
                                        <option>Diabetes</option><option>Hypertension</option><option>Glomerulonephritis</option><option>Polycystic</option><option>Other</option>
                                    </select>
                                </div>
                                {/* Creatinine */}
                                <div>
                                    <label className="mb-1 block text-xs font-medium text-text-secondary">Creatinine</label>
                                    <input className={INPUT_CLS} type="number" step="0.1" value={formData.creatinine} onChange={(e) => update('creatinine', parseFloat(e.target.value))} />
                                </div>
                                {/* Urea */}
                                <div>
                                    <label className="mb-1 block text-xs font-medium text-text-secondary">Urea</label>
                                    <input className={INPUT_CLS} type="number" step="0.1" value={formData.urea} onChange={(e) => update('urea', parseFloat(e.target.value))} />
                                </div>
                                {/* Fluid Removal */}
                                <div>
                                    <label className="mb-1 block text-xs font-medium text-text-secondary">Fluid Removal (ml/hr)</label>
                                    <input className={INPUT_CLS} type="number" value={formData.fluid_removal_rate} onChange={(e) => update('fluid_removal_rate', parseInt(e.target.value))} />
                                </div>
                                {/* Duration */}
                                <div>
                                    <label className="mb-1 block text-xs font-medium text-text-secondary">Duration (hrs)</label>
                                    <input className={INPUT_CLS} type="number" step="0.5" value={formData.dialysis_duration} onChange={(e) => update('dialysis_duration', parseFloat(e.target.value))} />
                                </div>
                            </div>

                            {/* Checkboxes */}
                            <div className="mt-4 flex gap-6">
                                <label className="flex items-center gap-2 text-sm text-text-secondary cursor-pointer">
                                    <input type="checkbox" className="accent-accent h-4 w-4 rounded" checked={formData.diabetes} onChange={(e) => update('diabetes', e.target.checked)} />
                                    Diabetes
                                </label>
                                <label className="flex items-center gap-2 text-sm text-text-secondary cursor-pointer">
                                    <input type="checkbox" className="accent-accent h-4 w-4 rounded" checked={formData.hypertension} onChange={(e) => update('hypertension', e.target.checked)} />
                                    Hypertension
                                </label>
                            </div>

                            {/* Actions */}
                            <div className="mt-6 flex justify-end gap-3">
                                <button
                                    type="button"
                                    onClick={() => setShowAddModal(false)}
                                    className="rounded-lg border border-border-subtle px-4 py-2 text-sm font-medium text-text-secondary transition-colors hover:bg-surface-hover cursor-pointer"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-bg-primary transition-all hover:bg-accent-hover cursor-pointer"
                                >
                                    Add Patient
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </PageShell>
    );
}
