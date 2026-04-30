'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { AlertCircle, Copy, KeyRound, Loader2, Plus, UserRoundCog } from 'lucide-react';
import PageShell from '@/components/PageShell';
import { useAuth } from '@/contexts/AuthContext';
import { orgAdmin } from '@/lib/api';
import { cn } from '@/lib/utils';

const INPUT_CLS = 'w-full rounded-2xl border border-border-subtle bg-bg-secondary px-4 py-3 text-sm text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none';
const emptyStaff = { name: '', email: '', role: 'doctor' };

export default function StaffPage() {
    const { user, loading: authLoading } = useAuth();
    const router = useRouter();
    const [summary, setSummary] = useState(null);
    const [staff, setStaff] = useState([]);
    const [showCreate, setShowCreate] = useState(false);
    const [form, setForm] = useState(emptyStaff);
    const [temporaryPassword, setTemporaryPassword] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    const loadData = async () => {
        setLoading(true);
        try {
            const [summaryRes, staffRes] = await Promise.all([orgAdmin.summary(), orgAdmin.listStaff()]);
            setSummary(summaryRes);
            setStaff(staffRes.users || []);
            setError('');
        } catch (err) {
            setError(err.message || 'Unable to load staff');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (!authLoading && !user) router.push('/login');
        if (!authLoading && user && user.role !== 'org_admin') router.push('/');
    }, [authLoading, router, user]);

    useEffect(() => {
        if (user?.role === 'org_admin') loadData();
    }, [user]);

    const createStaff = async (event) => {
        event.preventDefault();
        setError('');
        try {
            const res = await orgAdmin.createStaff(form);
            setTemporaryPassword(res.temporary_password);
            setForm(emptyStaff);
            setShowCreate(false);
            await loadData();
        } catch (err) {
            setError(err.message || 'Unable to create staff user');
        }
    };

    const resetPassword = async (staffId) => {
        try {
            const res = await orgAdmin.resetPassword(staffId);
            setTemporaryPassword(res.temporary_password);
            await loadData();
        } catch (err) {
            setError(err.message || 'Unable to reset password');
        }
    };

    const toggleStatus = async (item) => {
        try {
            if (item.status === 'disabled') await orgAdmin.activateStaff(item.id);
            else await orgAdmin.disableStaff(item.id);
            await loadData();
        } catch (err) {
            setError(err.message || 'Unable to update staff user');
        }
    };

    if (authLoading || !user || loading) {
        return <div className="flex min-h-screen items-center justify-center bg-bg-primary"><Loader2 className="h-8 w-8 animate-spin text-accent" /></div>;
    }

    return (
        <PageShell>
            {error && (
                <div className="mb-5 flex items-center gap-2 rounded-2xl border border-error/30 bg-error/10 px-4 py-3 text-sm text-error">
                    <AlertCircle className="h-4 w-4" />
                    {error}
                </div>
            )}

            {temporaryPassword && (
                <div className="mb-5 rounded-2xl border border-accent/30 bg-accent/10 px-5 py-4">
                    <p className="text-sm font-semibold text-text-primary">Temporary password</p>
                    <div className="mt-2 flex items-center justify-between gap-3 rounded-xl bg-bg-secondary px-4 py-3">
                        <code className="text-sm text-accent">{temporaryPassword}</code>
                        <button onClick={() => navigator.clipboard?.writeText(temporaryPassword)} className="rounded-lg p-2 text-text-muted hover:bg-surface-hover">
                            <Copy className="h-4 w-4" />
                        </button>
                    </div>
                </div>
            )}

            <section className="card p-6">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                        <p className="text-xs uppercase tracking-[0.2em] text-text-muted">{summary?.organization?.name}</p>
                        <h2 className="mt-2 text-3xl font-semibold text-text-primary">Doctors and nurses</h2>
                    </div>
                    <button onClick={() => setShowCreate(true)} className="inline-flex items-center gap-2 rounded-2xl bg-accent px-5 py-3 text-sm font-semibold text-bg-primary">
                        <Plus className="h-4 w-4" />
                        Create staff
                    </button>
                </div>
                <div className="mt-6 grid gap-3 sm:grid-cols-4">
                    {[
                        ['Staff', summary?.staff_count || 0],
                        ['Doctors', summary?.doctor_count || 0],
                        ['Nurses', summary?.nurse_count || 0],
                        ['Patients', summary?.patient_count || 0],
                    ].map(([label, value]) => (
                        <div key={label} className="rounded-2xl border border-border-subtle bg-surface px-4 py-3">
                            <p className="text-xs text-text-muted">{label}</p>
                            <p className="mt-1 text-2xl font-semibold text-text-primary">{value}</p>
                        </div>
                    ))}
                </div>
            </section>

            <section className="mt-6 space-y-3">
                {staff.map((item) => (
                    <div key={item.id} className="card flex flex-col gap-3 p-4 lg:flex-row lg:items-center">
                        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-accent/10 text-accent">
                            <UserRoundCog className="h-5 w-5" />
                        </div>
                        <div className="flex-1">
                            <p className="font-medium text-text-primary">{item.name}</p>
                            <p className="text-sm text-text-muted">{item.email} · {item.role === 'nurse' ? 'Nurse' : 'Doctor'}</p>
                        </div>
                        <span className={cn('rounded-full px-3 py-1 text-xs font-semibold', item.status === 'active' ? 'bg-risk-low/10 text-risk-low' : 'bg-risk-critical-bg text-risk-critical')}>
                            {item.status}
                        </span>
                        <button onClick={() => resetPassword(item.id)} className="inline-flex items-center gap-2 rounded-xl border border-border-subtle px-3 py-2 text-sm text-text-secondary">
                            <KeyRound className="h-4 w-4" />
                            Reset
                        </button>
                        <button onClick={() => toggleStatus(item)} className="rounded-xl border border-border-subtle px-3 py-2 text-sm text-text-secondary">
                            {item.status === 'disabled' ? 'Activate' : 'Disable'}
                        </button>
                    </div>
                ))}
                {staff.length === 0 && <div className="card p-8 text-center text-sm text-text-muted">No doctors or nurses have been created yet.</div>}
            </section>

            {showCreate && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 p-4 backdrop-blur-sm">
                    <form onSubmit={createStaff} className="card w-full max-w-lg p-6">
                        <h3 className="text-2xl font-semibold text-text-primary">Create staff account</h3>
                        <div className="mt-6 space-y-4">
                            <input className={INPUT_CLS} placeholder="Full name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required />
                            <input className={INPUT_CLS} placeholder="Email" type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} required />
                            <select className={INPUT_CLS} value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
                                <option value="doctor">Doctor</option>
                                <option value="nurse">Nurse</option>
                            </select>
                        </div>
                        <div className="mt-6 flex justify-end gap-3">
                            <button type="button" onClick={() => setShowCreate(false)} className="rounded-2xl border border-border-subtle px-5 py-3 text-sm text-text-secondary">Cancel</button>
                            <button type="submit" className="rounded-2xl bg-accent px-5 py-3 text-sm font-semibold text-bg-primary">Create</button>
                        </div>
                    </form>
                </div>
            )}
        </PageShell>
    );
}
