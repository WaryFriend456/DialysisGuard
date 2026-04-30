'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { AlertCircle, Building2, Loader2, Plus, Search } from 'lucide-react';
import PageShell from '@/components/PageShell';
import { useAuth } from '@/contexts/AuthContext';
import { admin } from '@/lib/api';
import { cn } from '@/lib/utils';

const INPUT_CLS = 'w-full rounded-2xl border border-border-subtle bg-bg-secondary px-4 py-3 text-sm text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none';
const emptyOrg = { name: '', code: '', address: '', phone: '', email: '' };

export default function OrganizationsPage() {
    const { user, loading: authLoading } = useAuth();
    const router = useRouter();
    const [organizations, setOrganizations] = useState([]);
    const [filter, setFilter] = useState('');
    const [showCreate, setShowCreate] = useState(false);
    const [form, setForm] = useState(emptyOrg);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    const loadOrganizations = async () => {
        setLoading(true);
        try {
            const res = await admin.listOrganizations();
            setOrganizations(res.organizations || []);
        } catch (err) {
            setError(err.message || 'Unable to load organizations');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (!authLoading && !user) router.push('/login');
        if (!authLoading && user && user.role !== 'super_admin') router.push('/');
    }, [authLoading, router, user]);

    useEffect(() => {
        if (user?.role === 'super_admin') loadOrganizations();
    }, [user]);

    const handleCreate = async (event) => {
        event.preventDefault();
        setError('');
        try {
            const created = await admin.createOrganization({
                ...form,
                email: form.email || undefined,
            });
            setForm(emptyOrg);
            setShowCreate(false);
            await loadOrganizations();
            router.push(`/admin/organizations/${created.id}`);
        } catch (err) {
            setError(err.message || 'Unable to create organization');
        }
    };

    if (authLoading || !user || loading) {
        return <div className="flex min-h-screen items-center justify-center bg-bg-primary"><Loader2 className="h-8 w-8 animate-spin text-accent" /></div>;
    }

    const visible = organizations.filter((org) => {
        const needle = filter.toLowerCase();
        return !needle || org.name?.toLowerCase().includes(needle) || org.code?.toLowerCase().includes(needle);
    });

    return (
        <PageShell>
            <section className="card p-6">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                        <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Platform administration</p>
                        <h2 className="mt-2 text-3xl font-semibold text-text-primary">Hospitals and organizations</h2>
                    </div>
                    <button
                        onClick={() => setShowCreate(true)}
                        className="inline-flex items-center gap-2 rounded-2xl bg-accent px-5 py-3 text-sm font-semibold text-bg-primary"
                    >
                        <Plus className="h-4 w-4" />
                        Create hospital
                    </button>
                </div>

                {error && (
                    <div className="mt-5 flex items-center gap-2 rounded-2xl border border-error/30 bg-error/10 px-4 py-3 text-sm text-error">
                        <AlertCircle className="h-4 w-4" />
                        {error}
                    </div>
                )}

                <label className="relative mt-6 block max-w-xl">
                    <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted" />
                    <input
                        className={cn(INPUT_CLS, 'pl-11')}
                        placeholder="Search hospitals"
                        value={filter}
                        onChange={(event) => setFilter(event.target.value)}
                    />
                </label>
            </section>

            <section className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {visible.map((org) => (
                    <article
                        key={org.id}
                        onClick={() => router.push(`/admin/organizations/${org.id}`)}
                        className="card card-hover cursor-pointer p-6"
                    >
                        <div className="flex items-start justify-between gap-3">
                            <div>
                                <Building2 className="mb-4 h-7 w-7 text-accent" />
                                <h3 className="text-lg font-semibold text-text-primary">{org.name}</h3>
                                <p className="mt-1 text-xs uppercase tracking-wide text-text-muted">{org.code}</p>
                            </div>
                            <span className={cn(
                                'rounded-full px-3 py-1 text-xs font-semibold',
                                org.status === 'active' ? 'bg-risk-low/10 text-risk-low' : 'bg-risk-critical-bg text-risk-critical'
                            )}>
                                {org.status}
                            </span>
                        </div>
                        <div className="mt-6 grid grid-cols-2 gap-3">
                            <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-3">
                                <p className="text-xs text-text-muted">Staff</p>
                                <p className="mt-1 text-xl font-semibold text-text-primary">{org.staff_count || 0}</p>
                            </div>
                            <div className="rounded-2xl border border-border-subtle bg-surface px-4 py-3">
                                <p className="text-xs text-text-muted">Patients</p>
                                <p className="mt-1 text-xl font-semibold text-text-primary">{org.patient_count || 0}</p>
                            </div>
                        </div>
                    </article>
                ))}
            </section>

            {showCreate && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 p-4 backdrop-blur-sm">
                    <form onSubmit={handleCreate} className="card w-full max-w-2xl p-6">
                        <h3 className="text-2xl font-semibold text-text-primary">Create hospital</h3>
                        <div className="mt-6 grid gap-4 md:grid-cols-2">
                            <input className={INPUT_CLS} placeholder="Hospital name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required />
                            <input className={INPUT_CLS} placeholder="Code" value={form.code} onChange={(e) => setForm({ ...form, code: e.target.value })} required />
                            <input className={INPUT_CLS} placeholder="Email" type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
                            <input className={INPUT_CLS} placeholder="Phone" value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} />
                            <input className={cn(INPUT_CLS, 'md:col-span-2')} placeholder="Address" value={form.address} onChange={(e) => setForm({ ...form, address: e.target.value })} />
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
