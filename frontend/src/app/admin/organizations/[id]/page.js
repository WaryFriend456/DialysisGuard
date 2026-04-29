'use client';

import { useCallback, useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { AlertCircle, Copy, KeyRound, Loader2, UserPlus } from 'lucide-react';
import PageShell from '@/components/PageShell';
import { useAuth } from '@/contexts/AuthContext';
import { admin } from '@/lib/api';
import { cn } from '@/lib/utils';

const INPUT_CLS = 'w-full rounded-2xl border border-border-subtle bg-bg-secondary px-4 py-3 text-sm text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none';
const emptyAdmin = { name: '', email: '', role: 'org_admin' };

export default function OrganizationDetailPage() {
    const { user, loading: authLoading } = useAuth();
    const router = useRouter();
    const params = useParams();
    const orgId = params.id;
    const [organization, setOrganization] = useState(null);
    const [users, setUsers] = useState([]);
    const [showCreate, setShowCreate] = useState(false);
    const [form, setForm] = useState(emptyAdmin);
    const [temporaryPassword, setTemporaryPassword] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    const loadData = useCallback(async () => {
        setLoading(true);
        try {
            const [org, userRes] = await Promise.all([
                admin.getOrganization(orgId),
                admin.listOrganizationUsers(orgId),
            ]);
            setOrganization(org);
            setUsers(userRes.users || []);
            setError('');
        } catch (err) {
            setError(err.message || 'Unable to load organization');
        } finally {
            setLoading(false);
        }
    }, [orgId]);

    useEffect(() => {
        if (!authLoading && !user) router.push('/login');
        if (!authLoading && user && user.role !== 'super_admin') router.push('/');
    }, [authLoading, router, user]);

    useEffect(() => {
        if (user?.role === 'super_admin' && orgId) loadData();
    }, [user, orgId, loadData]);

    const createOrgAdmin = async (event) => {
        event.preventDefault();
        setError('');
        try {
            const res = await admin.createOrgAdmin(orgId, form);
            setTemporaryPassword(res.temporary_password);
            setForm(emptyAdmin);
            setShowCreate(false);
            await loadData();
        } catch (err) {
            setError(err.message || 'Unable to create org admin');
        }
    };

    const toggleOrgStatus = async () => {
        if (!organization) return;
        try {
            if (organization.status === 'active') {
                await admin.suspendOrganization(orgId);
            } else {
                await admin.activateOrganization(orgId);
            }
            await loadData();
        } catch (err) {
            setError(err.message || 'Unable to update organization status');
        }
    };

    const resetPassword = async (userId) => {
        try {
            const res = await admin.resetPassword(userId);
            setTemporaryPassword(res.temporary_password);
            await loadData();
        } catch (err) {
            setError(err.message || 'Unable to reset password');
        }
    };

    const toggleUserStatus = async (target) => {
        try {
            if (target.status === 'disabled') await admin.activateUser(target.id);
            else await admin.disableUser(target.id);
            await loadData();
        } catch (err) {
            setError(err.message || 'Unable to update user');
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
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                    <div>
                        <p className="text-xs uppercase tracking-[0.2em] text-text-muted">{organization?.code}</p>
                        <h2 className="mt-2 text-3xl font-semibold text-text-primary">{organization?.name}</h2>
                        <p className="mt-2 text-sm text-text-secondary">{organization?.address || 'No address recorded'}</p>
                    </div>
                    <div className="flex gap-2">
                        <button onClick={toggleOrgStatus} className={cn(
                            'rounded-2xl px-5 py-3 text-sm font-semibold',
                            organization?.status === 'active' ? 'border border-risk-critical/30 bg-risk-critical-bg text-risk-critical' : 'bg-accent text-bg-primary'
                        )}>
                            {organization?.status === 'active' ? 'Suspend' : 'Activate'}
                        </button>
                        <button onClick={() => setShowCreate(true)} className="inline-flex items-center gap-2 rounded-2xl bg-accent px-5 py-3 text-sm font-semibold text-bg-primary">
                            <UserPlus className="h-4 w-4" />
                            Add org admin
                        </button>
                    </div>
                </div>
            </section>

            <section className="card mt-6 p-6">
                <h3 className="text-lg font-semibold text-text-primary">Organization users</h3>
                <div className="mt-5 space-y-3">
                    {users.map((item) => (
                        <div key={item.id} className="flex flex-col gap-3 rounded-2xl border border-border-subtle bg-surface px-4 py-3 lg:flex-row lg:items-center">
                            <div className="flex-1">
                                <p className="font-medium text-text-primary">{item.name}</p>
                                <p className="text-sm text-text-muted">{item.email} · {item.role}</p>
                            </div>
                            <span className={cn('rounded-full px-3 py-1 text-xs font-semibold', item.status === 'active' ? 'bg-risk-low/10 text-risk-low' : 'bg-risk-critical-bg text-risk-critical')}>
                                {item.status}
                            </span>
                            <button onClick={() => resetPassword(item.id)} className="inline-flex items-center gap-2 rounded-xl border border-border-subtle px-3 py-2 text-sm text-text-secondary">
                                <KeyRound className="h-4 w-4" />
                                Reset
                            </button>
                            <button onClick={() => toggleUserStatus(item)} className="rounded-xl border border-border-subtle px-3 py-2 text-sm text-text-secondary">
                                {item.status === 'disabled' ? 'Activate' : 'Disable'}
                            </button>
                        </div>
                    ))}
                    {users.length === 0 && <p className="text-sm text-text-muted">No users have been created for this hospital.</p>}
                </div>
            </section>

            {showCreate && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 p-4 backdrop-blur-sm">
                    <form onSubmit={createOrgAdmin} className="card w-full max-w-lg p-6">
                        <h3 className="text-2xl font-semibold text-text-primary">Create hospital admin</h3>
                        <div className="mt-6 space-y-4">
                            <input className={INPUT_CLS} placeholder="Full name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} required />
                            <input className={INPUT_CLS} placeholder="Email" type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} required />
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
