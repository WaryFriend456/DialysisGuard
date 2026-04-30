'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { AlertCircle, KeyRound, Loader2 } from 'lucide-react';
import { dashboardPathForRole, useAuth } from '@/contexts/AuthContext';

const INPUT_CLS = 'w-full rounded-xl border border-border-subtle bg-bg-secondary px-3.5 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none';

export default function ChangePasswordPage() {
    const { user, loading, changePassword } = useAuth();
    const router = useRouter();
    const [currentPassword, setCurrentPassword] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        if (!loading && !user) router.push('/login');
    }, [loading, router, user]);

    const handleSubmit = async (event) => {
        event.preventDefault();
        setError('');
        if (newPassword !== confirmPassword) {
            setError('New passwords do not match');
            return;
        }
        setSaving(true);
        try {
            const currentUser = await changePassword(currentPassword, newPassword);
            router.push(dashboardPathForRole(currentUser.role));
        } catch (err) {
            setError(err.message || 'Unable to change password');
        } finally {
            setSaving(false);
        }
    };

    if (loading || !user) {
        return <div className="flex min-h-screen items-center justify-center bg-bg-primary"><Loader2 className="h-8 w-8 animate-spin text-accent" /></div>;
    }

    return (
        <div className="flex min-h-screen items-center justify-center bg-bg-primary px-4">
            <div className="card w-full max-w-md px-8 py-9">
                <div className="mb-7 text-center">
                    <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-accent/15">
                        <KeyRound className="h-6 w-6 text-accent" />
                    </div>
                    <h1 className="mt-4 text-2xl font-bold text-text-primary">Change password</h1>
                    <p className="mt-1 text-sm text-text-muted">Set a new password before entering the workspace.</p>
                </div>

                {error && (
                    <div className="mb-5 flex items-center gap-2 rounded-lg border border-error/30 bg-error/10 px-4 py-2.5 text-sm text-error">
                        <AlertCircle className="h-4 w-4 shrink-0" />
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-4">
                    <input
                        className={INPUT_CLS}
                        type="password"
                        placeholder="Current password"
                        value={currentPassword}
                        onChange={(event) => setCurrentPassword(event.target.value)}
                        required
                    />
                    <input
                        className={INPUT_CLS}
                        type="password"
                        placeholder="New password"
                        value={newPassword}
                        onChange={(event) => setNewPassword(event.target.value)}
                        minLength={6}
                        required
                    />
                    <input
                        className={INPUT_CLS}
                        type="password"
                        placeholder="Confirm new password"
                        value={confirmPassword}
                        onChange={(event) => setConfirmPassword(event.target.value)}
                        minLength={6}
                        required
                    />
                    <button
                        type="submit"
                        disabled={saving}
                        className="flex w-full justify-center rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-bg-primary disabled:opacity-50"
                    >
                        {saving ? 'Saving...' : 'Save password'}
                    </button>
                </form>
            </div>
        </div>
    );
}
