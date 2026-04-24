'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Activity, AlertCircle, ArrowRight } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

const INPUT_CLS =
    'w-full rounded-xl border border-border-subtle bg-bg-secondary px-3.5 py-2.5 text-sm text-text-primary placeholder:text-text-muted transition-colors focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30';

export default function LoginPage() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { login } = useAuth();
    const router = useRouter();

    const handleSubmit = async (event) => {
        event.preventDefault();
        setError('');
        setLoading(true);
        try {
            const user = await login(email, password);
            router.push(user.role === 'doctor' ? '/dashboard/doctor' : '/dashboard/caregiver');
        } catch (err) {
            setError(err.message || 'Login failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-bg-primary px-4 py-8">
            <div className="pointer-events-none absolute top-1/4 left-1/6 h-56 w-56 rounded-full bg-accent/5 blur-[100px]" />
            <div className="pointer-events-none absolute right-1/6 bottom-1/4 h-56 w-56 rounded-full bg-chart-3/5 blur-[100px]" />

            <div className="card w-full max-w-md px-8 py-9 text-center backdrop-blur-xl animate-fade-in sm:px-10 sm:py-10">
                <div className="mb-8 flex flex-col items-center gap-3">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-accent/15 shadow-glow-accent">
                        <Activity className="h-6 w-6 text-accent" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-extrabold tracking-tight text-text-primary">DialysisGuard</h1>
                        <p className="mt-1 text-sm text-text-muted">Clinical workspace sign in</p>
                    </div>
                </div>

                {error && (
                    <div className="mb-5 flex items-center gap-2 rounded-lg border border-error/30 bg-error/10 px-4 py-2.5 text-sm text-error">
                        <AlertCircle className="h-4 w-4 shrink-0" />
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-4 text-left">
                    <div>
                        <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-text-secondary">
                            Email
                        </label>
                        <input
                            type="email"
                            value={email}
                            onChange={(event) => setEmail(event.target.value)}
                            placeholder="doctor@hospital.com"
                            required
                            className={INPUT_CLS}
                        />
                    </div>
                    <div>
                        <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-text-secondary">
                            Password
                        </label>
                        <input
                            type="password"
                            value={password}
                            onChange={(event) => setPassword(event.target.value)}
                            placeholder="••••••••"
                            required
                            className={INPUT_CLS}
                        />
                    </div>
                    <button
                        type="submit"
                        disabled={loading}
                        className="mt-2 flex w-full items-center justify-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-bg-primary transition-all hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        {loading ? 'Signing in...' : 'Sign In'}
                        {!loading && <ArrowRight className="h-4 w-4" />}
                    </button>
                </form>

                <p className="mt-6 text-sm text-text-muted">
                    Don&apos;t have an account?{' '}
                    <Link href="/register" className="font-semibold text-accent transition-colors hover:text-accent-hover">
                        Register
                    </Link>
                </p>
            </div>
        </div>
    );
}
