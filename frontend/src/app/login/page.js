'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';
import { Activity, ArrowRight, AlertCircle } from 'lucide-react';

export default function LoginPage() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { login } = useAuth();
    const router = useRouter();

    const handleSubmit = async (e) => {
        e.preventDefault();
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
        <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-bg-primary">
            {/* Ambient glow */}
            <div className="pointer-events-none absolute top-1/4 left-1/6 h-72 w-72 rounded-full bg-accent/5 blur-[80px]" />
            <div className="pointer-events-none absolute right-1/6 bottom-1/4 h-96 w-96 rounded-full bg-chart-5/4 blur-[100px]" />

            <div className="card w-full max-w-md px-10 py-10 text-center backdrop-blur-xl animate-fade-in">
                {/* Brand */}
                <div className="mb-8 flex flex-col items-center gap-3">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-accent/15 shadow-glow-accent">
                        <Activity className="h-6 w-6 text-accent" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-extrabold tracking-tight text-text-primary">
                            DialysisGuard
                        </h1>
                        <p className="mt-1 text-sm text-text-muted">
                            AI-Driven Hemodialysis Monitoring
                        </p>
                    </div>
                </div>

                {/* Error */}
                {error && (
                    <div className="mb-5 flex items-center gap-2 rounded-lg border border-error/30 bg-error/10 px-4 py-2.5 text-sm text-error">
                        <AlertCircle className="h-4 w-4 shrink-0" />
                        {error}
                    </div>
                )}

                {/* Form */}
                <form onSubmit={handleSubmit} className="space-y-4 text-left">
                    <div>
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">
                            Email
                        </label>
                        <input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="doctor@hospital.com"
                            required
                            className="w-full rounded-lg border border-border-subtle bg-bg-secondary px-3.5 py-2.5 text-sm text-text-primary placeholder:text-text-muted transition-colors focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
                        />
                    </div>
                    <div>
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">
                            Password
                        </label>
                        <input
                            type="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="••••••••"
                            required
                            className="w-full rounded-lg border border-border-subtle bg-bg-secondary px-3.5 py-2.5 text-sm text-text-primary placeholder:text-text-muted transition-colors focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
                        />
                    </div>
                    <button
                        type="submit"
                        disabled={loading}
                        className="mt-2 flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-bg-primary transition-all hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                    >
                        {loading ? 'Signing in…' : 'Sign In'}
                        {!loading && <ArrowRight className="h-4 w-4" />}
                    </button>
                </form>

                <p className="mt-6 text-sm text-text-muted">
                    Don&apos;t have an account?{' '}
                    <Link href="/register" className="font-medium text-accent hover:text-accent-hover transition-colors">
                        Register
                    </Link>
                </p>
            </div>
        </div>
    );
}
