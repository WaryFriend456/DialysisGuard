'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';
import { UserPlus, AlertCircle, ArrowRight } from 'lucide-react';

export default function RegisterPage() {
    const [form, setForm] = useState({ name: '', email: '', password: '', role: 'doctor' });
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { register } = useAuth();
    const router = useRouter();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            const user = await register(form);
            router.push(user.role === 'doctor' ? '/dashboard/doctor' : '/dashboard/caregiver');
        } catch (err) {
            setError(err.message || 'Registration failed');
        } finally {
            setLoading(false);
        }
    };

    const update = (key, val) => setForm((f) => ({ ...f, [key]: val }));

    return (
        <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-bg-primary">
            {/* Ambient glow */}
            <div className="pointer-events-none absolute top-1/3 right-1/5 h-72 w-72 rounded-full bg-accent/5 blur-[80px]" />

            <div className="card w-full max-w-md px-10 py-10 text-center backdrop-blur-xl animate-fade-in">
                {/* Brand */}
                <div className="mb-8 flex flex-col items-center gap-3">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-accent/15 shadow-glow-accent">
                        <UserPlus className="h-6 w-6 text-accent" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-extrabold tracking-tight text-text-primary">
                            Create Account
                        </h1>
                        <p className="mt-1 text-sm text-text-muted">
                            Join DialysisGuard monitoring platform
                        </p>
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
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">
                            Full Name
                        </label>
                        <input
                            value={form.name}
                            onChange={(e) => update('name', e.target.value)}
                            placeholder="Dr. John Smith"
                            required
                            className="w-full rounded-lg border border-border-subtle bg-bg-secondary px-3.5 py-2.5 text-sm text-text-primary placeholder:text-text-muted transition-colors focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
                        />
                    </div>
                    <div>
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">
                            Email
                        </label>
                        <input
                            type="email"
                            value={form.email}
                            onChange={(e) => update('email', e.target.value)}
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
                            value={form.password}
                            onChange={(e) => update('password', e.target.value)}
                            placeholder="••••••••"
                            required
                            className="w-full rounded-lg border border-border-subtle bg-bg-secondary px-3.5 py-2.5 text-sm text-text-primary placeholder:text-text-muted transition-colors focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
                        />
                    </div>
                    <div>
                        <label className="mb-1.5 block text-xs font-medium text-text-secondary">
                            Role
                        </label>
                        <select
                            value={form.role}
                            onChange={(e) => update('role', e.target.value)}
                            className="w-full appearance-none rounded-lg border border-border-subtle bg-bg-secondary px-3.5 py-2.5 text-sm text-text-primary transition-colors focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
                        >
                            <option value="doctor">Doctor</option>
                            <option value="caregiver">Caregiver</option>
                        </select>
                    </div>
                    <button
                        type="submit"
                        disabled={loading}
                        className="mt-2 flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-bg-primary transition-all hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                    >
                        {loading ? 'Creating…' : 'Create Account'}
                        {!loading && <ArrowRight className="h-4 w-4" />}
                    </button>
                </form>

                <p className="mt-6 text-sm text-text-muted">
                    Already have an account?{' '}
                    <Link href="/login" className="font-medium text-accent hover:text-accent-hover transition-colors">
                        Sign In
                    </Link>
                </p>
            </div>
        </div>
    );
}
