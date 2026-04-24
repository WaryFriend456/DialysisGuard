'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { AlertCircle, ArrowRight, BriefcaseMedical, ChevronDown, HeartPulse, UserPlus } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { cn } from '@/lib/utils';

const INPUT_CLS =
    'w-full rounded-xl border border-border-subtle bg-bg-secondary px-3.5 py-2.5 text-sm text-text-primary placeholder:text-text-muted transition-colors focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30';

const ROLE_OPTIONS = [
    { value: 'doctor', label: 'Doctor', icon: BriefcaseMedical, description: 'Primary clinician with full workspace access.' },
    { value: 'caregiver', label: 'Caregiver', icon: HeartPulse, description: 'Bedside monitoring and alert response role.' },
];

export default function RegisterPage() {
    const [form, setForm] = useState({ name: '', email: '', password: '', role: 'doctor' });
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [roleOpen, setRoleOpen] = useState(false);
    const menuRef = useRef(null);
    const { register } = useAuth();
    const router = useRouter();

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (menuRef.current && !menuRef.current.contains(event.target)) {
                setRoleOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleSubmit = async (event) => {
        event.preventDefault();
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

    const update = (key, val) => setForm((current) => ({ ...current, [key]: val }));
    const selectedRole = ROLE_OPTIONS.find((role) => role.value === form.role) || ROLE_OPTIONS[0];
    const RoleIcon = selectedRole.icon;

    return (
        <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-bg-primary px-4 py-8">
            <div className="pointer-events-none absolute top-1/4 left-[8%] h-56 w-56 rounded-full bg-accent/5 blur-[100px]" />
            <div className="pointer-events-none absolute right-[10%] bottom-1/4 h-56 w-56 rounded-full bg-risk-moderate-bg/50 blur-[100px]" />

            <div className="card w-full max-w-md px-8 py-8 text-center backdrop-blur-xl animate-fade-in sm:px-10 sm:py-10">
                <div className="mb-8 flex flex-col items-center gap-3">
                    <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-accent/15 shadow-glow-accent">
                        <UserPlus className="h-6 w-6 text-accent" />
                    </div>
                    <div>
                        <h1 className="text-3xl font-extrabold tracking-tight text-text-primary">Create Account</h1>
                        <p className="mt-1 text-sm text-text-muted">Join DialysisGuard clinical workspace</p>
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
                            Full Name
                        </label>
                        <input
                            value={form.name}
                            onChange={(event) => update('name', event.target.value)}
                            placeholder="Dr. John Smith"
                            required
                            className={INPUT_CLS}
                        />
                    </div>
                    <div>
                        <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-text-secondary">
                            Email
                        </label>
                        <input
                            type="email"
                            value={form.email}
                            onChange={(event) => update('email', event.target.value)}
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
                            value={form.password}
                            onChange={(event) => update('password', event.target.value)}
                            placeholder="••••••••"
                            required
                            className={INPUT_CLS}
                        />
                    </div>

                    <div ref={menuRef} className="relative">
                        <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-text-secondary">
                            Role
                        </label>
                        <button
                            type="button"
                            onClick={() => setRoleOpen((current) => !current)}
                            className={cn(
                                'flex w-full items-center justify-between gap-3 rounded-xl border bg-bg-secondary px-3.5 py-2.5 text-left transition-colors',
                                roleOpen ? 'border-accent shadow-glow-accent/30' : 'border-border-subtle hover:border-border'
                            )}
                            aria-haspopup="listbox"
                            aria-expanded={roleOpen}
                        >
                            <span className="flex items-center gap-2 text-sm text-text-primary">
                                <RoleIcon className="h-4 w-4 text-accent" />
                                {selectedRole.label}
                            </span>
                            <ChevronDown className={cn('h-4 w-4 text-text-muted transition-transform', roleOpen && 'rotate-180')} />
                        </button>
                        <p className="mt-1 text-xs text-text-muted">{selectedRole.description}</p>

                        {roleOpen && (
                            <ul
                                role="listbox"
                                className="absolute z-20 mt-2 w-full space-y-1 rounded-xl border border-border bg-surface p-2 shadow-elevated"
                            >
                                {ROLE_OPTIONS.map((option) => {
                                    const OptionIcon = option.icon;
                                    const isSelected = form.role === option.value;
                                    return (
                                        <li key={option.value}>
                                            <button
                                                type="button"
                                                role="option"
                                                aria-selected={isSelected}
                                                onClick={() => {
                                                    update('role', option.value);
                                                    setRoleOpen(false);
                                                }}
                                                className={cn(
                                                    'flex w-full items-start gap-2 rounded-lg px-3 py-2 text-left',
                                                    isSelected ? 'bg-accent/12 text-text-primary' : 'text-text-secondary hover:bg-surface-hover'
                                                )}
                                            >
                                                <OptionIcon className={cn('mt-0.5 h-4 w-4', isSelected ? 'text-accent' : 'text-text-muted')} />
                                                <span>
                                                    <span className="block text-sm font-medium">{option.label}</span>
                                                    <span className="block text-xs text-text-muted">{option.description}</span>
                                                </span>
                                            </button>
                                        </li>
                                    );
                                })}
                            </ul>
                        )}
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="mt-2 flex w-full items-center justify-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-sm font-semibold text-bg-primary transition-all hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        {loading ? 'Creating...' : 'Create Account'}
                        {!loading && <ArrowRight className="h-4 w-4" />}
                    </button>
                </form>

                <p className="mt-6 text-sm text-text-muted">
                    Already have an account?{' '}
                    <Link href="/login" className="font-semibold text-accent transition-colors hover:text-accent-hover">
                        Sign In
                    </Link>
                </p>
            </div>
        </div>
    );
}
