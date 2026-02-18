'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';

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

    return (
        <div style={{
            minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'linear-gradient(135deg, #0a0e17 0%, #111827 50%, #0f172a 100%)',
            position: 'relative', overflow: 'hidden'
        }}>
            <div style={{
                position: 'absolute', top: '30%', right: '20%', width: 300, height: 300,
                borderRadius: '50%', background: 'radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%)',
                filter: 'blur(40px)'
            }} />

            <div className="card" style={{
                width: 440, padding: 40, textAlign: 'center',
                background: 'rgba(17, 24, 39, 0.85)', backdropFilter: 'blur(20px)'
            }}>
                <h1 style={{
                    fontSize: '1.5rem',
                    background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                    marginBottom: 8, fontWeight: 800
                }}>
                    Create Account
                </h1>
                <p style={{ color: 'var(--text-muted)', marginBottom: 28, fontSize: '0.85rem' }}>
                    Join DialysisGuard monitoring platform
                </p>

                {error && (
                    <div style={{
                        background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                        borderRadius: 'var(--radius-sm)', padding: '10px 14px', marginBottom: 20,
                        fontSize: '0.85rem', color: 'var(--accent-red)'
                    }}>{error}</div>
                )}

                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label className="label">Full Name</label>
                        <input className="input" value={form.name}
                            onChange={e => setForm({ ...form, name: e.target.value })} placeholder="Dr. John Smith" required />
                    </div>
                    <div className="form-group">
                        <label className="label">Email</label>
                        <input className="input" type="email" value={form.email}
                            onChange={e => setForm({ ...form, email: e.target.value })} placeholder="doctor@hospital.com" required />
                    </div>
                    <div className="form-group">
                        <label className="label">Password</label>
                        <input className="input" type="password" value={form.password}
                            onChange={e => setForm({ ...form, password: e.target.value })} placeholder="••••••••" required />
                    </div>
                    <div className="form-group">
                        <label className="label">Role</label>
                        <select className="input select" value={form.role}
                            onChange={e => setForm({ ...form, role: e.target.value })}>
                            <option value="doctor">Doctor</option>
                            <option value="caregiver">Caregiver</option>
                        </select>
                    </div>
                    <button className="btn btn-primary btn-lg" type="submit" disabled={loading}
                        style={{ width: '100%', marginTop: 8 }}>
                        {loading ? 'Creating...' : 'Create Account'}
                    </button>
                </form>

                <p style={{ marginTop: 24, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    Already have an account?{' '}
                    <Link href="/login" style={{ color: 'var(--accent-cyan)', fontWeight: 500 }}>Sign In</Link>
                </p>
            </div>
        </div>
    );
}
