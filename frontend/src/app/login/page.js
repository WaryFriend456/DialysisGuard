'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';

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
        <div style={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'linear-gradient(135deg, #0a0e17 0%, #111827 50%, #0f172a 100%)',
            position: 'relative',
            overflow: 'hidden'
        }}>
            {/* Animated background orbs */}
            <div style={{
                position: 'absolute', top: '20%', left: '15%', width: 300, height: 300,
                borderRadius: '50%', background: 'radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%)',
                filter: 'blur(40px)', animation: 'float 8s ease-in-out infinite'
            }} />
            <div style={{
                position: 'absolute', bottom: '10%', right: '10%', width: 400, height: 400,
                borderRadius: '50%', background: 'radial-gradient(circle, rgba(139,92,246,0.06) 0%, transparent 70%)',
                filter: 'blur(60px)', animation: 'float 10s ease-in-out infinite reverse'
            }} />

            <div className="card" style={{
                width: 440, padding: 40, textAlign: 'center',
                background: 'rgba(17, 24, 39, 0.85)', backdropFilter: 'blur(20px)',
                border: '1px solid rgba(148, 163, 184, 0.1)'
            }}>
                <h1 style={{
                    fontSize: '1.8rem',
                    background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    marginBottom: 8, fontWeight: 800
                }}>
                    DialysisGuard
                </h1>
                <p style={{ color: 'var(--text-muted)', marginBottom: 32, fontSize: '0.85rem' }}>
                    AI-Driven Hemodialysis Monitoring
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
                        <label className="label">Email</label>
                        <input className="input" type="email" value={email}
                            onChange={e => setEmail(e.target.value)} placeholder="doctor@hospital.com" required />
                    </div>
                    <div className="form-group">
                        <label className="label">Password</label>
                        <input className="input" type="password" value={password}
                            onChange={e => setPassword(e.target.value)} placeholder="••••••••" required />
                    </div>
                    <button className="btn btn-primary btn-lg" type="submit" disabled={loading}
                        style={{ width: '100%', marginTop: 8 }}>
                        {loading ? 'Signing in...' : 'Sign In'}
                    </button>
                </form>

                <p style={{ marginTop: 24, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    Don&apos;t have an account?{' '}
                    <Link href="/register" style={{ color: 'var(--accent-cyan)', fontWeight: 500 }}>Register</Link>
                </p>
            </div>

            <style jsx>{`
        @keyframes float {
          0%, 100% { transform: translateY(0) scale(1); }
          50% { transform: translateY(-20px) scale(1.05); }
        }
      `}</style>
        </div>
    );
}
