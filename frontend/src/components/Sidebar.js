'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';

const navItems = [
    { href: '/dashboard/doctor', label: 'Dashboard', icon: '📊', roles: ['doctor'] },
    { href: '/dashboard/caregiver', label: 'Dashboard', icon: '🏥', roles: ['caregiver'] },
    { href: '/dashboard/command', label: 'Command Center', icon: '🖥️', roles: ['doctor'] },
    { href: '/patients', label: 'Patients', icon: '👥', roles: ['doctor', 'caregiver'] },
    { href: '/alerts', label: 'Alerts', icon: '🔔', roles: ['doctor', 'caregiver'] },
    { href: '/model-info', label: 'Model Info', icon: '🧠', roles: ['doctor'] },
];

export default function Sidebar() {
    const pathname = usePathname();
    const { user, logout } = useAuth();

    if (!user) return null;

    const filtered = navItems.filter(item => item.roles.includes(user.role));

    return (
        <div className="sidebar">
            <div style={{ marginBottom: 32 }}>
                <h2 style={{
                    fontSize: '1.3rem',
                    background: 'linear-gradient(135deg, #06b6d4, #8b5cf6)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    fontWeight: 800,
                    letterSpacing: '-0.03em'
                }}>
                    DialysisGuard
                </h2>
                <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4 }}>
                    AI-Driven Monitoring
                </p>
            </div>

            <nav style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {filtered.map(item => (
                    <Link
                        key={item.href}
                        href={item.href}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 10,
                            padding: '10px 14px',
                            borderRadius: 'var(--radius-sm)',
                            fontSize: '0.9rem',
                            fontWeight: pathname === item.href ? 600 : 400,
                            color: pathname === item.href ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                            background: pathname === item.href ? 'rgba(6, 182, 212, 0.1)' : 'transparent',
                            textDecoration: 'none',
                            transition: 'all 0.15s ease',
                        }}
                    >
                        <span>{item.icon}</span>
                        {item.label}
                    </Link>
                ))}
            </nav>

            <div style={{
                position: 'absolute', bottom: 24, left: 16, right: 16,
                borderTop: '1px solid var(--border-default)', paddingTop: 16
            }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 500, marginBottom: 4 }}>
                    {user.name}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 12, textTransform: 'capitalize' }}>
                    {user.role}
                </div>
                <button onClick={logout} className="btn btn-ghost btn-sm" style={{ width: '100%' }}>
                    Logout
                </button>
            </div>
        </div>
    );
}
