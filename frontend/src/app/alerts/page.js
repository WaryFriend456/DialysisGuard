'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Sidebar from '@/components/Sidebar';
import { useAuth } from '@/contexts/AuthContext';
import { alerts as alertsApi } from '@/lib/api';

export default function AlertsPage() {
    const { user, loading } = useAuth();
    const router = useRouter();
    const [alertList, setAlertList] = useState([]);
    const [stats, setStats] = useState(null);
    const [filter, setFilter] = useState('');

    useEffect(() => {
        if (!loading && !user) router.push('/login');
    }, [user, loading, router]);

    useEffect(() => {
        if (user) {
            const params = filter ? `severity=${filter}&limit=100` : 'limit=100';
            alertsApi.list(params).then(res => setAlertList(res.alerts || [])).catch(() => { });
            alertsApi.stats().then(setStats).catch(() => { });
        }
    }, [user, filter]);

    if (loading || !user) return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}><div className="spinner" /></div>;

    return (
        <div className="page-container">
            <Sidebar />
            <div className="main-content">
                <h1 style={{ marginBottom: 24 }}>Alerts</h1>

                {stats && (
                    <div className="stats-grid" style={{ marginBottom: 24 }}>
                        <div className="stat-card">
                            <div className="stat-value">{stats.total}</div>
                            <div className="stat-label">Total</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-value" style={{ color: 'var(--risk-critical)' }}>{stats.unacknowledged}</div>
                            <div className="stat-label">Unacknowledged</div>
                        </div>
                        {['CRITICAL', 'HIGH', 'MODERATE'].map(s => (
                            <div className="stat-card" key={s}>
                                <div className="stat-value" style={{ color: `var(--risk-${s.toLowerCase()})` }}>
                                    {stats.by_severity?.[s] || 0}
                                </div>
                                <div className="stat-label">{s}</div>
                            </div>
                        ))}
                    </div>
                )}

                <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
                    {['', 'CRITICAL', 'HIGH', 'MODERATE', 'LOW'].map(f => (
                        <button key={f} className={`btn btn-sm ${filter === f ? 'btn-primary' : 'btn-ghost'}`}
                            onClick={() => setFilter(f)}>
                            {f || 'All'}
                        </button>
                    ))}
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {alertList.length === 0 ? (
                        <div className="card" style={{ textAlign: 'center', padding: 40 }}>
                            <p style={{ color: 'var(--text-muted)' }}>No alerts found.</p>
                        </div>
                    ) : alertList.map((a, i) => (
                        <div key={i} className={`alert-banner ${a.severity?.toLowerCase()}`}>
                            <span className={`badge badge-${a.severity?.toLowerCase()}`}>{a.severity}</span>
                            <div style={{ flex: 1 }}>
                                <div style={{ fontSize: '0.85rem' }}>{a.message}</div>
                                {a.nl_explanation && (
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 4 }}>{a.nl_explanation}</div>
                                )}
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                {a.acknowledged ? (
                                    <span style={{ fontSize: '0.75rem', color: 'var(--risk-low)' }}>✓ Acknowledged</span>
                                ) : (
                                    <button className="btn btn-ghost btn-sm" onClick={async () => {
                                        try {
                                            await alertsApi.acknowledge(a.id);
                                            setAlertList(prev => prev.map(al => al.id === a.id ? { ...al, acknowledged: true } : al));
                                        } catch (e) { }
                                    }}>✓ Ack</button>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
