'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Sidebar from '@/components/Sidebar';
import { useAuth } from '@/contexts/AuthContext';
import { patients, alerts as alertsApi } from '@/lib/api';

export default function DoctorDashboard() {
    const { user, loading } = useAuth();
    const router = useRouter();
    const [stats, setStats] = useState({ patients: 0, alerts: { total: 0, unacknowledged: 0, by_severity: {} } });
    const [patientList, setPatientList] = useState([]);
    const [recentAlerts, setRecentAlerts] = useState([]);

    useEffect(() => {
        if (!loading && !user) router.push('/login');
    }, [user, loading, router]);

    useEffect(() => {
        if (user) {
            patients.list('limit=10').then(res => {
                setPatientList(res.patients || []);
                setStats(prev => ({ ...prev, patients: res.total || 0 }));
            }).catch(() => { });

            alertsApi.stats().then(res => {
                setStats(prev => ({ ...prev, alerts: res }));
            }).catch(() => { });

            alertsApi.list('limit=5').then(res => {
                setRecentAlerts(res.alerts || []);
            }).catch(() => { });
        }
    }, [user]);

    if (loading || !user) return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}><div className="spinner" /></div>;

    const alertStats = stats.alerts;

    return (
        <div className="page-container">
            <Sidebar />
            <div className="main-content">
                <div style={{ marginBottom: 32 }}>
                    <h1>Welcome, Dr. {user.name?.split(' ').pop()}</h1>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: 4 }}>
                        AI-Driven Hemodialysis Monitoring Dashboard
                    </p>
                </div>

                {/* Stats Grid */}
                <div className="stats-grid" style={{ marginBottom: 28 }}>
                    <div className="stat-card">
                        <div className="stat-value">{stats.patients}</div>
                        <div className="stat-label">Total Patients</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value" style={{ color: 'var(--accent-green)' }}>0</div>
                        <div className="stat-label">Active Sessions</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value" style={{ color: alertStats.unacknowledged > 0 ? 'var(--risk-critical)' : 'var(--text-muted)' }}>
                            {alertStats.unacknowledged || 0}
                        </div>
                        <div className="stat-label">Unack. Alerts</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-value" style={{ color: 'var(--accent-purple)' }}>{alertStats.total || 0}</div>
                        <div className="stat-label">Total Alerts</div>
                    </div>
                </div>

                {/* Alert Breakdown */}
                {alertStats.by_severity && (
                    <div className="card" style={{ marginBottom: 24 }}>
                        <h3 style={{ marginBottom: 16, fontSize: '1rem' }}>Alert Breakdown</h3>
                        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                            {['CRITICAL', 'HIGH', 'MODERATE', 'LOW'].map(sev => (
                                <div key={sev} style={{
                                    display: 'flex', alignItems: 'center', gap: 8
                                }}>
                                    <span className={`badge badge-${sev.toLowerCase()}`}>{sev}</span>
                                    <span style={{ fontWeight: 600, fontSize: '1.1rem' }}>
                                        {alertStats.by_severity[sev] || 0}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
                    {/* Recent Patients */}
                    <div className="card">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                            <h3 style={{ fontSize: '1rem' }}>Recent Patients</h3>
                            <button className="btn btn-ghost btn-sm" onClick={() => router.push('/patients')}>View All</button>
                        </div>
                        {patientList.length === 0 ? (
                            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No patients yet. Add your first patient.</p>
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                {patientList.slice(0, 5).map(p => (
                                    <div key={p.id} style={{
                                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                        padding: '10px 14px', background: 'rgba(255,255,255,0.03)',
                                        borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                                        transition: 'background 0.15s'
                                    }}
                                        onClick={() => router.push(`/patients/${p.id}`)}
                                    >
                                        <div>
                                            <div style={{ fontWeight: 500, fontSize: '0.9rem' }}>{p.name || `Patient #${p.id?.slice(0, 6)}`}</div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                                Age {p.age} · {p.gender} · {p.disease_severity || 'Unknown'}
                                            </div>
                                        </div>
                                        <button className="btn btn-primary btn-sm"
                                            onClick={(e) => { e.stopPropagation(); router.push(`/monitor?patient=${p.id}`); }}>
                                            Monitor
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Recent Alerts */}
                    <div className="card">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                            <h3 style={{ fontSize: '1rem' }}>Recent Alerts</h3>
                            <button className="btn btn-ghost btn-sm" onClick={() => router.push('/alerts')}>View All</button>
                        </div>
                        {recentAlerts.length === 0 ? (
                            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No alerts yet.</p>
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                {recentAlerts.map((a, i) => (
                                    <div key={i} className={`alert-banner ${a.severity?.toLowerCase()}`} style={{ padding: '10px 14px' }}>
                                        <span className={`badge badge-${a.severity?.toLowerCase()}`}>{a.severity}</span>
                                        <span style={{ fontSize: '0.85rem', flex: 1 }}>{a.message}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {/* Quick Actions */}
                <div className="card" style={{ marginTop: 24 }}>
                    <h3 style={{ marginBottom: 16, fontSize: '1rem' }}>Quick Actions</h3>
                    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                        <button className="btn btn-primary" onClick={() => router.push('/patients?action=add')}>
                            ➕ Add Patient
                        </button>
                        <button className="btn btn-ghost" onClick={() => router.push('/dashboard/command')}>
                            🖥️ Command Center
                        </button>
                        <button className="btn btn-ghost" onClick={() => router.push('/model-info')}>
                            🧠 Model Info
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
