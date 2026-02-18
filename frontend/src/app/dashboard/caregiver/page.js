'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Sidebar from '@/components/Sidebar';
import { useAuth } from '@/contexts/AuthContext';
import { patients, alerts as alertsApi } from '@/lib/api';

export default function CaregiverDashboard() {
    const { user, loading } = useAuth();
    const router = useRouter();
    const [patientList, setPatientList] = useState([]);
    const [alertList, setAlertList] = useState([]);

    useEffect(() => {
        if (!loading && !user) router.push('/login');
    }, [user, loading, router]);

    useEffect(() => {
        if (user) {
            patients.list('limit=20').then(res => setPatientList(res.patients || [])).catch(() => { });
            alertsApi.list('limit=10&acknowledged=false').then(res => setAlertList(res.alerts || [])).catch(() => { });
        }
    }, [user]);

    if (loading || !user) return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}><div className="spinner" /></div>;

    return (
        <div className="page-container">
            <Sidebar />
            <div className="main-content">
                <h1 style={{ marginBottom: 8 }}>Caregiver Dashboard</h1>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: 24 }}>
                    Welcome, {user.name}
                </p>

                {/* Unacknowledged Alerts */}
                {alertList.length > 0 && (
                    <div className="card" style={{ marginBottom: 24 }}>
                        <h3 style={{ marginBottom: 16, fontSize: '1rem' }}>⚠️ Active Alerts</h3>
                        {alertList.map((a, i) => (
                            <div key={i} className={`alert-banner ${a.severity?.toLowerCase()}`} style={{ marginBottom: 8 }}>
                                <span className={`badge badge-${a.severity?.toLowerCase()}`}>{a.severity}</span>
                                <span style={{ flex: 1, fontSize: '0.85rem' }}>{a.message}</span>
                                <button className="btn btn-ghost btn-sm" onClick={async () => {
                                    try {
                                        await alertsApi.acknowledge(a.id);
                                        setAlertList(prev => prev.filter(al => al.id !== a.id));
                                    } catch (e) { }
                                }}>✓ Ack</button>
                            </div>
                        ))}
                    </div>
                )}

                {/* Patient List */}
                <div className="card">
                    <h3 style={{ marginBottom: 16, fontSize: '1rem' }}>Patients</h3>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                        {patientList.map(p => (
                            <div key={p.id} style={{
                                padding: '14px 16px', background: 'rgba(255,255,255,0.03)',
                                borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                                border: '1px solid var(--border-default)',
                                transition: 'all 0.15s ease'
                            }} onClick={() => router.push(`/monitor?patient=${p.id}`)}>
                                <div style={{ fontWeight: 500 }}>{p.name || `Patient #${p.id?.slice(0, 6)}`}</div>
                                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 4 }}>
                                    {p.age} yrs · {p.gender} · {p.disease_severity}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
