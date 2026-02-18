'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Sidebar from '@/components/Sidebar';
import { useAuth } from '@/contexts/AuthContext';
import { patients as patientsApi } from '@/lib/api';

export default function CommandCenter() {
    const { user, loading } = useAuth();
    const router = useRouter();
    const [patientList, setPatientList] = useState([]);

    useEffect(() => {
        if (!loading && !user) router.push('/login');
    }, [user, loading, router]);

    useEffect(() => {
        if (user) {
            patientsApi.list('limit=50').then(res => setPatientList(res.patients || [])).catch(() => { });
        }
    }, [user]);

    if (loading || !user) return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}><div className="spinner" /></div>;

    return (
        <div className="page-container">
            <Sidebar />
            <div className="main-content">
                <div style={{ marginBottom: 24 }}>
                    <h1 style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        🖥️ Multi-Patient Command Center
                    </h1>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: 4 }}>
                        Overview of all patients — click any card to start monitoring
                    </p>
                </div>

                {patientList.length === 0 ? (
                    <div className="card" style={{ textAlign: 'center', padding: 60 }}>
                        <p style={{ color: 'var(--text-muted)' }}>No patients available. Add patients first.</p>
                        <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={() => router.push('/patients')}>
                            Go to Patients
                        </button>
                    </div>
                ) : (
                    <div className="patient-grid">
                        {patientList.map(p => {
                            const sevClass = (p.disease_severity || 'moderate').toLowerCase();
                            return (
                                <div key={p.id} className={`patient-grid-cell risk-${sevClass}`}
                                    onClick={() => router.push(`/monitor?patient=${p.id}`)}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                                        <h3 style={{ fontSize: '1rem' }}>{p.name || `Patient #${p.id?.slice(0, 6)}`}</h3>
                                        <span className={`badge badge-${sevClass}`}>{p.disease_severity || 'Moderate'}</span>
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                                        <span>Age: {p.age}</span>
                                        <span>Gender: {p.gender}</span>
                                        <span>Weight: {p.weight} kg</span>
                                        <span>Duration: {p.dialysis_duration}h</span>
                                    </div>
                                    <div style={{ marginTop: 12, display: 'flex', gap: 6 }}>
                                        {p.diabetes && <span style={{ fontSize: '0.7rem', padding: '2px 6px', background: 'rgba(245,158,11,0.15)', borderRadius: 10, color: 'var(--accent-yellow)' }}>Diabetes</span>}
                                        {p.hypertension && <span style={{ fontSize: '0.7rem', padding: '2px 6px', background: 'rgba(239,68,68,0.15)', borderRadius: 10, color: 'var(--accent-red)' }}>Hypertension</span>}
                                    </div>
                                    <button className="btn btn-primary btn-sm" style={{ width: '100%', marginTop: 14 }}
                                        onClick={(e) => { e.stopPropagation(); router.push(`/monitor?patient=${p.id}`); }}>
                                        🔴 Start Monitor
                                    </button>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        </div>
    );
}
