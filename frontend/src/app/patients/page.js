'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Sidebar from '@/components/Sidebar';
import { useAuth } from '@/contexts/AuthContext';
import { patients as patientsApi } from '@/lib/api';

export default function PatientsPage() {
    const { user, loading } = useAuth();
    const router = useRouter();
    const [patientList, setPatientList] = useState([]);
    const [search, setSearch] = useState('');
    const [showAddModal, setShowAddModal] = useState(false);
    const [formData, setFormData] = useState({
        name: '', age: 55, gender: 'Male', weight: 75, diabetes: false, hypertension: false,
        kidney_failure_cause: 'Other', creatinine: 5.0, urea: 50.0, potassium: 4.5,
        hemoglobin: 11.0, hematocrit: 33.0, albumin: 3.8, dialysis_duration: 4.0,
        dialysis_frequency: 3, dialysate_composition: 'Standard', vascular_access_type: 'Fistula',
        dialyzer_type: 'High-flux', urine_output: 500, dry_weight: 70.0,
        fluid_removal_rate: 350, disease_severity: 'Moderate'
    });

    useEffect(() => {
        if (!loading && !user) router.push('/login');
    }, [user, loading, router]);

    useEffect(() => {
        if (user) loadPatients();
    }, [user, search]);

    const loadPatients = async () => {
        try {
            const res = await patientsApi.list(`search=${search}&limit=50`);
            setPatientList(res.patients || []);
        } catch (e) { console.error(e); }
    };

    const handleAdd = async (e) => {
        e.preventDefault();
        try {
            await patientsApi.create(formData);
            setShowAddModal(false);
            loadPatients();
        } catch (e) { alert(e.message); }
    };

    if (loading || !user) return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}><div className="spinner" /></div>;

    return (
        <div className="page-container">
            <Sidebar />
            <div className="main-content">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
                    <h1>Patients</h1>
                    <button className="btn btn-primary" onClick={() => setShowAddModal(true)}>➕ Add Patient</button>
                </div>

                <div style={{ marginBottom: 20 }}>
                    <input className="input" placeholder="Search patients..." value={search}
                        onChange={e => setSearch(e.target.value)} style={{ maxWidth: 400 }} />
                </div>

                {patientList.length === 0 ? (
                    <div className="card" style={{ textAlign: 'center', padding: 60 }}>
                        <p style={{ fontSize: '2rem', marginBottom: 12 }}>👥</p>
                        <p style={{ color: 'var(--text-muted)' }}>No patients found. Add your first patient to get started.</p>
                    </div>
                ) : (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 }}>
                        {patientList.map(p => (
                            <div key={p.id} className="card" style={{ cursor: 'pointer' }}
                                onClick={() => router.push(`/monitor?patient=${p.id}`)}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                    <div>
                                        <h3 style={{ fontSize: '1rem', marginBottom: 4 }}>{p.name || `Patient #${p.id?.slice(0, 6)}`}</h3>
                                        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                            {p.age} yrs · {p.gender} · {p.weight} kg
                                        </p>
                                    </div>
                                    <span className={`badge badge-${(p.disease_severity || 'Moderate').toLowerCase()}`}>
                                        {p.disease_severity || 'Moderate'}
                                    </span>
                                </div>

                                <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                                    <span>BP Access: {p.vascular_access_type}</span>
                                    <span>Dialyzer: {p.dialyzer_type}</span>
                                    <span>Duration: {p.dialysis_duration}h</span>
                                    <span>Freq: {p.dialysis_frequency}x/wk</span>
                                </div>

                                <div style={{ marginTop: 14, display: 'flex', gap: 8 }}>
                                    <button className="btn btn-primary btn-sm" style={{ flex: 1 }}
                                        onClick={(e) => { e.stopPropagation(); router.push(`/monitor?patient=${p.id}`); }}>
                                        🔴 Start Monitor
                                    </button>
                                    <button className="btn btn-ghost btn-sm"
                                        onClick={(e) => { e.stopPropagation(); router.push(`/patients/${p.id}/history`); }}>
                                        📊 History
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Add Patient Modal */}
                {showAddModal && (
                    <div style={{
                        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
                    }}>
                        <div className="card" style={{ width: 600, maxHeight: '85vh', overflowY: 'auto', padding: 32 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 20 }}>
                                <h2>Add New Patient</h2>
                                <button className="btn btn-ghost btn-sm" onClick={() => setShowAddModal(false)}>✕</button>
                            </div>
                            <form onSubmit={handleAdd}>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                                    <div className="form-group">
                                        <label className="label">Name</label>
                                        <input className="input" value={formData.name}
                                            onChange={e => setFormData({ ...formData, name: e.target.value })} required />
                                    </div>
                                    <div className="form-group">
                                        <label className="label">Age</label>
                                        <input className="input" type="number" value={formData.age}
                                            onChange={e => setFormData({ ...formData, age: parseInt(e.target.value) })} required />
                                    </div>
                                    <div className="form-group">
                                        <label className="label">Gender</label>
                                        <select className="input select" value={formData.gender}
                                            onChange={e => setFormData({ ...formData, gender: e.target.value })}>
                                            <option>Male</option><option>Female</option>
                                        </select>
                                    </div>
                                    <div className="form-group">
                                        <label className="label">Weight (kg)</label>
                                        <input className="input" type="number" step="0.1" value={formData.weight}
                                            onChange={e => setFormData({ ...formData, weight: parseFloat(e.target.value) })} />
                                    </div>
                                    <div className="form-group">
                                        <label className="label">Severity</label>
                                        <select className="input select" value={formData.disease_severity}
                                            onChange={e => setFormData({ ...formData, disease_severity: e.target.value })}>
                                            <option>Mild</option><option>Moderate</option><option>Severe</option><option>Critical</option>
                                        </select>
                                    </div>
                                    <div className="form-group">
                                        <label className="label">Kidney Failure Cause</label>
                                        <select className="input select" value={formData.kidney_failure_cause}
                                            onChange={e => setFormData({ ...formData, kidney_failure_cause: e.target.value })}>
                                            <option>Diabetes</option><option>Hypertension</option><option>Glomerulonephritis</option>
                                            <option>Polycystic</option><option>Other</option>
                                        </select>
                                    </div>
                                    <div className="form-group">
                                        <label className="label">Creatinine</label>
                                        <input className="input" type="number" step="0.1" value={formData.creatinine}
                                            onChange={e => setFormData({ ...formData, creatinine: parseFloat(e.target.value) })} />
                                    </div>
                                    <div className="form-group">
                                        <label className="label">Urea</label>
                                        <input className="input" type="number" step="0.1" value={formData.urea}
                                            onChange={e => setFormData({ ...formData, urea: parseFloat(e.target.value) })} />
                                    </div>
                                    <div className="form-group">
                                        <label className="label">Fluid Removal (ml/hr)</label>
                                        <input className="input" type="number" value={formData.fluid_removal_rate}
                                            onChange={e => setFormData({ ...formData, fluid_removal_rate: parseInt(e.target.value) })} />
                                    </div>
                                    <div className="form-group">
                                        <label className="label">Dialysis Duration (hrs)</label>
                                        <input className="input" type="number" step="0.5" value={formData.dialysis_duration}
                                            onChange={e => setFormData({ ...formData, dialysis_duration: parseFloat(e.target.value) })} />
                                    </div>
                                </div>

                                <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                        <input type="checkbox" checked={formData.diabetes}
                                            onChange={e => setFormData({ ...formData, diabetes: e.target.checked })} /> Diabetes
                                    </label>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                        <input type="checkbox" checked={formData.hypertension}
                                            onChange={e => setFormData({ ...formData, hypertension: e.target.checked })} /> Hypertension
                                    </label>
                                </div>

                                <div style={{ display: 'flex', gap: 12, marginTop: 20, justifyContent: 'flex-end' }}>
                                    <button type="button" className="btn btn-ghost" onClick={() => setShowAddModal(false)}>Cancel</button>
                                    <button type="submit" className="btn btn-primary">Add Patient</button>
                                </div>
                            </form>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
