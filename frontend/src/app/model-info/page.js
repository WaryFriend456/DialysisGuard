'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Sidebar from '@/components/Sidebar';
import { useAuth } from '@/contexts/AuthContext';
import { explain } from '@/lib/api';

export default function ModelInfoPage() {
    const { user, loading } = useAuth();
    const router = useRouter();
    const [modelCard, setModelCard] = useState(null);

    useEffect(() => {
        if (!loading && !user) router.push('/login');
    }, [user, loading, router]);

    useEffect(() => {
        if (user) {
            explain.modelCard().then(setModelCard).catch(() => { });
        }
    }, [user]);

    if (loading || !user) return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}><div className="spinner" /></div>;

    return (
        <div className="page-container">
            <Sidebar />
            <div className="main-content">
                <h1 style={{ marginBottom: 8 }}>Model Transparency</h1>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: 24 }}>
                    Full transparency into the AI model powering DialysisGuard
                </p>

                {modelCard ? (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                        <div className="card">
                            <h3 style={{ marginBottom: 16 }}>Architecture</h3>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                                <p><strong>Model:</strong> {modelCard.model_name}</p>
                                <p><strong>Type:</strong> {modelCard.model_type}</p>
                                <p><strong>Framework:</strong> {modelCard.framework}</p>
                                <p><strong>Features:</strong> {modelCard.n_features}</p>
                                <p><strong>Time Steps:</strong> {modelCard.n_timesteps}</p>
                                <p><strong>Total Parameters:</strong> {modelCard.total_params?.toLocaleString()}</p>
                            </div>
                        </div>

                        <div className="card">
                            <h3 style={{ marginBottom: 16 }}>Performance</h3>
                            {modelCard.performance && (
                                <div style={{ fontSize: '0.85rem', lineHeight: 1.8, color: 'var(--text-secondary)' }}>
                                    <p><strong>Accuracy:</strong> <span style={{ color: 'var(--accent-green)', fontWeight: 700 }}>{(modelCard.performance.accuracy * 100).toFixed(1)}%</span></p>
                                    <p><strong>AUC-ROC:</strong> <span style={{ color: 'var(--accent-green)', fontWeight: 700 }}>{(modelCard.performance.auc * 100).toFixed(1)}%</span></p>
                                    <p><strong>Precision:</strong> {(modelCard.performance.precision * 100).toFixed(1)}%</p>
                                    <p><strong>Recall:</strong> {(modelCard.performance.recall * 100).toFixed(1)}%</p>
                                    <p><strong>F1 Score:</strong> {(modelCard.performance.f1 * 100).toFixed(1)}%</p>
                                </div>
                            )}
                        </div>

                        <div className="card">
                            <h3 style={{ marginBottom: 16 }}>Training Data</h3>
                            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                                <p><strong>Dataset:</strong> {modelCard.training_data?.dataset}</p>
                                <p><strong>Patients:</strong> {modelCard.training_data?.n_patients?.toLocaleString()}</p>
                                <p><strong>Time Steps:</strong> {modelCard.training_data?.n_timesteps_per_patient}</p>
                                <p><strong>Positive Rate:</strong> {modelCard.training_data?.positive_rate}</p>
                                <p><strong>Split Strategy:</strong> {modelCard.training_data?.split_strategy}</p>
                            </div>
                        </div>

                        <div className="card">
                            <h3 style={{ marginBottom: 16 }}>XAI Capabilities</h3>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                {[
                                    { icon: '🔬', name: 'SHAP Values', desc: 'Feature attribution via DeepExplainer' },
                                    { icon: '👁️', name: 'Attention Weights', desc: 'Temporal focus visualization' },
                                    { icon: '🔀', name: 'What-If Analysis', desc: 'Parameter modification scenarios' },
                                    { icon: '🔄', name: 'Counterfactuals', desc: 'Minimal changes for target risk' },
                                    { icon: '📊', name: 'Sensitivity', desc: 'Feature sensitivity ±10%' },
                                    { icon: '📖', name: 'NL Explanations', desc: 'Human-readable risk reasoning' },
                                    { icon: '📋', name: 'Model Card', desc: 'Full transparency documentation' },
                                ].map((cap, i) => (
                                    <div key={i} style={{
                                        display: 'flex', gap: 10, alignItems: 'center', padding: '8px 12px',
                                        background: 'rgba(6, 182, 212, 0.05)', borderRadius: 'var(--radius-sm)',
                                        border: '1px solid rgba(6, 182, 212, 0.1)'
                                    }}>
                                        <span style={{ fontSize: '1.2rem' }}>{cap.icon}</span>
                                        <div>
                                            <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>{cap.name}</div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{cap.desc}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {modelCard.features && (
                            <div className="card" style={{ gridColumn: '1 / -1' }}>
                                <h3 style={{ marginBottom: 16 }}>Feature List ({modelCard.features.length})</h3>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                    {modelCard.features.map((f, i) => (
                                        <span key={i} style={{
                                            padding: '4px 10px', background: 'rgba(255,255,255,0.05)',
                                            borderRadius: 20, fontSize: '0.78rem', color: 'var(--text-secondary)',
                                            border: '1px solid var(--border-default)'
                                        }}>{f}</span>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="card" style={{ textAlign: 'center', padding: 40 }}>
                        <div className="spinner" style={{ margin: '0 auto' }} />
                        <p style={{ color: 'var(--text-muted)', marginTop: 16 }}>Loading model information...</p>
                    </div>
                )}
            </div>
        </div>
    );
}
