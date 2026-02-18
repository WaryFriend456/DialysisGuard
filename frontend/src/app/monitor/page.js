'use client';
import { useState, useEffect, useRef, useCallback, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Sidebar from '@/components/Sidebar';
import { useAuth } from '@/contexts/AuthContext';
import { sessions, patients as patientsApi, connectMonitor, alerts as alertsApi } from '@/lib/api';

function MonitorContent() {
    const { user, loading } = useAuth();
    const router = useRouter();
    const searchParams = useSearchParams();
    const patientId = searchParams.get('patient');

    const [patient, setPatient] = useState(null);
    const [sessionId, setSessionId] = useState(null);
    const [connected, setConnected] = useState(false);
    const [currentStep, setCurrentStep] = useState(0);
    const [riskProfile, setRiskProfile] = useState('');
    const [latestData, setLatestData] = useState(null);
    const [vitalsHistory, setVitalsHistory] = useState([]);
    const [riskHistory, setRiskHistory] = useState([]);
    const [activeAlerts, setActiveAlerts] = useState([]);
    const [sessionComplete, setSessionComplete] = useState(false);
    const [screenFlash, setScreenFlash] = useState(false);
    const wsRef = useRef(null);
    const alertSoundRef = useRef(null);

    useEffect(() => {
        if (!loading && !user) router.push('/login');
    }, [user, loading, router]);

    useEffect(() => {
        if (user && patientId) {
            patientsApi.get(patientId).then(setPatient).catch(e => {
                alert('Patient not found');
                router.push('/patients');
            });
        }
    }, [user, patientId, router]);

    const startSession = async (profile) => {
        try {
            const session = await sessions.create({
                patient_id: patientId,
                risk_profile: profile || undefined
            });
            setSessionId(session.id);
            setRiskProfile(profile);

            const ws = connectMonitor(
                session.id,
                handleMessage,
                (e) => console.error('WS error:', e),
                () => { setConnected(false); }
            );
            wsRef.current = ws;
            setConnected(true);
        } catch (e) {
            alert('Failed to start session: ' + e.message);
        }
    };

    const stopSession = async () => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        if (sessionId) {
            try { await sessions.stop(sessionId); } catch (e) { }
        }
        setConnected(false);
        setSessionComplete(true);
    };

    const handleMessage = useCallback((data) => {
        if (data.type === 'session_start') {
            setRiskProfile(data.risk_profile);
            return;
        }
        if (data.type === 'session_complete') {
            setSessionComplete(true);
            setConnected(false);
            return;
        }
        if (data.type === 'monitoring_data') {
            setLatestData(data);
            setCurrentStep(data.step);

            setVitalsHistory(prev => [...prev, {
                step: data.step,
                time: data.time_minutes,
                bp: data.vitals.bp,
                hr: data.vitals.hr
            }]);

            setRiskHistory(prev => [...prev, {
                step: data.step,
                risk: data.prediction.risk_probability,
                level: data.prediction.risk_level
            }]);

            // Handle alerts
            if (data.alert) {
                setActiveAlerts(prev => [data.alert, ...prev]);

                if (data.alert.severity === 'CRITICAL') {
                    setScreenFlash(true);
                    setTimeout(() => setScreenFlash(false), 500);
                    try {
                        if (alertSoundRef.current) alertSoundRef.current.play().catch(() => { });
                    } catch (e) { }
                }
            }

            if (data.escalation_alerts?.length > 0) {
                setActiveAlerts(prev => [...data.escalation_alerts, ...prev]);
            }
        }
    }, []);

    const acknowledgeAlert = async (alertId) => {
        try {
            await alertsApi.acknowledge(alertId);
            setActiveAlerts(prev => prev.filter(a => a.id !== alertId));
        } catch (e) { }
    };

    if (loading || !user) return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}><div className="spinner" /></div>;

    if (!patientId) return (
        <div className="page-container">
            <Sidebar />
            <div className="main-content" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <div className="card" style={{ textAlign: 'center', padding: 40 }}>
                    <p style={{ fontSize: '2rem', marginBottom: 12 }}>🔴</p>
                    <h2>No Patient Selected</h2>
                    <p style={{ color: 'var(--text-muted)', marginTop: 8 }}>
                        Go to the Patients page to select a patient for monitoring.
                    </p>
                    <button className="btn btn-primary" style={{ marginTop: 20 }} onClick={() => router.push('/patients')}>
                        Go to Patients
                    </button>
                </div>
            </div>
        </div>
    );

    const risk = latestData?.prediction;
    const xai = latestData?.xai;
    const vitals = latestData?.vitals;
    const anomalies = latestData?.anomalies || [];

    return (
        <div className="page-container">
            <Sidebar />
            <div className="main-content">
                {/* Screen flash for critical alerts */}
                {screenFlash && <div className="screen-flash" />}

                {/* Audio element for critical alerts */}
                <audio ref={alertSoundRef} preload="auto">
                    <source src="data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU" type="audio/wav" />
                </audio>

                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
                    <div>
                        <h1 style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                            {connected && <span style={{
                                width: 12, height: 12, borderRadius: '50%',
                                background: 'var(--risk-low)', boxShadow: '0 0 8px var(--risk-low)',
                                display: 'inline-block', animation: 'pulse-text 2s infinite'
                            }} />}
                            Real-Time Monitor
                        </h1>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 4 }}>
                            {patient?.name || `Patient #${patientId?.slice(0, 6)}`}
                            {riskProfile && ` · Profile: ${riskProfile}`}
                            {connected && ` · Step ${currentStep + 1}/30`}
                        </p>
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                        {!connected && !sessionComplete && (
                            <>
                                <button className="btn btn-primary" onClick={() => startSession(null)}>
                                    🔴 Start (Auto Profile)
                                </button>
                                <button className="btn btn-ghost" onClick={() => startSession('high')}>
                                    ⚡ High Risk
                                </button>
                                <button className="btn btn-ghost" onClick={() => startSession('critical')}>
                                    🔥 Critical
                                </button>
                            </>
                        )}
                        {connected && (
                            <button className="btn btn-danger" onClick={stopSession}>⏹ Stop Session</button>
                        )}
                        {sessionComplete && (
                            <button className="btn btn-primary" onClick={() => {
                                setSessionComplete(false); setLatestData(null);
                                setVitalsHistory([]); setRiskHistory([]);
                                setActiveAlerts([]); setCurrentStep(0);
                            }}>🔄 New Session</button>
                        )}
                    </div>
                </div>

                {/* Active Alerts Bar */}
                {activeAlerts.length > 0 && (
                    <div style={{ marginBottom: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {activeAlerts.slice(0, 3).map((a, i) => (
                            <div key={i} className={`alert-banner ${a.severity?.toLowerCase()}`}>
                                <span className={`badge badge-${a.severity?.toLowerCase()}`}>
                                    {a.escalation_level > 0 ? '⬆️ ' : ''}{a.severity}
                                </span>
                                <span style={{ flex: 1, fontSize: '0.85rem' }}>{a.message}</span>
                                <button className="btn btn-ghost btn-sm" onClick={() => acknowledgeAlert(a.id)}>
                                    ✓ Ack
                                </button>
                            </div>
                        ))}
                    </div>
                )}

                {!connected && !latestData && !sessionComplete && (
                    <div className="card" style={{ textAlign: 'center', padding: 60 }}>
                        <p style={{ fontSize: '3rem', marginBottom: 16 }}>🫀</p>
                        <h2>Ready to Monitor</h2>
                        <p style={{ color: 'var(--text-muted)', marginTop: 8, maxWidth: 500, marginInline: 'auto' }}>
                            Start a session to begin real-time AI monitoring. The system will simulate
                            realistic vital signs and predict instability using the trained GRU model.
                        </p>
                    </div>
                )}

                {latestData && (
                    <>
                        {/* Top Row: Risk Gauge + Vitals */}
                        <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr 1fr', gap: 16, marginBottom: 16 }}>
                            {/* Risk Gauge */}
                            <div className="card" style={{ textAlign: 'center', padding: 20 }}>
                                <h4 style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                    Instability Risk
                                </h4>
                                <div className={`risk-value ${risk?.risk_level?.toLowerCase()}`}>
                                    {(risk?.risk_probability * 100).toFixed(0)}%
                                </div>
                                <span className={`badge badge-${risk?.risk_level?.toLowerCase()}`} style={{ marginTop: 8 }}>
                                    {risk?.risk_level}
                                </span>

                                {/* Confidence interval */}
                                <div style={{ marginTop: 12, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                    95% CI: {(risk?.confidence?.lower * 100).toFixed(0)}% – {(risk?.confidence?.upper * 100).toFixed(0)}%
                                </div>

                                {/* Trend arrow */}
                                {xai?.risk_trend && (
                                    <div className={`trend-arrow ${xai.risk_trend}`} style={{ marginTop: 8 }}>
                                        {xai.risk_trend === 'increasing' ? '↑ Rising' :
                                            xai.risk_trend === 'decreasing' ? '↓ Falling' : '→ Stable'}
                                    </div>
                                )}
                            </div>

                            {/* BP */}
                            <div className="card" style={{ padding: 20 }}>
                                <h4 style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                    Blood Pressure
                                </h4>
                                <div style={{ fontSize: '2.2rem', fontWeight: 800, color: vitals?.bp < 80 ? 'var(--risk-critical)' : 'var(--accent-cyan)' }}>
                                    {vitals?.bp?.toFixed(0)} <span style={{ fontSize: '1rem', fontWeight: 400, color: 'var(--text-muted)' }}>mmHg</span>
                                </div>
                                <div style={{ marginTop: 8, fontSize: '0.8rem', color: vitals?.bp_change < -5 ? 'var(--risk-critical)' : 'var(--text-muted)' }}>
                                    Change: {vitals?.bp_change > 0 ? '+' : ''}{vitals?.bp_change?.toFixed(1)}
                                </div>
                                {/* Mini sparkline as text */}
                                <div style={{ marginTop: 12, display: 'flex', gap: 2, alignItems: 'end', height: 32 }}>
                                    {vitalsHistory.slice(-15).map((v, i) => {
                                        const h = Math.max(4, ((v.bp - 60) / 130) * 32);
                                        return <div key={i} style={{
                                            width: 4, height: h, borderRadius: 2,
                                            background: v.bp < 80 ? 'var(--risk-critical)' : 'var(--accent-cyan)',
                                            opacity: 0.4 + (i / 15) * 0.6
                                        }} />;
                                    })}
                                </div>
                            </div>

                            {/* HR */}
                            <div className="card" style={{ padding: 20 }}>
                                <h4 style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                    Heart Rate
                                </h4>
                                <div style={{ fontSize: '2.2rem', fontWeight: 800, color: vitals?.hr > 110 ? 'var(--risk-critical)' : 'var(--accent-purple)' }}>
                                    {vitals?.hr?.toFixed(0)} <span style={{ fontSize: '1rem', fontWeight: 400, color: 'var(--text-muted)' }}>bpm</span>
                                </div>
                                <div style={{ marginTop: 8, fontSize: '0.8rem', color: vitals?.hr_change > 5 ? 'var(--risk-high)' : 'var(--text-muted)' }}>
                                    Change: {vitals?.hr_change > 0 ? '+' : ''}{vitals?.hr_change?.toFixed(1)}
                                </div>
                                <div style={{ marginTop: 12, display: 'flex', gap: 2, alignItems: 'end', height: 32 }}>
                                    {vitalsHistory.slice(-15).map((v, i) => {
                                        const h = Math.max(4, ((v.hr - 40) / 100) * 32);
                                        return <div key={i} style={{
                                            width: 4, height: h, borderRadius: 2,
                                            background: v.hr > 110 ? 'var(--risk-critical)' : 'var(--accent-purple)',
                                            opacity: 0.4 + (i / 15) * 0.6
                                        }} />;
                                    })}
                                </div>
                            </div>
                        </div>

                        {/* Anomalies */}
                        {anomalies.length > 0 && (
                            <div style={{ marginBottom: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                                {anomalies.map((a, i) => (
                                    <span key={i} className={`anomaly-marker ${a.severity}`}>
                                        ⚠️ {a.type?.replace(/_/g, ' ')} ({a.feature?.replace('Current_', '')})
                                    </span>
                                ))}
                            </div>
                        )}

                        {/* Middle Row: NL Explanation + Risk History */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
                            {/* NL Explanation */}
                            <div className="card">
                                <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 12 }}>
                                    🧠 AI Explanation
                                </h4>
                                <div className="nl-explanation">
                                    {xai?.nl_explanation || 'Waiting for AI analysis...'}
                                </div>
                            </div>

                            {/* Risk Timeline */}
                            <div className="card">
                                <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 12 }}>
                                    📈 Risk Timeline
                                </h4>
                                <div style={{ display: 'flex', gap: 2, alignItems: 'end', height: 80 }}>
                                    {riskHistory.map((r, i) => {
                                        const h = Math.max(4, r.risk * 80);
                                        const color = r.risk >= 0.75 ? 'var(--risk-critical)' :
                                            r.risk >= 0.50 ? 'var(--risk-high)' :
                                                r.risk >= 0.30 ? 'var(--risk-moderate)' : 'var(--risk-low)';
                                        return <div key={i} style={{
                                            flex: 1, height: h, borderRadius: 2, background: color,
                                            opacity: 0.5 + (i / riskHistory.length) * 0.5,
                                            transition: 'height 0.3s ease'
                                        }} />;
                                    })}
                                </div>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8, fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                                    <span>0 min</span>
                                    <span>{latestData?.time_minutes} min</span>
                                </div>

                                {/* Forecast */}
                                {xai?.risk_forecast_5step && (
                                    <div style={{ marginTop: 12 }}>
                                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 6 }}>
                                            📐 5-Step Forecast
                                        </div>
                                        <div style={{ display: 'flex', gap: 6 }}>
                                            {xai.risk_forecast_5step.map((f, i) => (
                                                <div key={i} style={{
                                                    flex: 1, textAlign: 'center', padding: '4px 0',
                                                    background: f >= 0.5 ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)',
                                                    borderRadius: 4, fontSize: '0.75rem', fontWeight: 600,
                                                    color: f >= 0.75 ? 'var(--risk-critical)' : f >= 0.5 ? 'var(--risk-high)' : f >= 0.3 ? 'var(--risk-moderate)' : 'var(--risk-low)'
                                                }}>
                                                    {(f * 100).toFixed(0)}%
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Bottom Row: SHAP + Attention */}
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                            {/* SHAP Top Features */}
                            <div className="card">
                                <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 16 }}>
                                    🔬 Feature Attributions (SHAP)
                                </h4>
                                {xai?.top_features?.map((f, i) => {
                                    const maxVal = Math.max(...(xai.top_features.map(x => x.abs_value) || [0.1]));
                                    const pct = (f.abs_value / maxVal) * 100;
                                    const isPositive = f.direction === 'risk_increasing';

                                    return (
                                        <div key={i} className="shap-bar">
                                            <span className="shap-label">{f.name?.replace(/_/g, ' ')}</span>
                                            <div className="shap-bar-track">
                                                <div className={`shap-bar-fill ${isPositive ? 'positive' : 'negative'}`}
                                                    style={{ width: `${pct}%` }} />
                                            </div>
                                            <span className="shap-value" style={{
                                                color: isPositive ? 'var(--risk-critical)' : 'var(--risk-low)'
                                            }}>
                                                {isPositive ? '+' : ''}{f.value?.toFixed(3)}
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>

                            {/* Attention Weights */}
                            <div className="card">
                                <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 16 }}>
                                    👁️ Temporal Attention
                                </h4>
                                {xai?.attention_weights && (
                                    <>
                                        <div style={{ display: 'flex', gap: 2, alignItems: 'end', height: 60, marginBottom: 8 }}>
                                            {xai.attention_weights.map((w, i) => {
                                                const h = Math.max(3, w * 60 / Math.max(...xai.attention_weights));
                                                const opacity = 0.3 + (w / Math.max(...xai.attention_weights)) * 0.7;
                                                return <div key={i} style={{
                                                    flex: 1, height: h, borderRadius: 2,
                                                    background: `rgba(6, 182, 212, ${opacity})`,
                                                    transition: 'height 0.3s ease'
                                                }} />;
                                            })}
                                        </div>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                                            <span>0 min</span>
                                            <span>Peak: {xai.attention_weights ? `${(xai.attention_weights.indexOf(Math.max(...xai.attention_weights))) * 8} min` : '-'}</span>
                                            <span>232 min</span>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>

                        {/* Session Progress */}
                        <div style={{ marginTop: 16 }}>
                            <div style={{
                                height: 4, background: 'rgba(255,255,255,0.05)', borderRadius: 2, overflow: 'hidden'
                            }}>
                                <div style={{
                                    height: '100%', width: `${((currentStep + 1) / 30) * 100}%`,
                                    background: 'linear-gradient(90deg, var(--accent-cyan), var(--accent-purple))',
                                    borderRadius: 2, transition: 'width 0.5s ease'
                                }} />
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                <span>Step {currentStep + 1} / 30</span>
                                <span>{latestData?.time_minutes || 0} / 232 min</span>
                            </div>
                        </div>
                    </>
                )}

                {/* Session Complete Report */}
                {sessionComplete && !connected && (
                    <div className="card" style={{ marginTop: 24, textAlign: 'center', padding: 40 }}>
                        <p style={{ fontSize: '2rem', marginBottom: 12 }}>✅</p>
                        <h2>Session Complete</h2>
                        <p style={{ color: 'var(--text-muted)', marginTop: 8 }}>
                            {vitalsHistory.length} vitals recorded · {activeAlerts.length} alerts generated
                        </p>
                        <div style={{ marginTop: 20, display: 'flex', gap: 12, justifyContent: 'center' }}>
                            <button className="btn btn-primary" onClick={() => {
                                setSessionComplete(false); setLatestData(null);
                                setVitalsHistory([]); setRiskHistory([]);
                                setActiveAlerts([]); setCurrentStep(0);
                            }}>🔄 New Session</button>
                            <button className="btn btn-ghost" onClick={() => router.push('/patients')}>Back to Patients</button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default function MonitorPage() {
    return (
        <Suspense fallback={<div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}><div className="spinner" /></div>}>
            <MonitorContent />
        </Suspense>
    );
}
