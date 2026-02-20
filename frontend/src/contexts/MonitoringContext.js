'use client';
import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { sessions, connectMonitor } from '@/lib/api';

const MonitoringContext = createContext(null);

export function MonitoringProvider({ children }) {
    const [status, setStatus] = useState('idle'); // idle, working, error
    const [session, setSession] = useState(null);
    const [connected, setConnected] = useState(false);
    const [data, setData] = useState(null);
    const [history, setHistory] = useState({ risk: [], vitals: [] });
    const [alerts, setAlerts] = useState([]);
    const [error, setError] = useState(null);

    // Refs for WebSocket to prevent closure staleness
    const wsRef = useRef(null);
    const retryCount = useRef(0);

    // ─── Restore Session on Mount ───
    useEffect(() => {
        const storedSessionId = localStorage.getItem('active_monitoring_session');
        if (storedSessionId) {
            console.log("Found active session, attempting restore:", storedSessionId);
            restoreSession(storedSessionId);
        }
    }, []);

    const restoreSession = async (sessionId) => {
        setStatus('working');
        try {
            const sess = await sessions.get(sessionId);
            if (sess.status === 'active' || sess.status === 'ACTIVE') {
                setSession(sess);
                connect(sessionId);
            } else {
                // Session ended while we were away
                localStorage.removeItem('active_monitoring_session');
                setStatus('idle');
            }
        } catch (err) {
            console.error("Failed to restore session:", err);
            localStorage.removeItem('active_monitoring_session');
            setStatus('error');
        }
    };

    const startSession = async (patientId, riskProfile) => {
        if (status === 'working') return;
        setStatus('working');
        setError(null);
        try {
            const newSession = await sessions.create({
                patient_id: patientId,
                risk_profile: riskProfile
            });
            setSession(newSession);
            localStorage.setItem('active_monitoring_session', newSession.id);
            connect(newSession.id);
        } catch (err) {
            setError(err.message);
            setStatus('error');
        }
    };

    const stopSession = async () => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setConnected(false);
        if (session?.id) {
            try {
                await sessions.stop(session.id);
            } catch (err) {
                console.error("Failed to stop session backend:", err);
            }
        }
        localStorage.removeItem('active_monitoring_session');
        setSession(null);
        setData(null);
        setHistory({ risk: [], vitals: [] });
        setAlerts([]);
        setStatus('idle');
    };

    const connect = useCallback((sessionId) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        console.log("Connecting WebSocket for session:", sessionId);

        wsRef.current = connectMonitor(
            sessionId,
            (msg) => {
                if (msg.type === 'session_start') {
                    // Initialize if needed
                } else if (msg.type === 'session_complete') {
                    setConnected(false);
                    localStorage.removeItem('active_monitoring_session');
                    setStatus('idle');
                } else if (msg.type === 'monitoring_data') {
                    setData(msg);

                    // Update History efficienty
                    setHistory(prev => {
                        // Keep last 30 points
                        const newRisk = [...prev.risk, {
                            step: msg.step,
                            val: msg.prediction.risk_probability
                        }].slice(-30);

                        const newVitals = [...prev.vitals, {
                            step: msg.step,
                            bp: msg.vitals.bp,
                            hr: msg.vitals.hr
                        }].slice(-30);

                        return { risk: newRisk, vitals: newVitals };
                    });

                    if (msg.alert) {
                        setAlerts(prev => [msg.alert, ...prev]);
                    }
                }
            },
            (err) => {
                console.error("WS Error:", err);
                // Simple retry logic could go here
            },
            () => {
                console.log("WS Closed");
                setConnected(false);
            }
        );
        setConnected(true);
        setStatus('working');
    }, []);

    return (
        <MonitoringContext.Provider value={{
            session,
            connected,
            data,
            history,
            alerts,
            error,
            status,
            startSession,
            stopSession,
            restoreSession
        }}>
            {children}
        </MonitoringContext.Provider>
    );
}

export const useMonitoring = () => useContext(MonitoringContext);
