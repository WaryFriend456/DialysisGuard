'use client';

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { connectMonitor, sessions } from '@/lib/api';

const MonitoringContext = createContext(null);
const STORAGE_KEY = 'active_monitoring_session';

const emptyHistory = { risk: [], vitals: [] };

function readStoredSession() {
    if (typeof window === 'undefined') return null;
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch {
        localStorage.removeItem(STORAGE_KEY);
        return null;
    }
}

function persistSession(meta) {
    if (typeof window === 'undefined') return;
    if (!meta?.sessionId) {
        localStorage.removeItem(STORAGE_KEY);
        return;
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(meta));
}

export function MonitoringProvider({ children }) {
    const [status, setStatus] = useState('idle');
    const [session, setSession] = useState(null);
    const [connected, setConnected] = useState(false);
    const [data, setData] = useState(null);
    const [history, setHistory] = useState(emptyHistory);
    const [alerts, setAlerts] = useState([]);
    const [error, setError] = useState(null);
    const [patientSummary, setPatientSummary] = useState(null);

    const wsRef = useRef(null);
    const stopIntentRef = useRef(false);
    const bootstrappedRef = useRef(false);

    const clearActiveState = useCallback(() => {
        persistSession(null);
        setSession(null);
        setConnected(false);
        setData(null);
        setHistory(emptyHistory);
        setAlerts([]);
        setPatientSummary(null);
        setError(null);
    }, []);

    const hydrateSession = useCallback((nextSession, extras = {}) => {
        if (!nextSession) return;
        const merged = {
            ...nextSession,
            ...extras,
        };
        setSession(merged);
        persistSession({
            sessionId: merged.id,
            patientId: merged.patient_id,
            patientName: extras.patientName || patientSummary?.name || null,
        });
    }, [patientSummary]);

    const disconnectSocket = useCallback(() => {
        if (wsRef.current) {
            try {
                wsRef.current.close();
            } catch {
                // ignore close errors
            }
            wsRef.current = null;
        }
    }, []);

    const handleMessage = useCallback((msg) => {
        if (!msg?.type) return;

        if (msg.type === 'session_start') {
            setConnected(true);
            setStatus('working');
            setPatientSummary({
                id: msg.patient_id,
                name: msg.patient_name || null,
            });
            setSession((prev) => {
                const next = {
                    ...(prev || {}),
                    id: msg.session_id,
                    patient_id: msg.patient_id,
                    risk_profile: msg.risk_profile,
                    current_step: msg.current_step ?? prev?.current_step ?? 0,
                    total_steps: msg.total_steps ?? prev?.total_steps ?? 30,
                    can_resume: true,
                    status: 'active',
                };
                persistSession({
                    sessionId: next.id,
                    patientId: next.patient_id,
                    patientName: msg.patient_name || null,
                });
                return next;
            });
            return;
        }

        if (msg.type === 'monitoring_data') {
            setStatus('working');
            setData(msg);
            setPatientSummary((prev) => ({
                id: msg.patient_id || prev?.id || session?.patient_id || null,
                name: msg.patient_name || prev?.name || null,
            }));
            setSession((prev) => prev ? {
                ...prev,
                current_step: (msg.step ?? prev.current_step ?? 0) + 1,
                total_steps: prev.total_steps ?? 30,
                can_resume: true,
                status: 'active',
            } : prev);
            setHistory((prev) => ({
                risk: [
                    ...prev.risk,
                    { step: msg.step, val: msg.prediction?.risk_probability ?? 0, level: msg.prediction?.risk_level ?? 'low' },
                ].slice(-30),
                vitals: [
                    ...prev.vitals,
                    { step: msg.step, bp: msg.vitals?.bp ?? 0, hr: msg.vitals?.hr ?? 0, time: msg.time_minutes ?? 0 },
                ].slice(-30),
            }));
            if (msg.alert) {
                setAlerts((prev) => [msg.alert, ...prev].slice(0, 20));
            }
            if (msg.escalation_alerts?.length) {
                setAlerts((prev) => [...msg.escalation_alerts, ...prev].slice(0, 20));
            }
            return;
        }

        if (msg.type === 'session_complete') {
            disconnectSocket();
            persistSession(null);
            setConnected(false);
            setStatus('idle');
            setSession((prev) => prev ? {
                ...prev,
                current_step: msg.total_steps ?? prev.current_step,
                can_resume: false,
                status: 'completed',
            } : prev);
            return;
        }

        if (msg.type === 'error') {
            setError(msg.message || 'Monitoring error');
            setConnected(false);
            setStatus('error');
        }
    }, [disconnectSocket, session?.patient_id]);

    const connect = useCallback((sessionId) => {
        if (!sessionId) return;
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        disconnectSocket();
        stopIntentRef.current = false;
        wsRef.current = connectMonitor(
            sessionId,
            handleMessage,
            () => {
                setError('WebSocket connection error');
                setConnected(false);
                setStatus('error');
            },
            () => {
                wsRef.current = null;
                setConnected(false);
                setStatus((prev) => (stopIntentRef.current ? 'idle' : (session ? 'paused' : prev)));
                setSession((prev) => prev && !stopIntentRef.current ? {
                    ...prev,
                    status: 'paused',
                    can_resume: true,
                } : prev);
            }
        );
    }, [disconnectSocket, handleMessage, session]);

    const restoreSession = useCallback(async (sessionId) => {
        if (!sessionId) return null;
        setStatus('restoring');
        try {
            const active = await sessions.get(sessionId);
            if (!['active', 'paused'].includes(active.status)) {
                clearActiveState();
                setStatus('idle');
                return null;
            }
            hydrateSession(active);
            connect(active.id);
            return active;
        } catch (err) {
            clearActiveState();
            setStatus('idle');
            setError(err.message || 'Unable to restore session');
            return null;
        }
    }, [clearActiveState, connect, hydrateSession]);

    const getResumableSession = useCallback(async (patientId) => {
        const res = await sessions.current(patientId);
        return res?.session || null;
    }, []);

    const ensurePatientSession = useCallback(async (patientId, options = {}) => {
        if (!patientId) return null;
        const { autoStart = false, riskProfile } = options;

        if (session?.patient_id === patientId && session?.id) {
            if (!connected && ['active', 'paused'].includes(session.status)) {
                connect(session.id);
            }
            return session;
        }

        setStatus(autoStart ? 'starting' : 'restoring');
        setError(null);

        try {
            const resumable = await getResumableSession(patientId);
            if (resumable) {
                hydrateSession(resumable);
                connect(resumable.id);
                return resumable;
            }

            if (!autoStart) {
                setStatus('idle');
                return null;
            }

            const created = await sessions.create({
                patient_id: patientId,
                risk_profile: riskProfile || undefined,
            });
            hydrateSession(created);
            setHistory(emptyHistory);
            setAlerts([]);
            setData(null);
            connect(created.id);
            return created;
        } catch (err) {
            setStatus('error');
            setError(err.message || 'Unable to start monitoring');
            return null;
        }
    }, [connect, connected, getResumableSession, hydrateSession, session]);

    const startSession = useCallback(async (patientId, riskProfile = null) => {
        return ensurePatientSession(patientId, { autoStart: true, riskProfile });
    }, [ensurePatientSession]);

    const stopSession = useCallback(async () => {
        if (!session?.id) return;
        stopIntentRef.current = true;
        disconnectSocket();
        try {
            await sessions.stop(session.id);
        } catch (err) {
            setError(err.message || 'Unable to stop monitoring');
        } finally {
            clearActiveState();
            setStatus('idle');
            stopIntentRef.current = false;
        }
    }, [clearActiveState, disconnectSocket, session]);

    useEffect(() => {
        if (bootstrappedRef.current) return;
        bootstrappedRef.current = true;
        const stored = readStoredSession();
        if (stored?.sessionId) {
            restoreSession(stored.sessionId);
        }
    }, [restoreSession]);

    const value = useMemo(() => ({
        session,
        connected,
        data,
        history,
        alerts,
        error,
        status,
        patientSummary,
        activeSession: session,
        hasActiveSession: Boolean(session?.id && ['active', 'paused'].includes(session.status)),
        startSession,
        stopSession,
        restoreSession,
        ensurePatientSession,
        clearMonitoring: clearActiveState,
    }), [
        alerts,
        clearActiveState,
        connected,
        data,
        ensurePatientSession,
        error,
        history,
        patientSummary,
        restoreSession,
        session,
        startSession,
        status,
        stopSession,
    ]);

    return (
        <MonitoringContext.Provider value={value}>
            {children}
        </MonitoringContext.Provider>
    );
}

export function useMonitoring() {
    const context = useContext(MonitoringContext);
    if (!context) {
        throw new Error('useMonitoring must be used within MonitoringProvider');
    }
    return context;
}
