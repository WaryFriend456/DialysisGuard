/**
 * API utility — handles all backend communication.
 */
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

function getToken() {
    if (typeof window !== 'undefined') {
        return localStorage.getItem('token');
    }
    return null;
}

async function request(endpoint, options = {}) {
    const token = getToken();
    const headers = {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
        ...options.headers,
    };

    const res = await fetch(`${API_BASE}${endpoint}`, { ...options, headers });

    if (res.status === 401) {
        if (typeof window !== 'undefined') {
            localStorage.removeItem('token');
            window.location.href = '/login';
        }
        throw new Error('Unauthorized');
    }

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed: ${res.status}`);
    }

    return res.json();
}

// Auth
export const auth = {
    register: (data) => request('/api/auth/register', { method: 'POST', body: JSON.stringify(data) }),
    login: (data) => request('/api/auth/login', { method: 'POST', body: JSON.stringify(data) }),
    me: () => request('/api/auth/me'),
};

// Patients
export const patients = {
    list: (params = '') => request(`/api/patients/?${params}`),
    get: (id) => request(`/api/patients/${id}`),
    create: (data) => request('/api/patients/', { method: 'POST', body: JSON.stringify(data) }),
    update: (id, data) => request(`/api/patients/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
    delete: (id) => request(`/api/patients/${id}`, { method: 'DELETE' }),
};

// Sessions
export const sessions = {
    create: (data) => request('/api/sessions/', { method: 'POST', body: JSON.stringify(data) }),
    get: (id) => request(`/api/sessions/${id}`),
    stop: (id) => request(`/api/sessions/${id}/stop`, { method: 'POST' }),
    report: (id) => request(`/api/sessions/${id}/report`),
    forPatient: (patientId) => request(`/api/sessions/patient/${patientId}`),
};

// Predictions
export const predictions = {
    predict: (data) => request('/api/predict/', { method: 'POST', body: JSON.stringify(data) }),
    riskAssessment: (data) => request('/api/predict/risk-assessment', { method: 'POST', body: JSON.stringify(data) }),
};

// Alerts
export const alerts = {
    list: (params = '') => request(`/api/alerts/?${params}`),
    acknowledge: (id) => request(`/api/alerts/${id}/acknowledge`, { method: 'POST' }),
    stats: () => request('/api/alerts/stats'),
};

// XAI
export const explain = {
    shap: (data) => request('/api/explain/shap', { method: 'POST', body: JSON.stringify(data) }),
    attention: (data) => request('/api/explain/attention', { method: 'POST', body: JSON.stringify(data) }),
    whatIf: (data) => request('/api/explain/what-if', { method: 'POST', body: JSON.stringify(data) }),
    counterfactual: (data) => request('/api/explain/counterfactual', { method: 'POST', body: JSON.stringify(data) }),
    sensitivity: (data) => request('/api/explain/sensitivity', { method: 'POST', body: JSON.stringify(data) }),
    modelCard: () => request('/api/explain/model-card'),
};

// WebSocket
export function connectMonitor(sessionId, onMessage, onError, onClose) {
    const token = getToken();
    const ws = new WebSocket(`${WS_BASE}/ws/monitor/${sessionId}`);

    ws.onopen = () => console.log('WebSocket connected');
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            onMessage(data);
        } catch (e) {
            console.error('WebSocket parse error:', e);
        }
    };
    ws.onerror = (e) => onError?.(e);
    ws.onclose = (e) => onClose?.(e);

    return ws;
}
