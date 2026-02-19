'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import PageShell from '@/components/PageShell';
import { useAuth } from '@/contexts/AuthContext';
import { patients, alerts as alertsApi } from '@/lib/api';
import { cn } from '@/lib/utils';
import {
    Users, Bell, BellOff, Activity, MonitorDot, UserPlus,
    Brain, ArrowRight, Loader2,
} from 'lucide-react';

const RISK_BG = { CRITICAL: 'badge-risk-critical', HIGH: 'badge-risk-high', MODERATE: 'badge-risk-moderate', LOW: 'badge-risk-low' };

export default function DoctorDashboard() {
    const { user, loading } = useAuth();
    const router = useRouter();
    const [stats, setStats] = useState({ patients: 0, alerts: { total: 0, unacknowledged: 0, by_severity: {} } });
    const [patientList, setPatientList] = useState([]);
    const [recentAlerts, setRecentAlerts] = useState([]);

    useEffect(() => { if (!loading && !user) router.push('/login'); }, [user, loading, router]);

    useEffect(() => {
        if (user) {
            patients.list('limit=10').then(r => { setPatientList(r.patients || []); setStats(p => ({ ...p, patients: r.total || 0 })); }).catch(() => { });
            alertsApi.stats().then(r => setStats(p => ({ ...p, alerts: r }))).catch(() => { });
            alertsApi.list('limit=5').then(r => setRecentAlerts(r.alerts || [])).catch(() => { });
        }
    }, [user]);

    if (loading || !user) return <div className="flex min-h-screen items-center justify-center bg-bg-primary"><Loader2 className="h-8 w-8 animate-spin text-accent" /></div>;

    const a = stats.alerts;

    return (
        <PageShell>
            <div className="mb-8">
                <h1 className="text-2xl font-bold text-text-primary">Welcome, Dr. {user.name?.split(' ').pop()}</h1>
                <p className="mt-1 text-sm text-text-muted">AI-Driven Hemodialysis Monitoring Dashboard</p>
            </div>

            {/* Stats */}
            <div className="mb-7 grid grid-cols-4 gap-4">
                {[
                    { label: 'Total Patients', value: stats.patients, icon: Users, color: 'text-accent' },
                    { label: 'Active Sessions', value: 0, icon: Activity, color: 'text-risk-low' },
                    { label: 'Unack. Alerts', value: a.unacknowledged || 0, icon: BellOff, color: a.unacknowledged > 0 ? 'text-risk-critical' : 'text-text-muted' },
                    { label: 'Total Alerts', value: a.total || 0, icon: Bell, color: 'text-chart-5' },
                ].map((s, i) => (
                    <div key={i} className="card p-5">
                        <div className="flex items-center justify-between">
                            <p className="text-xs font-medium uppercase tracking-wider text-text-muted">{s.label}</p>
                            <s.icon className={cn('h-4 w-4', s.color)} />
                        </div>
                        <p className={cn('mt-2 text-3xl font-extrabold tabular-nums', s.color)}>{s.value}</p>
                    </div>
                ))}
            </div>

            {/* Alert Breakdown */}
            {a.by_severity && (
                <div className="card mb-6 p-5">
                    <h3 className="mb-4 text-sm font-semibold text-text-primary">Alert Breakdown</h3>
                    <div className="flex gap-4">
                        {['CRITICAL', 'HIGH', 'MODERATE', 'LOW'].map((sev) => (
                            <div key={sev} className="flex items-center gap-2">
                                <span className={cn('rounded-md px-2 py-0.5 text-[11px] font-bold', RISK_BG[sev])}>{sev}</span>
                                <span className="text-lg font-bold text-text-primary">{a.by_severity[sev] || 0}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Two-column: Patients | Alerts */}
            <div className="mb-6 grid grid-cols-2 gap-6">
                {/* Recent Patients */}
                <div className="card p-5">
                    <div className="mb-4 flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-text-primary">Recent Patients</h3>
                        <button onClick={() => router.push('/patients')} className="text-xs font-medium text-accent hover:text-accent-hover cursor-pointer">View All</button>
                    </div>
                    {patientList.length === 0 ? (
                        <p className="text-sm text-text-muted">No patients yet. Add your first patient.</p>
                    ) : (
                        <div className="space-y-2">
                            {patientList.slice(0, 5).map((p) => (
                                <div key={p.id} className="flex items-center justify-between rounded-lg bg-bg-secondary px-4 py-2.5 transition-colors hover:bg-surface-hover cursor-pointer" onClick={() => router.push(`/patients/${p.id}`)}>
                                    <div>
                                        <p className="text-sm font-medium text-text-primary">{p.name || `Patient #${p.id?.slice(0, 6)}`}</p>
                                        <p className="text-xs text-text-muted">Age {p.age} · {p.gender} · {p.disease_severity || 'Unknown'}</p>
                                    </div>
                                    <button className="rounded-lg bg-accent/10 px-3 py-1 text-xs font-medium text-accent hover:bg-accent/20 cursor-pointer" onClick={(e) => { e.stopPropagation(); router.push(`/monitor?patient=${p.id}`); }}>
                                        Monitor
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Recent Alerts */}
                <div className="card p-5">
                    <div className="mb-4 flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-text-primary">Recent Alerts</h3>
                        <button onClick={() => router.push('/alerts')} className="text-xs font-medium text-accent hover:text-accent-hover cursor-pointer">View All</button>
                    </div>
                    {recentAlerts.length === 0 ? (
                        <p className="text-sm text-text-muted">No alerts yet.</p>
                    ) : (
                        <div className="space-y-2">
                            {recentAlerts.map((al, i) => (
                                <div key={i} className={cn('flex items-center gap-3 rounded-lg border px-3 py-2 text-sm',
                                    al.severity === 'CRITICAL' ? 'border-risk-critical/20 bg-risk-critical-bg' :
                                        al.severity === 'HIGH' ? 'border-risk-high/20 bg-risk-high-bg' :
                                            'border-risk-moderate/20 bg-risk-moderate-bg'
                                )}>
                                    <span className={cn('rounded-md px-1.5 py-0.5 text-[10px] font-bold', RISK_BG[al.severity])}>{al.severity}</span>
                                    <span className="flex-1 truncate text-xs text-text-secondary">{al.message}</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Quick Actions */}
            <div className="card p-5">
                <h3 className="mb-4 text-sm font-semibold text-text-primary">Quick Actions</h3>
                <div className="flex gap-3">
                    <button onClick={() => router.push('/patients?action=add')} className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-bg-primary hover:bg-accent-hover cursor-pointer">
                        <UserPlus className="h-4 w-4" /> Add Patient
                    </button>
                    <button onClick={() => router.push('/dashboard/command')} className="flex items-center gap-2 rounded-lg border border-border-subtle px-4 py-2 text-sm font-medium text-text-secondary hover:bg-surface-hover cursor-pointer">
                        <MonitorDot className="h-4 w-4" /> Command Center
                    </button>
                    <button onClick={() => router.push('/model-info')} className="flex items-center gap-2 rounded-lg border border-border-subtle px-4 py-2 text-sm font-medium text-text-secondary hover:bg-surface-hover cursor-pointer">
                        <Brain className="h-4 w-4" /> Model Info
                    </button>
                </div>
            </div>
        </PageShell>
    );
}
