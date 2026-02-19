'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import PageShell from '@/components/PageShell';
import { useAuth } from '@/contexts/AuthContext';
import { patients, alerts as alertsApi } from '@/lib/api';
import { cn } from '@/lib/utils';
import { CheckCircle2, Loader2 } from 'lucide-react';

const RISK_BG = { CRITICAL: 'badge-risk-critical', HIGH: 'badge-risk-high', MODERATE: 'badge-risk-moderate', LOW: 'badge-risk-low' };

export default function CaregiverDashboard() {
    const { user, loading } = useAuth();
    const router = useRouter();
    const [patientList, setPatientList] = useState([]);
    const [alertList, setAlertList] = useState([]);

    useEffect(() => { if (!loading && !user) router.push('/login'); }, [user, loading, router]);

    useEffect(() => {
        if (user) {
            patients.list('limit=20').then(r => setPatientList(r.patients || [])).catch(() => { });
            alertsApi.list('limit=10&acknowledged=false').then(r => setAlertList(r.alerts || [])).catch(() => { });
        }
    }, [user]);

    if (loading || !user) return <div className="flex min-h-screen items-center justify-center bg-bg-primary"><Loader2 className="h-8 w-8 animate-spin text-accent" /></div>;

    return (
        <PageShell>
            <div className="mb-8">
                <h1 className="text-2xl font-bold text-text-primary">Caregiver Dashboard</h1>
                <p className="mt-1 text-sm text-text-muted">Welcome, {user.name}</p>
            </div>

            {/* Active Alerts */}
            {alertList.length > 0 && (
                <div className="card mb-6 p-5">
                    <h3 className="mb-4 text-sm font-semibold text-text-primary">⚠ Active Alerts</h3>
                    <div className="space-y-2">
                        {alertList.map((a, i) => (
                            <div key={i} className={cn('flex items-center gap-3 rounded-lg border px-4 py-2.5 text-sm',
                                a.severity === 'CRITICAL' ? 'border-risk-critical/20 bg-risk-critical-bg' :
                                    a.severity === 'HIGH' ? 'border-risk-high/20 bg-risk-high-bg' :
                                        'border-risk-moderate/20 bg-risk-moderate-bg'
                            )}>
                                <span className={cn('rounded-md px-2 py-0.5 text-[10px] font-bold', RISK_BG[a.severity])}>{a.severity}</span>
                                <span className="flex-1 text-text-secondary">{a.message}</span>
                                <button onClick={async () => {
                                    try { await alertsApi.acknowledge(a.id); setAlertList(p => p.filter(al => al.id !== a.id)); } catch { }
                                }} className="rounded-md px-2 py-1 text-xs font-medium text-text-secondary hover:bg-surface-hover cursor-pointer">
                                    <CheckCircle2 className="h-4 w-4" />
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Patient List */}
            <div className="card p-5">
                <h3 className="mb-4 text-sm font-semibold text-text-primary">Patients</h3>
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
                    {patientList.map((p) => (
                        <div key={p.id} className="cursor-pointer rounded-lg border border-border-subtle bg-bg-secondary px-4 py-3 transition-all hover:border-border hover:bg-surface-hover"
                            onClick={() => router.push(`/monitor?patient=${p.id}`)}>
                            <p className="text-sm font-medium text-text-primary">{p.name || `Patient #${p.id?.slice(0, 6)}`}</p>
                            <p className="mt-1 text-xs text-text-muted">{p.age} yrs · {p.gender} · {p.disease_severity}</p>
                        </div>
                    ))}
                </div>
            </div>
        </PageShell>
    );
}
