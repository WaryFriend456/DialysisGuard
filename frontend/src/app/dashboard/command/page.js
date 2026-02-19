'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import PageShell from '@/components/PageShell';
import { useAuth } from '@/contexts/AuthContext';
import { patients as patientsApi } from '@/lib/api';
import { cn } from '@/lib/utils';
import { MonitorDot, Users, Loader2, Monitor } from 'lucide-react';

const SEV_STYLE = { Mild: 'badge-risk-low', Moderate: 'badge-risk-moderate', Severe: 'badge-risk-high', Critical: 'badge-risk-critical' };

export default function CommandCenter() {
    const { user, loading } = useAuth();
    const router = useRouter();
    const [patientList, setPatientList] = useState([]);

    useEffect(() => { if (!loading && !user) router.push('/login'); }, [user, loading, router]);
    useEffect(() => { if (user) patientsApi.list('limit=50').then(r => setPatientList(r.patients || [])).catch(() => { }); }, [user]);

    if (loading || !user) return <div className="flex min-h-screen items-center justify-center bg-bg-primary"><Loader2 className="h-8 w-8 animate-spin text-accent" /></div>;

    return (
        <PageShell>
            <div className="mb-6">
                <h1 className="flex items-center gap-3 text-2xl font-bold text-text-primary">
                    <MonitorDot className="h-6 w-6 text-accent" /> Multi-Patient Command Center
                </h1>
                <p className="mt-1 text-sm text-text-muted">Overview of all patients — click any card to start monitoring</p>
            </div>

            {patientList.length === 0 ? (
                <div className="card flex flex-col items-center py-16 text-center">
                    <Users className="mb-3 h-12 w-12 text-text-muted/40" />
                    <p className="text-text-muted">No patients available. Add patients first.</p>
                    <button onClick={() => router.push('/patients')} className="mt-4 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-bg-primary hover:bg-accent-hover cursor-pointer">Go to Patients</button>
                </div>
            ) : (
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
                    {patientList.map((p) => {
                        const sev = p.disease_severity || 'Moderate';
                        return (
                            <div key={p.id} className="card card-hover cursor-pointer p-5" onClick={() => router.push(`/monitor?patient=${p.id}`)}>
                                <div className="mb-3 flex items-start justify-between">
                                    <h3 className="text-sm font-semibold text-text-primary">{p.name || `Patient #${p.id?.slice(0, 6)}`}</h3>
                                    <span className={cn('rounded-md px-2 py-0.5 text-[11px] font-semibold', SEV_STYLE[sev] || SEV_STYLE.Moderate)}>{sev}</span>
                                </div>
                                <div className="grid grid-cols-2 gap-1 text-xs text-text-secondary">
                                    <span>Age: {p.age}</span><span>Gender: {p.gender}</span>
                                    <span>Weight: {p.weight} kg</span><span>Duration: {p.dialysis_duration}h</span>
                                </div>
                                <div className="mt-3 flex gap-2">
                                    {p.diabetes && <span className="rounded-full bg-risk-moderate-bg px-2 py-0.5 text-[10px] font-medium text-risk-moderate">Diabetes</span>}
                                    {p.hypertension && <span className="rounded-full bg-risk-critical-bg px-2 py-0.5 text-[10px] font-medium text-risk-critical">Hypertension</span>}
                                </div>
                                <button className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg bg-accent/10 py-2 text-xs font-medium text-accent hover:bg-accent/20 cursor-pointer"
                                    onClick={(e) => { e.stopPropagation(); router.push(`/monitor?patient=${p.id}`); }}>
                                    <Monitor className="h-3.5 w-3.5" /> Start Monitor
                                </button>
                            </div>
                        );
                    })}
                </div>
            )}
        </PageShell>
    );
}
