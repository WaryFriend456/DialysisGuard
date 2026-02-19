'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import PageShell from '@/components/PageShell';
import { useAuth } from '@/contexts/AuthContext';
import { alerts as alertsApi } from '@/lib/api';
import { cn } from '@/lib/utils';
import { Bell, CheckCircle2, Loader2, BellOff } from 'lucide-react';

const RISK_BG = { CRITICAL: 'badge-risk-critical', HIGH: 'badge-risk-high', MODERATE: 'badge-risk-moderate', LOW: 'badge-risk-low' };
const RISK_COLOR = { CRITICAL: 'text-risk-critical', HIGH: 'text-risk-high', MODERATE: 'text-risk-moderate', LOW: 'text-risk-low' };

export default function AlertsPage() {
    const { user, loading } = useAuth();
    const router = useRouter();
    const [alertList, setAlertList] = useState([]);
    const [stats, setStats] = useState(null);
    const [filter, setFilter] = useState('');

    useEffect(() => { if (!loading && !user) router.push('/login'); }, [user, loading, router]);

    useEffect(() => {
        if (user) {
            const params = filter ? `severity=${filter}&limit=100` : 'limit=100';
            alertsApi.list(params).then(r => setAlertList(r.alerts || [])).catch(() => { });
            alertsApi.stats().then(setStats).catch(() => { });
        }
    }, [user, filter]);

    if (loading || !user) return <div className="flex min-h-screen items-center justify-center bg-bg-primary"><Loader2 className="h-8 w-8 animate-spin text-accent" /></div>;

    return (
        <PageShell>
            <h1 className="mb-6 text-2xl font-bold text-text-primary">Alerts</h1>

            {/* Stats */}
            {stats && (
                <div className="mb-6 grid grid-cols-5 gap-4">
                    {[
                        { label: 'Total', value: stats.total, color: 'text-text-primary' },
                        { label: 'Unacknowledged', value: stats.unacknowledged, color: 'text-risk-critical' },
                        ...['CRITICAL', 'HIGH', 'MODERATE'].map((s) => ({
                            label: s, value: stats.by_severity?.[s] || 0, color: RISK_COLOR[s],
                        })),
                    ].map((s, i) => (
                        <div key={i} className="card p-4 text-center">
                            <p className={cn('text-2xl font-extrabold tabular-nums', s.color)}>{s.value}</p>
                            <p className="mt-1 text-[11px] font-medium uppercase tracking-wider text-text-muted">{s.label}</p>
                        </div>
                    ))}
                </div>
            )}

            {/* Filter Tabs */}
            <div className="mb-5 flex gap-2">
                {['', 'CRITICAL', 'HIGH', 'MODERATE', 'LOW'].map((f) => (
                    <button key={f} onClick={() => setFilter(f)}
                        className={cn(
                            'rounded-lg px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer',
                            filter === f ? 'bg-accent text-bg-primary' : 'bg-surface text-text-secondary hover:bg-surface-hover'
                        )}>
                        {f || 'All'}
                    </button>
                ))}
            </div>

            {/* Alerts List */}
            <div className="space-y-2">
                {alertList.length === 0 ? (
                    <div className="card flex flex-col items-center py-16 text-center">
                        <BellOff className="mb-3 h-12 w-12 text-text-muted/40" />
                        <p className="text-text-muted">No alerts found.</p>
                    </div>
                ) : alertList.map((a, i) => (
                    <div key={i} className={cn(
                        'flex items-center gap-3 rounded-xl border px-4 py-3',
                        a.severity === 'CRITICAL' ? 'border-risk-critical/20 bg-risk-critical-bg' :
                            a.severity === 'HIGH' ? 'border-risk-high/20 bg-risk-high-bg' :
                                a.severity === 'MODERATE' ? 'border-risk-moderate/20 bg-risk-moderate-bg' :
                                    'border-border-subtle bg-bg-secondary'
                    )}>
                        <span className={cn('rounded-md px-2 py-0.5 text-[10px] font-bold', RISK_BG[a.severity])}>{a.severity}</span>
                        <div className="flex-1 min-w-0">
                            <p className="text-sm text-text-primary">{a.message}</p>
                            {a.nl_explanation && <p className="mt-0.5 truncate text-xs text-text-muted">{a.nl_explanation}</p>}
                        </div>
                        {a.acknowledged ? (
                            <span className="flex items-center gap-1 text-xs text-risk-low"><CheckCircle2 className="h-3.5 w-3.5" /> Ack'd</span>
                        ) : (
                            <button onClick={async () => {
                                try { await alertsApi.acknowledge(a.id); setAlertList(p => p.map(al => al.id === a.id ? { ...al, acknowledged: true } : al)); } catch { }
                            }} className="rounded-md bg-surface px-2.5 py-1 text-xs font-medium text-text-secondary hover:bg-surface-hover cursor-pointer">
                                Acknowledge
                            </button>
                        )}
                    </div>
                ))}
            </div>
        </PageShell>
    );
}
