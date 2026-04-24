'use client';

import { useMemo, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, MonitorDot } from 'lucide-react';
import Sidebar from './Sidebar';
import { useMonitoring } from '@/contexts/MonitoringContext';
import { useAuth } from '@/contexts/AuthContext';

const pageTitles = [
    { prefix: '/dashboard/doctor', title: 'Doctor Dashboard', description: 'Operational overview for physicians and senior clinicians.' },
    { prefix: '/dashboard/caregiver', title: 'Caregiver Dashboard', description: 'Focused queue for bedside monitoring and alert response.' },
    { prefix: '/dashboard/command', title: 'Command Center', description: 'Secondary room overview for concurrent patient supervision.' },
    { prefix: '/patients', title: 'Patients', description: 'Clinical profiles, dialysis baselines, and monitoring history.' },
    { prefix: '/monitor', title: 'Monitoring', description: 'Live vitals, risk progression, alerts, and explainability.' },
    { prefix: '/alerts', title: 'Alerts', description: 'Review and acknowledge hemodynamic risk alerts.' },
    { prefix: '/model-info', title: 'Model Transparency', description: 'Reference material for model behavior, data, and limitations.' },
];

export default function PageShell({ children }) {
    const pathname = usePathname();
    const { user } = useAuth();
    const { session, patientSummary, hasActiveSession } = useMonitoring();
    const [sidebarOpen, setSidebarOpen] = useState(false);

    const pageMeta = useMemo(
        () => pageTitles.find((item) => pathname.startsWith(item.prefix)) || pageTitles[0],
        [pathname]
    );

    return (
        <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,rgba(6,182,212,0.12),transparent_28%),radial-gradient(circle_at_top_right,rgba(249,115,22,0.08),transparent_22%),var(--color-bg-primary)]">
            <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

            <div className="lg:pl-72">
                <header className="sticky top-0 z-30 border-b border-border-subtle bg-bg-primary/85 backdrop-blur">
                    <div className="flex flex-col gap-4 px-4 py-4 sm:px-6 lg:px-8">
                        <div className="flex items-center justify-between gap-4">
                            <div className="flex items-center gap-3">
                                <button
                                    onClick={() => setSidebarOpen(true)}
                                    className="rounded-2xl border border-border-subtle p-2 text-text-secondary hover:bg-surface-hover hover:text-text-primary lg:hidden"
                                >
                                    <Menu className="h-5 w-5" />
                                </button>
                                <div>
                                    <p className="text-xs uppercase tracking-[0.2em] text-text-muted">{user?.role || 'workspace'}</p>
                                    <h1 className="text-2xl font-semibold text-text-primary">{pageMeta.title}</h1>
                                </div>
                            </div>
                            <div className="hidden rounded-2xl border border-border-subtle bg-surface px-4 py-2 text-right sm:block">
                                <p className="text-sm font-medium text-text-primary">{user?.name}</p>
                                <p className="text-xs text-text-muted">Demo mode</p>
                            </div>
                        </div>
                        <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
                            <p className="max-w-3xl text-sm text-text-secondary">{pageMeta.description}</p>
                            {hasActiveSession && session && (
                                <Link
                                    href={`/monitor?patient=${session.patient_id}`}
                                    className="inline-flex items-center gap-2 rounded-2xl border border-accent/30 bg-accent/10 px-4 py-2 text-sm font-medium text-accent transition-colors hover:bg-accent/15"
                                >
                                    <MonitorDot className="h-4 w-4" />
                                    {patientSummary?.name ? `Resume ${patientSummary.name}` : 'Resume active monitoring'}
                                    <span className="text-text-muted">Step {session.current_step || 0}/{session.total_steps || 30}</span>
                                </Link>
                            )}
                        </div>
                    </div>
                </header>

                <main className="px-4 py-6 sm:px-6 lg:px-8">{children}</main>
            </div>
        </div>
    );
}
