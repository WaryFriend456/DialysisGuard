'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { Laptop, Menu, MonitorDot, Moon, Sun } from 'lucide-react';
import Sidebar from './Sidebar';
import { useMonitoring } from '@/contexts/MonitoringContext';
import { useAuth } from '@/contexts/AuthContext';
import { cn } from '@/lib/utils';
import { useTheme } from '@/contexts/ThemeContext';

const pageTitles = [
    { prefix: '/admin/organizations', title: 'Organizations', description: 'Create hospitals and manage organization administrators.' },
    { prefix: '/admin/staff', title: 'Staff', description: 'Manage doctors and nurses in your hospital.' },
    { prefix: '/dashboard/doctor', title: 'Doctor Dashboard', description: 'Operational overview for physicians and senior clinicians.' },
    { prefix: '/dashboard/nurse', title: 'Nurse Dashboard', description: 'Focused queue for bedside monitoring and alert response.' },
    { prefix: '/patients', title: 'Patients', description: 'Clinical profiles, dialysis baselines, and monitoring history.' },
    { prefix: '/monitor', title: 'Monitoring', description: 'Live vitals, risk progression, alerts, and explainability.' },
    { prefix: '/alerts', title: 'Alerts', description: 'Review and acknowledge hemodynamic risk alerts.' },
    { prefix: '/model-info', title: 'Model Transparency', description: 'Reference material for model behavior, data, and limitations.' },
];

const roleLabels = {
    super_admin: 'Super Admin',
    org_admin: 'Hospital Admin',
    doctor: 'Doctor',
    nurse: 'Nurse',
};

export default function PageShell({ children }) {
    const pathname = usePathname();
    const router = useRouter();
    const { user } = useAuth();
    const { session, patientSummary, hasActiveSession } = useMonitoring();
    const { preference, setPreference } = useTheme();
    const [sidebarOpen, setSidebarOpen] = useState(false);

    const pageMeta = useMemo(
        () => pageTitles.find((item) => pathname.startsWith(item.prefix)) || pageTitles[0],
        [pathname]
    );
    const isResumable = Boolean(
        hasActiveSession &&
        session?.id &&
        session?.can_resume &&
        (session?.current_step || 0) < (session?.total_steps || 30)
    );

    useEffect(() => {
        if (user?.must_change_password && pathname !== '/change-password') {
            router.push('/change-password');
        }
    }, [pathname, router, user]);

    return (
        <div className="min-h-screen bg-bg-primary">
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
                                    <p className="text-xs uppercase tracking-[0.2em] text-text-muted">
                                        {roleLabels[user?.role] || user?.role || 'Workspace'}
                                        {user?.org_name ? ` · ${user.org_name}` : ''}
                                    </p>
                                    <h1 className="text-2xl font-semibold text-text-primary">{pageMeta.title}</h1>
                                </div>
                            </div>
                            <div className="hidden items-center gap-3 rounded-2xl border border-border-subtle bg-surface px-3 py-2 sm:flex">
                                <div className="text-right">
                                    <p className="text-sm font-medium text-text-primary">{user?.name}</p>
                                    <p className="text-xs uppercase tracking-[0.14em] text-text-muted">{user?.org_name || roleLabels[user?.role] || 'Workspace'}</p>
                                </div>
                                <div className="flex items-center rounded-xl border border-border-subtle bg-bg-secondary p-1">
                                    <button
                                        onClick={() => setPreference('light')}
                                        className={cn(
                                            'rounded-lg p-1.5 text-text-muted transition-colors',
                                            preference === 'light' && 'bg-surface text-text-primary'
                                        )}
                                        title="Light theme"
                                        aria-label="Light theme"
                                    >
                                        <Sun className="h-4 w-4" />
                                    </button>
                                    <button
                                        onClick={() => setPreference('dark')}
                                        className={cn(
                                            'rounded-lg p-1.5 text-text-muted transition-colors',
                                            preference === 'dark' && 'bg-surface text-text-primary'
                                        )}
                                        title="Dark theme"
                                        aria-label="Dark theme"
                                    >
                                        <Moon className="h-4 w-4" />
                                    </button>
                                    <button
                                        onClick={() => setPreference('system')}
                                        className={cn(
                                            'rounded-lg p-1.5 text-text-muted transition-colors',
                                            preference === 'system' && 'bg-surface text-text-primary'
                                        )}
                                        title="System theme"
                                        aria-label="System theme"
                                    >
                                        <Laptop className="h-4 w-4" />
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
                            <p className="max-w-3xl text-sm text-text-secondary">{pageMeta.description}</p>
                            {isResumable && session && (
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
