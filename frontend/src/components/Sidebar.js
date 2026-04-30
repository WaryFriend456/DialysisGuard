'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, Bell, Brain, Building2, LayoutDashboard, LogOut, MonitorDot, UserCog, Users, X } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { cn } from '@/lib/utils';

const navItems = [
    {
        label: 'Organizations',
        icon: Building2,
        roles: ['super_admin'],
        href: '/admin/organizations',
        match: ['/admin/organizations'],
    },
    {
        label: 'Staff',
        icon: UserCog,
        roles: ['org_admin'],
        href: '/admin/staff',
        match: ['/admin/staff'],
    },
    {
        label: 'Dashboard',
        icon: LayoutDashboard,
        roles: ['doctor'],
        href: '/dashboard/doctor',
        match: ['/dashboard/doctor'],
    },
    {
        label: 'Dashboard',
        icon: LayoutDashboard,
        roles: ['nurse'],
        href: '/dashboard/nurse',
        match: ['/dashboard/nurse'],
    },
    {
        label: 'Patients',
        icon: Users,
        roles: ['org_admin', 'doctor', 'nurse'],
        href: '/patients',
        match: ['/patients'],
    },
    {
        label: 'Monitor',
        icon: MonitorDot,
        roles: ['org_admin', 'doctor', 'nurse'],
        href: '/monitor',
        match: ['/monitor'],
    },
    {
        label: 'Alerts',
        icon: Bell,
        roles: ['org_admin', 'doctor', 'nurse'],
        href: '/alerts',
        match: ['/alerts'],
    },
    {
        label: 'Model Info',
        icon: Brain,
        roles: ['org_admin', 'doctor'],
        href: '/model-info',
        match: ['/model-info'],
    },
];

const roleLabels = {
    super_admin: 'Super Admin',
    org_admin: 'Hospital Admin',
    doctor: 'Doctor',
    nurse: 'Nurse',
};

export default function Sidebar({ open, onClose }) {
    const pathname = usePathname();
    const { user, logout } = useAuth();

    if (!user) return null;

    const filtered = navItems.filter((item) => item.roles.includes(user.role));

    return (
        <>
            <div
                className={cn(
                    'fixed inset-0 z-40 bg-black/50 transition-opacity lg:hidden',
                    open ? 'opacity-100' : 'pointer-events-none opacity-0'
                )}
                onClick={onClose}
            />
            <aside
                className={cn(
                    'fixed inset-y-0 left-0 z-50 flex w-72 flex-col border-r border-border-subtle bg-bg-secondary/95 backdrop-blur transition-transform lg:translate-x-0',
                    open ? 'translate-x-0' : '-translate-x-full'
                )}
            >
                <div className="flex items-center justify-between border-b border-border-subtle px-5 py-5">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-accent/15">
                            <Activity className="h-5 w-5 text-accent" />
                        </div>
                        <div>
                            <p className="text-lg font-semibold text-text-primary">DialysisGuard</p>
                            <p className="text-xs uppercase tracking-[0.2em] text-text-muted">Clinical Workspace</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="rounded-xl p-2 text-text-muted hover:bg-surface-hover hover:text-text-primary lg:hidden"
                    >
                        <X className="h-4 w-4" />
                    </button>
                </div>

                <nav className="flex-1 space-y-1 overflow-y-auto px-4 py-4">
                    {filtered.map((item) => {
                        const Icon = item.icon;
                        const active = item.match.some((prefix) => pathname.startsWith(prefix));
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                onClick={onClose}
                                className={cn(
                                    'flex items-center gap-3 rounded-2xl px-4 py-3 text-sm font-medium transition-all',
                                    active
                                        ? 'bg-accent text-bg-primary shadow-[0_8px_24px_rgba(0,0,0,0.18)]'
                                        : 'text-text-secondary hover:bg-surface-hover hover:text-text-primary'
                                )}
                            >
                                <Icon className="h-4 w-4" />
                                <span>{item.label}</span>
                            </Link>
                        );
                    })}
                </nav>

                <div className="border-t border-border-subtle px-4 py-4">
                    <div className="mb-4 rounded-2xl border border-border-subtle bg-surface px-4 py-3">
                        <p className="text-sm font-semibold text-text-primary">{user.name}</p>
                        <p className="text-xs uppercase tracking-wide text-text-muted">{roleLabels[user.role] || user.role}</p>
                        {user.org_name && <p className="mt-1 text-xs text-text-muted">{user.org_name}</p>}
                    </div>
                    <button
                        onClick={logout}
                        className="flex w-full items-center justify-center gap-2 rounded-2xl border border-border-subtle px-4 py-3 text-sm font-medium text-text-secondary transition-all hover:bg-surface-hover hover:text-text-primary"
                    >
                        <LogOut className="h-4 w-4" />
                        Sign Out
                    </button>
                </div>
            </aside>
        </>
    );
}
