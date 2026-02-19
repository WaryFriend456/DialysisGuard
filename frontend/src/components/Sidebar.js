'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { cn } from '@/lib/utils';
import {
    LayoutDashboard,
    Building2,
    MonitorDot,
    Users,
    Bell,
    Brain,
    LogOut,
    Activity,
    ChevronRight,
} from 'lucide-react';

const navItems = [
    { href: '/dashboard/doctor', label: 'Dashboard', icon: LayoutDashboard, roles: ['doctor'] },
    { href: '/dashboard/caregiver', label: 'Dashboard', icon: Building2, roles: ['caregiver'] },
    { href: '/dashboard/command', label: 'Command Center', icon: MonitorDot, roles: ['doctor'] },
    { href: '/patients', label: 'Patients', icon: Users, roles: ['doctor', 'caregiver'] },
    { href: '/alerts', label: 'Alerts', icon: Bell, roles: ['doctor', 'caregiver'] },
    { href: '/model-info', label: 'Model Info', icon: Brain, roles: ['doctor'] },
];

export default function Sidebar() {
    const pathname = usePathname();
    const { user, logout } = useAuth();

    if (!user) return null;

    const filtered = navItems.filter(item => item.roles.includes(user.role));

    return (
        <aside className="fixed top-0 left-0 z-40 flex h-screen w-60 flex-col border-r border-border-subtle bg-bg-secondary">
            {/* Brand */}
            <div className="px-5 pt-6 pb-4">
                <div className="flex items-center gap-2.5">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/15">
                        <Activity className="h-4.5 w-4.5 text-accent" />
                    </div>
                    <div>
                        <h1 className="text-base font-bold tracking-tight text-text-primary">
                            DialysisGuard
                        </h1>
                        <p className="text-[10px] font-medium tracking-wider uppercase text-text-muted">
                            AI Monitoring
                        </p>
                    </div>
                </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 space-y-0.5 overflow-y-auto px-3 py-2">
                <p className="mb-2 px-2 text-[10px] font-semibold tracking-widest uppercase text-text-muted">
                    Navigation
                </p>
                {filtered.map(item => {
                    const Icon = item.icon;
                    const active = pathname === item.href;
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={cn(
                                'group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all',
                                'hover:bg-surface-hover hover:text-text-primary',
                                active
                                    ? 'bg-accent/10 text-accent'
                                    : 'text-text-secondary'
                            )}
                        >
                            <Icon className={cn(
                                'h-4 w-4 shrink-0 transition-colors',
                                active ? 'text-accent' : 'text-text-muted group-hover:text-text-secondary'
                            )} />
                            <span className="flex-1">{item.label}</span>
                            {active && (
                                <ChevronRight className="h-3.5 w-3.5 text-accent/60" />
                            )}
                        </Link>
                    );
                })}
            </nav>

            {/* User footer */}
            <div className="border-t border-border-subtle px-4 py-4">
                <div className="mb-3 flex items-center gap-3">
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-surface text-xs font-bold text-accent uppercase">
                        {user.name?.charAt(0) || '?'}
                    </div>
                    <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium text-text-primary">
                            {user.name}
                        </p>
                        <p className="text-xs text-text-muted capitalize">
                            {user.role}
                        </p>
                    </div>
                </div>
                <button
                    onClick={logout}
                    className="flex w-full items-center justify-center gap-2 rounded-lg border border-border-subtle bg-transparent px-3 py-1.5 text-xs font-medium text-text-secondary transition-all hover:border-border hover:bg-surface-hover hover:text-text-primary cursor-pointer"
                >
                    <LogOut className="h-3.5 w-3.5" />
                    Sign Out
                </button>
            </div>
        </aside>
    );
}
