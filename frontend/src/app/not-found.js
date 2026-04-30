'use client';

import { useRouter } from 'next/navigation';
import { ArrowLeft, MonitorDot, Users } from 'lucide-react';

export default function NotFound() {
    const router = useRouter();

    return (
        <div className="flex min-h-screen items-center justify-center bg-bg-primary px-4">
            <div className="card max-w-2xl p-8 text-center">
                <p className="text-xs uppercase tracking-[0.2em] text-text-muted">404</p>
                <h1 className="mt-3 text-3xl font-semibold text-text-primary">This route is not part of the demo workflow</h1>
                <p className="mt-3 text-sm text-text-secondary">
                    Use the patient registry or the monitor workspace to return to a supported clinical flow.
                </p>
                <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:justify-center">
                    <button
                        onClick={() => router.push('/patients')}
                        className="inline-flex items-center justify-center gap-2 rounded-2xl bg-accent px-5 py-3 text-sm font-semibold text-bg-primary"
                    >
                        <Users className="h-4 w-4" />
                        Open patients
                    </button>
                    <button
                        onClick={() => router.push('/monitor')}
                        className="inline-flex items-center justify-center gap-2 rounded-2xl border border-border-subtle px-5 py-3 text-sm font-medium text-text-primary hover:bg-surface-hover"
                    >
                        <MonitorDot className="h-4 w-4" />
                        Open monitor
                    </button>
                    <button
                        onClick={() => router.back()}
                        className="inline-flex items-center justify-center gap-2 rounded-2xl border border-border-subtle px-5 py-3 text-sm font-medium text-text-primary hover:bg-surface-hover"
                    >
                        <ArrowLeft className="h-4 w-4" />
                        Go back
                    </button>
                </div>
            </div>
        </div>
    );
}
