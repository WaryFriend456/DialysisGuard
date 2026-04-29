'use client';

import Link from 'next/link';
import { ShieldCheck } from 'lucide-react';

export default function RegisterDisabledPage() {
    return (
        <div className="flex min-h-screen items-center justify-center bg-bg-primary px-4">
            <div className="card w-full max-w-md px-8 py-9 text-center">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-accent/15">
                    <ShieldCheck className="h-6 w-6 text-accent" />
                </div>
                <h1 className="mt-5 text-2xl font-bold text-text-primary">Account creation is managed by admins</h1>
                <p className="mt-3 text-sm text-text-secondary">
                    Ask your hospital administrator to create your DialysisGuard account.
                </p>
                <Link
                    href="/login"
                    className="mt-6 inline-flex rounded-2xl bg-accent px-5 py-3 text-sm font-semibold text-bg-primary"
                >
                    Back to sign in
                </Link>
            </div>
        </div>
    );
}
