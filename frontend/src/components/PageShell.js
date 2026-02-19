'use client';
import Sidebar from './Sidebar';

/**
 * PageShell — wraps authenticated pages with the sidebar layout.
 * 
 * Usage:
 *   <PageShell>
 *     <h1>Page Title</h1>
 *     <div>Content</div>
 *   </PageShell>
 */
export default function PageShell({ children }) {
    return (
        <div className="flex min-h-screen">
            <Sidebar />
            <main className="ml-60 flex-1 overflow-y-auto px-8 py-6">
                {children}
            </main>
        </div>
    );
}
