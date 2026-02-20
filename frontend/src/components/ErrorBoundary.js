'use client';
import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

export default class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error("Uncaught error:", error, errorInfo);
    }

    handleRetry = () => {
        this.setState({ hasError: false, error: null });
        window.location.reload();
    };

    render() {
        if (this.state.hasError) {
            return (
                <div className="flex min-h-[400px] w-full flex-col items-center justify-center rounded-xl bg-bg-secondary/50 p-8 text-center backdrop-blur-sm border border-red-500/20">
                    <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-risk-critical/10 shadow-[0_0_20px_rgba(255,0,0,0.2)]">
                        <AlertCircle className="h-8 w-8 text-risk-critical animate-pulse" />
                    </div>
                    <h2 className="mb-2 text-xl font-bold tracking-tight text-white">System Malfunction</h2>
                    <p className="mb-6 max-w-sm text-sm text-text-muted font-mono bg-black/30 p-2 rounded border border-white/5">
                        {this.state.error?.message || 'Unknown critical error'}
                    </p>
                    <button
                        onClick={this.handleRetry}
                        className="flex items-center gap-2 rounded-lg bg-risk-critical/20 px-6 py-2.5 text-sm font-bold text-risk-critical hover:bg-risk-critical/30 transition-all border border-risk-critical/50 shadow-lg hover:shadow-red-900/40 cursor-pointer"
                    >
                        <RefreshCw className="h-4 w-4" /> REBOOT_SYSTEM
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}
