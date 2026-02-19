'use client';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import PageShell from '@/components/PageShell';
import { useAuth } from '@/contexts/AuthContext';
import { explain } from '@/lib/api';
import {
    Cpu, BarChart3, Database, Eye, Loader2,
    Microscope, Shuffle, ArrowLeftRight, BookOpen, FileText,
    Activity, Lightbulb,
} from 'lucide-react';

const XAI_CAPS = [
    { icon: Microscope, name: 'SHAP Values', desc: 'Feature attribution via DeepExplainer' },
    { icon: Eye, name: 'Attention Weights', desc: 'Temporal focus visualization' },
    { icon: Shuffle, name: 'What-If Analysis', desc: 'Parameter modification scenarios' },
    { icon: ArrowLeftRight, name: 'Counterfactuals', desc: 'Minimal changes for target risk' },
    { icon: BarChart3, name: 'Sensitivity', desc: 'Feature sensitivity ±10%' },
    { icon: BookOpen, name: 'NL Explanations', desc: 'Human-readable risk reasoning' },
    { icon: FileText, name: 'Model Card', desc: 'Full transparency documentation' },
];

export default function ModelInfoPage() {
    const { user, loading } = useAuth();
    const router = useRouter();
    const [modelCard, setModelCard] = useState(null);

    useEffect(() => { if (!loading && !user) router.push('/login'); }, [user, loading, router]);
    useEffect(() => { if (user) explain.modelCard().then(setModelCard).catch(() => { }); }, [user]);

    if (loading || !user) return <div className="flex min-h-screen items-center justify-center bg-bg-primary"><Loader2 className="h-8 w-8 animate-spin text-accent" /></div>;

    return (
        <PageShell>
            <div className="mb-8">
                <h1 className="text-2xl font-bold text-text-primary">Model Transparency</h1>
                <p className="mt-1 text-sm text-text-muted">Full transparency into the AI model powering DialysisGuard</p>
            </div>

            {modelCard ? (
                <div className="grid grid-cols-2 gap-5">
                    {/* Architecture */}
                    <div className="card p-5">
                        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-text-primary">
                            <Cpu className="h-4 w-4 text-accent" /> Architecture
                        </h3>
                        <div className="space-y-2 text-sm text-text-secondary leading-relaxed">
                            {[
                                ['Model', modelCard.model_name],
                                ['Type', modelCard.model_type],
                                ['Framework', modelCard.framework],
                                ['Features', modelCard.n_features],
                                ['Time Steps', modelCard.n_timesteps],
                                ['Parameters', modelCard.total_params?.toLocaleString()],
                            ].map(([k, v]) => (
                                <p key={k}><span className="font-medium text-text-primary">{k}:</span> {v}</p>
                            ))}
                        </div>
                    </div>

                    {/* Performance */}
                    <div className="card p-5">
                        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-text-primary">
                            <BarChart3 className="h-4 w-4 text-risk-low" /> Performance
                        </h3>
                        {modelCard.performance && (
                            <div className="space-y-3">
                                {[
                                    { label: 'Accuracy', value: modelCard.performance.accuracy, highlight: true },
                                    { label: 'AUC-ROC', value: modelCard.performance.auc, highlight: true },
                                    { label: 'Precision', value: modelCard.performance.precision },
                                    { label: 'Recall', value: modelCard.performance.recall },
                                    { label: 'F1 Score', value: modelCard.performance.f1 },
                                ].map(({ label, value, highlight }) => (
                                    <div key={label}>
                                        <div className="mb-1 flex justify-between text-sm">
                                            <span className="text-text-secondary">{label}</span>
                                            <span className={highlight ? 'font-bold text-risk-low' : 'text-text-primary'}>{(value * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="h-1.5 overflow-hidden rounded-full bg-surface">
                                            <div className={`h-full rounded-full ${highlight ? 'bg-risk-low' : 'bg-accent/40'}`} style={{ width: `${value * 100}%` }} />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Training Data */}
                    <div className="card p-5">
                        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-text-primary">
                            <Database className="h-4 w-4 text-chart-5" /> Training Data
                        </h3>
                        <div className="space-y-2 text-sm text-text-secondary leading-relaxed">
                            {[
                                ['Dataset', modelCard.training_data?.dataset],
                                ['Patients', modelCard.training_data?.n_patients?.toLocaleString()],
                                ['Time Steps', modelCard.training_data?.n_timesteps_per_patient],
                                ['Positive Rate', modelCard.training_data?.positive_rate],
                                ['Split Strategy', modelCard.training_data?.split_strategy],
                            ].map(([k, v]) => (
                                <p key={k}><span className="font-medium text-text-primary">{k}:</span> {v}</p>
                            ))}
                        </div>
                    </div>

                    {/* XAI Capabilities */}
                    <div className="card p-5">
                        <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-text-primary">
                            <Lightbulb className="h-4 w-4 text-risk-moderate" /> XAI Capabilities
                        </h3>
                        <div className="space-y-2">
                            {XAI_CAPS.map(({ icon: Icon, name, desc }, i) => (
                                <div key={i} className="flex items-center gap-3 rounded-lg border border-accent/10 bg-accent/5 px-3 py-2.5">
                                    <Icon className="h-4 w-4 shrink-0 text-accent" />
                                    <div>
                                        <p className="text-sm font-medium text-text-primary">{name}</p>
                                        <p className="text-xs text-text-muted">{desc}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Feature List */}
                    {modelCard.features && (
                        <div className="card col-span-2 p-5">
                            <h3 className="mb-4 text-sm font-semibold text-text-primary">Feature List ({modelCard.features.length})</h3>
                            <div className="flex flex-wrap gap-2">
                                {modelCard.features.map((f, i) => (
                                    <span key={i} className="rounded-full border border-border-subtle bg-bg-secondary px-3 py-1 text-xs text-text-secondary">{f}</span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            ) : (
                <div className="card flex flex-col items-center py-16 text-center">
                    <Loader2 className="mb-3 h-8 w-8 animate-spin text-accent" />
                    <p className="text-sm text-text-muted">Loading model information…</p>
                </div>
            )}
        </PageShell>
    );
}
