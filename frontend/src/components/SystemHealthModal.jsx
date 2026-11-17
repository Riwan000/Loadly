import React, { useState, useEffect } from 'react';
import { 
  X, CheckCircle2, AlertTriangle, XCircle, Cpu, Database, 
  CloudSun, Key, Activity, RefreshCw, RotateCcw, ShieldCheck, 
  TrendingUp, BarChart2, History, ChevronDown, ChevronUp, AlertCircle
} from 'lucide-react';
import { api } from '../services/api';

export default function SystemHealthModal({ isOpen, onClose, health, metadata, currentState = 'All-India' }) {
  const [driftReport, setDriftReport] = useState(null);
  const [retrainHistory, setRetrainHistory] = useState([]);
  const [loadingDrift, setLoadingDrift] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const [rollingBack, setRollingBack] = useState(false);
  const [retrainResult, setRetrainResult] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [selectedScope, setSelectedScope] = useState(currentState || 'All-India');

  const isHealthy = health?.status === 'healthy';

  // Fetch Drift & Retrain History when modal opens
  useEffect(() => {
    if (isOpen) {
      loadDriftData(selectedScope);
      loadHistory();
    }
  }, [isOpen, selectedScope]);

  const loadDriftData = async (scope) => {
    setLoadingDrift(true);
    setActionError(null);
    try {
      const scopeParam = scope === 'All Grid' || !scope ? null : scope;
      const data = await api.getDriftStatus({ state: scopeParam, windowDays: 14 });
      setDriftReport(data);
    } catch (err) {
      console.error('Failed to load drift data:', err);
    } finally {
      setLoadingDrift(false);
    }
  };

  const loadHistory = async () => {
    try {
      const res = await api.getRetrainingHistory({ limit: 10 });
      setRetrainHistory(res.history || []);
    } catch (err) {
      console.error('Failed to load retraining history:', err);
    }
  };

  const handleRetrain = async () => {
    setRetraining(true);
    setActionError(null);
    setRetrainResult(null);
    try {
      const res = await api.triggerRetrain({ triggerReason: 'operator_manual_calibration', force: false });
      setRetrainResult(res);
      await loadDriftData(selectedScope);
      await loadHistory();
    } catch (err) {
      setActionError(err.message || 'Model retraining failed.');
    } finally {
      setRetraining(false);
    }
  };

  const handleRollback = async () => {
    if (!window.confirm('Are you sure you want to roll back the production model to the previous champion weights?')) {
      return;
    }
    setRollingBack(true);
    setActionError(null);
    try {
      await api.triggerRollback();
      setRetrainResult({
        decision: 'ROLLED_BACK',
        notes: 'Production model successfully restored to previous champion weights.'
      });
      await loadDriftData(selectedScope);
      await loadHistory();
    } catch (err) {
      setActionError(err.message || 'Rollback failed.');
    } finally {
      setRollingBack(false);
    }
  };

  if (!isOpen) return null;

  const components = [
    {
      name: `ML Model (${metadata?.model_name || 'Reduced_LR'})`,
      ready: health?.components?.model?.ready ?? true,
      details: health?.components?.model?.details || 'Loaded and ready for inference',
      icon: Cpu,
    },
    {
      name: 'RAG Vector Store',
      ready: health?.components?.rag_vector_store?.ready ?? true,
      details: health?.components?.rag_vector_store?.details || 'FAISS index initialized',
      icon: Database,
    },
    {
      name: 'Weather API (Open-Meteo)',
      ready: health?.components?.weather_api?.status === 'online',
      details: health?.components?.weather_api?.provider ? `Provider: ${health.components.weather_api.provider}` : 'Online',
      icon: CloudSun,
    },
    {
      name: 'LLM Engine / API Key',
      ready: health?.components?.openrouter_api_key?.configured ?? true,
      details: health?.components?.openrouter_api_key?.details || 'Ready (Local embeddings + LLM)',
      icon: Key,
    },
  ];

  // Drift Status styling helpers
  const getStatusBadge = (status) => {
    switch (status) {
      case 'HEALTHY':
        return {
          bg: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20',
          label: 'Healthy & Calibrated',
          icon: CheckCircle2
        };
      case 'WARNING':
        return {
          bg: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20',
          label: 'Drift Warning',
          icon: AlertTriangle
        };
      case 'CRITICAL':
        return {
          bg: 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20',
          label: 'Critical Decay',
          icon: AlertCircle
        };
      default:
        return {
          bg: 'bg-slate-500/10 text-slate-600 dark:text-slate-400 border-slate-500/20',
          label: 'Insufficient Data',
          icon: Activity
        };
    }
  };

  const statusBadge = getStatusBadge(driftReport?.status);
  const StatusIcon = statusBadge.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fadeIn">
      <div 
        className="relative w-full max-w-3xl bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark rounded-none shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-executive-borderLight dark:border-executive-borderDark bg-slate-50/50 dark:bg-slate-900/30">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-none ${isHealthy ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400' : 'bg-amber-500/10 text-amber-600 dark:text-amber-400'}`}>
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-executive-textLight dark:text-executive-textDark">
                System Health & MLOps Engine
              </h3>
              <p className="text-xs text-executive-mutedLight dark:text-executive-mutedDark">
                Real-time drift detection, champion-challenger validation gate, and model lifecycle
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-none text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 max-h-[75vh] overflow-y-auto">
          {/* Status Banner */}
          <div className={`p-4 rounded-none flex items-start gap-3 border ${
            isHealthy 
              ? 'bg-emerald-50 dark:bg-emerald-950/30 border-emerald-200 dark:border-emerald-900/50 text-emerald-800 dark:text-emerald-300' 
              : 'bg-amber-50 dark:bg-amber-950/30 border-amber-200 dark:border-amber-900/50 text-amber-800 dark:text-amber-300'
          }`}>
            {isHealthy ? <CheckCircle2 className="w-5 h-5 mt-0.5 flex-shrink-0" /> : <AlertTriangle className="w-5 h-5 mt-0.5 flex-shrink-0" />}
            <div className="flex-1">
              <div className="font-semibold text-sm">
                System Status: {isHealthy ? 'All Subsystems Operational' : 'Degraded Performance'}
              </div>
              <p className="text-xs mt-0.5 opacity-90">
                {isHealthy 
                  ? 'ML inference model, Open-Meteo weather connector, RAG engine, and Prediction Ledger are active.'
                  : health?.last_error || 'Some components are experiencing delays or fallbacks.'}
              </p>
            </div>
          </div>

          {/* ========================================================================= */}
          {/* MLOps Drift Detection & Adaptive Retraining Control Hub */}
          {/* ========================================================================= */}
          <div className="p-4 border border-executive-borderLight dark:border-executive-borderDark bg-slate-50/50 dark:bg-slate-900/40 rounded-none space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-200/60 dark:border-slate-800/80 pb-3">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-executive-navy dark:text-executive-cyan" />
                <h4 className="text-xs font-bold uppercase tracking-wider text-executive-textLight dark:text-executive-textDark">
                  Model Drift & Calibration Monitor
                </h4>
              </div>

              <div className="flex items-center gap-2">
                <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold border ${statusBadge.bg}`}>
                  <StatusIcon className="w-3.5 h-3.5" />
                  {statusBadge.label}
                </span>

                <select
                  value={selectedScope}
                  onChange={(e) => setSelectedScope(e.target.value)}
                  className="px-2 py-1 text-xs border border-executive-borderLight dark:border-executive-borderDark bg-white dark:bg-executive-cardDark text-executive-textLight dark:text-executive-textDark focus:outline-none"
                >
                  <option value="All-India">All-India National</option>
                  <option value="Maharashtra">Maharashtra</option>
                  <option value="Gujarat">Gujarat</option>
                  <option value="Tamil Nadu">Tamil Nadu</option>
                  <option value="Uttar Pradesh">Uttar Pradesh</option>
                  <option value="All Grid">All States Aggregated</option>
                </select>

                <button
                  onClick={() => loadDriftData(selectedScope)}
                  disabled={loadingDrift}
                  className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
                  title="Refresh Drift Metrics"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${loadingDrift ? 'animate-spin' : ''}`} />
                </button>
              </div>
            </div>

            {/* Metrics 4-Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <div className="p-3 bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark">
                <div className="text-[10px] text-executive-mutedLight dark:text-executive-mutedDark font-medium">Trailing MAPE</div>
                <div className="text-base font-bold text-executive-textLight dark:text-executive-textDark mt-0.5">
                  {driftReport?.metrics?.mape_pct !== null && driftReport?.metrics?.mape_pct !== undefined ? `${driftReport.metrics.mape_pct}%` : '—'}
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  Base: {driftReport?.metrics?.baseline_mape_pct ?? '1.68'}%
                </div>
              </div>

              <div className="p-3 bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark">
                <div className="text-[10px] text-executive-mutedLight dark:text-executive-mutedDark font-medium">Mean Bias Error</div>
                <div className={`text-base font-bold mt-0.5 ${
                  (driftReport?.metrics?.mean_bias_mu || 0) > 0 ? 'text-amber-600 dark:text-amber-400' : 'text-slate-700 dark:text-slate-200'
                }`}>
                  {driftReport?.metrics?.mean_bias_mu !== null && driftReport?.metrics?.mean_bias_mu !== undefined
                    ? `${driftReport.metrics.mean_bias_mu > 0 ? '+' : ''}${driftReport.metrics.mean_bias_mu} MU`
                    : '—'}
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  z-score: {driftReport?.metrics?.bias_z_score ?? '0.00'}
                </div>
              </div>

              <div className="p-3 bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark">
                <div className="text-[10px] text-executive-mutedLight dark:text-executive-mutedDark font-medium">90% CI Hit Rate</div>
                <div className="text-base font-bold text-emerald-600 dark:text-emerald-400 mt-0.5">
                  {driftReport?.metrics?.ci_coverage_pct !== null && driftReport?.metrics?.ci_coverage_pct !== undefined
                    ? `${driftReport.metrics.ci_coverage_pct}%`
                    : '—'}
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  Nominal: ≥ 85%
                </div>
              </div>

              <div className="p-3 bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark">
                <div className="text-[10px] text-executive-mutedLight dark:text-executive-mutedDark font-medium">Feature Drift (PSI)</div>
                <div className="text-base font-bold text-executive-textLight dark:text-executive-textDark mt-0.5">
                  {driftReport?.feature_drift?.temperature_psi ?? '0.000'}
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  Shift: &lt; 0.10 Nominal
                </div>
              </div>
            </div>

            {/* Diagnostics List */}
            {driftReport?.diagnostics && driftReport.diagnostics.length > 0 && (
              <div className="space-y-1.5 pt-1">
                {driftReport.diagnostics.map((diag, i) => (
                  <div key={i} className="text-xs flex items-start gap-2 text-executive-mutedLight dark:text-executive-mutedDark">
                    <span className="text-sky-500 font-bold">•</span>
                    <span>{diag}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Retrain Action & Rollback Bar */}
            <div className="pt-3 border-t border-slate-200/60 dark:border-slate-800/80 flex flex-wrap items-center justify-between gap-3">
              <div className="text-xs text-executive-mutedLight dark:text-executive-mutedDark">
                Reconciled Ledger Samples: <strong>{driftReport?.sample_count ?? 0}</strong>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleRetrain}
                  disabled={retraining || rollingBack}
                  className="px-3 py-1.5 text-xs font-semibold bg-emerald-600 hover:bg-emerald-700 text-white rounded-none flex items-center gap-1.5 transition-colors disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${retraining ? 'animate-spin' : ''}`} />
                  {retraining ? 'Evaluating Challenger...' : 'Calibrate & Retrain Champion'}
                </button>

                <button
                  onClick={handleRollback}
                  disabled={retraining || rollingBack}
                  className="px-3 py-1.5 text-xs font-semibold bg-slate-200 hover:bg-slate-300 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-none flex items-center gap-1.5 transition-colors disabled:opacity-50"
                  title="Roll back to previous champion model weights"
                >
                  <RotateCcw className={`w-3.5 h-3.5 ${rollingBack ? 'animate-spin' : ''}`} />
                  Rollback
                </button>
              </div>
            </div>

            {/* Retrain Result Card */}
            {retrainResult && (
              <div className={`p-3 text-xs border ${
                retrainResult.decision === 'PROMOTED'
                  ? 'bg-emerald-50 dark:bg-emerald-950/30 border-emerald-300 dark:border-emerald-800 text-emerald-800 dark:text-emerald-300'
                  : retrainResult.decision === 'ROLLED_BACK'
                  ? 'bg-sky-50 dark:bg-sky-950/30 border-sky-300 dark:border-sky-800 text-sky-800 dark:text-sky-300'
                  : 'bg-amber-50 dark:bg-amber-950/30 border-amber-300 dark:border-amber-800 text-amber-800 dark:text-amber-300'
              }`}>
                <div className="font-bold flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4" />
                  Validation Gate Result: {retrainResult.decision}
                </div>
                <div className="mt-1 opacity-90">{retrainResult.notes}</div>
                {retrainResult.challenger && (
                  <div className="mt-2 grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono bg-white/60 dark:bg-black/30 p-2">
                    <div>MAE: {retrainResult.challenger.mae} MU</div>
                    <div>RMSE: {retrainResult.challenger.rmse} MU</div>
                    <div>R²: {retrainResult.challenger.r2}</div>
                    <div>Coverage: {retrainResult.challenger.ci_coverage_pct}%</div>
                  </div>
                )}
              </div>
            )}

            {actionError && (
              <div className="p-2.5 text-xs bg-rose-50 dark:bg-rose-950/30 border border-rose-300 dark:border-rose-800 text-rose-700 dark:text-rose-300">
                {actionError}
              </div>
            )}

            {/* Collapsible Retraining History */}
            <div className="pt-1">
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="flex items-center gap-1.5 text-xs text-executive-navy dark:text-executive-cyan font-medium hover:underline"
              >
                <History className="w-3.5 h-3.5" />
                <span>{showHistory ? 'Hide MLOps Retraining Audit Log' : 'View MLOps Retraining Audit Log'}</span>
                {showHistory ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>

              {showHistory && (
                <div className="mt-3 border border-slate-200 dark:border-slate-800 overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-100 dark:bg-slate-800/60 text-executive-mutedLight dark:text-executive-mutedDark">
                      <tr>
                        <th className="p-2">Timestamp</th>
                        <th className="p-2">Trigger</th>
                        <th className="p-2">Decision</th>
                        <th className="p-2">Challenger RMSE</th>
                        <th className="p-2">Notes</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                      {retrainHistory.length === 0 ? (
                        <tr>
                          <td colSpan="5" className="p-3 text-center text-slate-400">No retraining events logged yet.</td>
                        </tr>
                      ) : (
                        retrainHistory.map((item) => (
                          <tr key={item.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                            <td className="p-2 text-slate-500 font-mono text-[11px] whitespace-nowrap">{item.retrained_at}</td>
                            <td className="p-2 font-medium">{item.trigger_reason}</td>
                            <td className="p-2">
                              <span className={`px-1.5 py-0.5 text-[10px] font-semibold ${
                                item.decision === 'PROMOTED' ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400' :
                                item.decision === 'ROLLED_BACK' ? 'bg-sky-500/10 text-sky-600 dark:text-sky-400' :
                                'bg-amber-500/10 text-amber-600 dark:text-amber-400'
                              }`}>
                                {item.decision}
                              </span>
                            </td>
                            <td className="p-2 font-mono">{item.challenger_rmse ? `${item.challenger_rmse.toFixed(3)} MU` : '—'}</td>
                            <td className="p-2 text-slate-500 text-[11px] max-w-xs truncate">{item.notes}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>

          {/* Subsystems Grid */}
          <div>
            <h4 className="text-xs font-bold uppercase tracking-wider text-executive-mutedLight dark:text-executive-mutedDark mb-3">
              Platform Subsystems
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {components.map((comp, idx) => {
                const Icon = comp.icon;
                return (
                  <div 
                    key={idx}
                    className="p-3.5 rounded-none border border-executive-borderLight dark:border-executive-borderDark bg-slate-50/50 dark:bg-slate-900/40 flex items-center justify-between"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-none bg-executive-navy/5 dark:bg-executive-cyan/10 text-executive-navy dark:text-executive-cyan">
                        <Icon className="w-4 h-4" />
                      </div>
                      <div>
                        <div className="text-xs font-semibold text-executive-textLight dark:text-executive-textDark">
                          {comp.name}
                        </div>
                        <div className="text-[11px] text-executive-mutedLight dark:text-executive-mutedDark truncate max-w-[180px]">
                          {comp.details}
                        </div>
                      </div>
                    </div>
                    <div>
                      {comp.ready ? (
                        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-none">
                          <CheckCircle2 className="w-3 h-3" /> Ready
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-600 dark:text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded-none">
                          <AlertTriangle className="w-3 h-3" /> Offline
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Model Architecture & Baseline Specs */}
          {metadata && (
            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-executive-mutedLight dark:text-executive-mutedDark mb-3">
                Production Model Architecture & Baseline Benchmarks
              </h4>
              <div className="p-4 rounded-none border border-executive-borderLight dark:border-executive-borderDark bg-slate-50/50 dark:bg-slate-900/40 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-200/60 dark:border-slate-800/80 pb-3">
                  <span className="text-xs text-executive-mutedLight dark:text-executive-mutedDark">Production Architecture</span>
                  <span className="text-xs font-semibold text-executive-navy dark:text-executive-cyan">
                    {metadata.model_name || 'Reduced_LR (Linear Regression, 15 Features)'}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="p-2.5 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark">
                    <div className="text-[10px] text-executive-mutedLight dark:text-executive-mutedDark font-medium">Baseline MAE</div>
                    <div className="text-base font-bold text-executive-textLight dark:text-executive-textDark">
                      {metadata.mae ? metadata.mae.toFixed(2) : '3.21'} <span className="text-[10px] font-normal text-slate-400">MU</span>
                    </div>
                  </div>
                  <div className="p-2.5 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark">
                    <div className="text-[10px] text-executive-mutedLight dark:text-executive-mutedDark font-medium">Baseline RMSE</div>
                    <div className="text-base font-bold text-executive-textLight dark:text-executive-textDark">
                      {metadata.rmse ? metadata.rmse.toFixed(2) : '5.36'} <span className="text-[10px] font-normal text-slate-400">MU</span>
                    </div>
                  </div>
                  <div className="p-2.5 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark">
                    <div className="text-[10px] text-executive-mutedLight dark:text-executive-mutedDark font-medium">Baseline R²</div>
                    <div className="text-base font-bold text-emerald-600 dark:text-emerald-400">
                      {metadata.r2 ? metadata.r2.toFixed(4) : '0.9989'}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs text-executive-mutedLight dark:text-executive-mutedDark pt-1">
                  <div>
                    Nominal Confidence Interval: <strong className="text-executive-textLight dark:text-executive-textDark">{((metadata.confidence_level || 0.9) * 100).toFixed(0)}%</strong>
                  </div>
                  <div>
                    Active Features: <strong className="text-executive-textLight dark:text-executive-textDark">{metadata.n_features || 15} Input Features</strong>
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-200/60 dark:border-slate-800/80 text-[11px] text-executive-mutedLight dark:text-executive-mutedDark leading-relaxed">
                  <span className="font-semibold text-sky-600 dark:text-sky-400">Deployment Policy:</span> Zero-downtime atomic swap with champion-challenger validation guardrail. Automatic fallback to <code className="text-[10px] bg-slate-200 dark:bg-slate-800 px-1 py-0.5">best_model_prev.pkl</code> upon operator rollback.
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-executive-borderLight dark:border-executive-borderDark bg-slate-50 dark:bg-slate-900/60 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-semibold rounded-none bg-executive-navy dark:bg-executive-cyan text-white dark:text-slate-950 hover:opacity-90 transition-opacity"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
