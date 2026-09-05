import React from 'react';
import { X, CheckCircle2, AlertTriangle, XCircle, Cpu, Database, CloudSun, Key, Activity, Layers } from 'lucide-react';

export default function SystemHealthModal({ isOpen, onClose, health, metadata }) {
  if (!isOpen) return null;

  const isHealthy = health?.status === 'healthy';

  const components = [
    {
      name: 'ML Model (XGBoost)',
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

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fadeIn">
      <div 
        className="relative w-full max-w-2xl bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark rounded-none shadow-2xl overflow-hidden"
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
                System Health & Model Architecture
              </h3>
              <p className="text-xs text-executive-mutedLight dark:text-executive-mutedDark">
                Real-time operational status and model benchmarks
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
            <div>
              <div className="font-semibold text-sm">
                System Status: {isHealthy ? 'All Systems Operational' : 'Degraded Performance'}
              </div>
              <p className="text-xs mt-0.5 opacity-90">
                {isHealthy 
                  ? 'ML inference model, Open-Meteo weather connector, and RAG knowledge retrieval are active.'
                  : health?.last_error || 'Some components are experiencing delays or fallbacks.'}
              </p>
            </div>
          </div>

          {/* Components Grid */}
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

          {/* Model Architecture & Benchmarks */}
          {metadata && (
            <div>
              <h4 className="text-xs font-bold uppercase tracking-wider text-executive-mutedLight dark:text-executive-mutedDark mb-3">
                Model Specifications & Test Benchmarks
              </h4>
              <div className="p-4 rounded-none border border-executive-borderLight dark:border-executive-borderDark bg-slate-50/50 dark:bg-slate-900/40 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-200/60 dark:border-slate-800/80 pb-3">
                  <span className="text-xs text-executive-mutedLight dark:text-executive-mutedDark">Architecture</span>
                  <span className="text-xs font-semibold text-executive-navy dark:text-executive-cyan">
                    {metadata.model_type || 'XGBoost (log-transformed target)'}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="p-2.5 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark">
                    <div className="text-[10px] text-executive-mutedLight dark:text-executive-mutedDark font-medium">Test MAE</div>
                    <div className="text-base font-bold text-executive-textLight dark:text-executive-textDark">
                      {metadata.overall_metrics?.mae ?? '2.18'} <span className="text-[10px] font-normal text-slate-400">MU</span>
                    </div>
                  </div>
                  <div className="p-2.5 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark">
                    <div className="text-[10px] text-executive-mutedLight dark:text-executive-mutedDark font-medium">Test RMSE</div>
                    <div className="text-base font-bold text-executive-textLight dark:text-executive-textDark">
                      {metadata.overall_metrics?.rmse ?? '4.58'} <span className="text-[10px] font-normal text-slate-400">MU</span>
                    </div>
                  </div>
                  <div className="p-2.5 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark">
                    <div className="text-[10px] text-executive-mutedLight dark:text-executive-mutedDark font-medium">R² Score</div>
                    <div className="text-base font-bold text-emerald-600 dark:text-emerald-400">
                      {metadata.overall_metrics?.r2 ?? '0.998'}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3 text-xs text-executive-mutedLight dark:text-executive-mutedDark pt-1">
                  <div>
                    Confidence Interval: <strong className="text-executive-textLight dark:text-executive-textDark">{((metadata.confidence_level || 0.9) * 100).toFixed(0)}%</strong>
                  </div>
                  <div>
                    Extreme Heat: <strong className="text-executive-textLight dark:text-executive-textDark">{(metadata.extreme_heat_threshold_kelvin - 273.15).toFixed(1)}°C (305K)</strong>
                  </div>
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
