import React, { useState, useEffect } from 'react';
import { 
  X, 
  CheckCircle2, 
  AlertTriangle, 
  Clock, 
  TrendingUp, 
  Database, 
  Filter,
  RefreshCw,
  Percent,
  Check,
  AlertCircle
} from 'lucide-react';
import { api } from '../services/api';

export default function LedgerModal({ isOpen, onClose, selectedState }) {
  const [metrics, setMetrics] = useState(null);
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterState, setFilterState] = useState(selectedState || '');
  const [reconciledOnly, setReconciledOnly] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncToast, setSyncToast] = useState(null);

  const fetchLedgerData = async () => {
    setLoading(true);
    try {
      const [metricsData, recordsData] = await Promise.all([
        api.getLedgerMetrics({ state: filterState || undefined, daysWindow: 30 }),
        api.getLedgerRecords({ state: filterState || undefined, limit: 100, reconciledOnly })
      ]);
      setMetrics(metricsData);
      setRecords(recordsData.records || []);
    } catch (err) {
      console.error('Failed to fetch prediction ledger:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleSyncActuals = async () => {
    setIsSyncing(true);
    setSyncToast(null);
    try {
      const res = await api.triggerIngestion({ catchupDays: 3 });
      setSyncToast(`Sync complete! Reconciled ${res.reconciled_forecasts} forecasts across ${res.dates_processed?.length || 1} day(s).`);
      await fetchLedgerData();
    } catch (err) {
      setSyncToast(`Sync failed: ${err.message || 'Unknown error'}`);
    } finally {
      setIsSyncing(false);
      setTimeout(() => setSyncToast(null), 5000);
    }
  };

  useEffect(() => {
    if (isOpen) {
      setFilterState(selectedState || '');
      fetchLedgerData();
    }
  }, [isOpen, selectedState, reconciledOnly]);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchLedgerData();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-fadeIn">
      <div className="bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark w-full max-w-5xl max-h-[90vh] flex flex-col shadow-2xl rounded-none text-executive-textLight dark:text-executive-textDark">
        
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-executive-borderLight dark:border-executive-borderDark bg-slate-50 dark:bg-slate-900/50">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-executive-navy/10 dark:bg-executive-cyan/20 text-executive-navy dark:text-executive-cyan">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-extrabold text-base sm:text-lg flex items-center gap-2">
                Prediction Ledger & Verification Engine
                <span className="text-xs px-2 py-0.5 font-bold uppercase tracking-wider bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800">
                  Closed-Loop Audit
                </span>
              </h3>
              <p className="text-xs text-executive-mutedLight dark:text-executive-mutedDark">
                Immutable record of forecasts reconciled against ground-truth grid consumption
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              title="Refresh ledger records"
              className="p-1.5 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400 transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-500 dark:text-slate-400 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Verification KPI Ribbon */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 p-6 border-b border-executive-borderLight dark:border-executive-borderDark bg-white dark:bg-executive-cardDark">
          
          <div className="p-3 bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-800">
            <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Logged Forecasts</div>
            <div className="text-xl font-black mt-1 text-slate-900 dark:text-slate-100">
              {metrics?.total_predictions_logged ?? '—'}
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">Total slices recorded</div>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-800">
            <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Verified Actuals</div>
            <div className="text-xl font-black mt-1 text-emerald-600 dark:text-emerald-400">
              {metrics?.reconciled_count ?? '—'}
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">Ground-truth matched</div>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-800">
            <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Realized MAPE</div>
            <div className="text-xl font-black mt-1 text-executive-navy dark:text-executive-cyan">
              {metrics?.mape_pct != null ? `${metrics.mape_pct}%` : 'N/A'}
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">Mean Abs % Error</div>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-800">
            <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">90% CI Hit Rate</div>
            <div className="text-xl font-black mt-1 text-indigo-600 dark:text-indigo-400">
              {metrics?.ci_coverage_pct != null ? `${metrics.ci_coverage_pct}%` : 'N/A'}
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">Within uncertainty band</div>
          </div>

          <div className="p-3 bg-slate-50 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-800">
            <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">Mean Bias</div>
            <div className="text-xl font-black mt-1 text-amber-600 dark:text-amber-400">
              {metrics?.mean_bias_mu != null ? `${metrics.mean_bias_mu > 0 ? '+' : ''}${metrics.mean_bias_mu} MU` : 'N/A'}
            </div>
            <div className="text-[10px] text-slate-400 mt-0.5">Systematic error drift</div>
          </div>

        </div>

        {/* Filter Bar */}
        <div className="px-6 py-3 bg-slate-50/80 dark:bg-slate-900/30 border-b border-executive-borderLight dark:border-executive-borderDark flex flex-wrap items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-slate-500 dark:text-slate-400">Filter Scope:</span>
            <input
              type="text"
              placeholder="State name..."
              value={filterState}
              onChange={(e) => setFilterState(e.target.value)}
              className="px-2.5 py-1 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 text-xs w-36 focus:outline-none"
            />
            <label className="flex items-center gap-1.5 cursor-pointer text-slate-600 dark:text-slate-300">
              <input
                type="checkbox"
                checked={reconciledOnly}
                onChange={(e) => setReconciledOnly(e.target.checked)}
                className="rounded-none text-executive-navy focus:ring-0 cursor-pointer"
              />
              <span>Reconciled Only</span>
            </label>
            <button
              onClick={handleSyncActuals}
              disabled={isSyncing}
              className="flex items-center gap-1.5 px-3 py-1 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold transition-colors disabled:opacity-50 text-xs"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isSyncing ? 'animate-spin' : ''}`} />
              <span>{isSyncing ? 'Syncing...' : 'Sync & Verify Actuals'}</span>
            </button>
          </div>
          <div className="flex items-center gap-3">
            {syncToast && (
              <span className="text-[11px] font-semibold text-emerald-600 dark:text-emerald-400 animate-fadeIn">
                {syncToast}
              </span>
            )}
            <span className="text-[11px] text-slate-400">
              Showing latest {records.length} records
            </span>
          </div>
        </div>

        {/* Records Table */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading ? (
            <div className="flex items-center justify-center py-16 text-slate-400 gap-2">
              <RefreshCw className="w-5 h-5 animate-spin" />
              <span>Loading ledger history...</span>
            </div>
          ) : records.length === 0 ? (
            <div className="text-center py-16 text-slate-400">
              <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="font-medium text-sm">No ledger entries found</p>
              <p className="text-xs mt-1">Generate a demand forecast to automatically record predictions.</p>
            </div>
          ) : (
            <div className="overflow-x-auto border border-slate-200 dark:border-slate-800">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-100 dark:bg-slate-900/80 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 font-semibold">
                    <th className="py-2.5 px-3">Target Date</th>
                    <th className="py-2.5 px-3">State</th>
                    <th className="py-2.5 px-3 text-right">Predicted</th>
                    <th className="py-2.5 px-3 text-center">90% CI Range</th>
                    <th className="py-2.5 px-3 text-right">Actual</th>
                    <th className="py-2.5 px-3 text-right">Error</th>
                    <th className="py-2.5 px-3 text-right">MAPE %</th>
                    <th className="py-2.5 px-3 text-center">Audit Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800/60 font-mono">
                  {records.map((r) => {
                    const isReconciled = r.actual_demand_mu != null;
                    return (
                      <tr 
                        key={r.id}
                        className="hover:bg-slate-50 dark:hover:bg-slate-900/40 transition-colors"
                      >
                        <td className="py-2.5 px-3 font-semibold text-slate-900 dark:text-slate-100">
                          {r.target_date}
                        </td>
                        <td className="py-2.5 px-3 font-sans">
                          {r.state}
                        </td>
                        <td className="py-2.5 px-3 text-right font-bold text-executive-navy dark:text-executive-cyan">
                          {r.predicted_demand_mu?.toFixed(1)} MU
                        </td>
                        <td className="py-2.5 px-3 text-center text-slate-500">
                          {r.lower_bound_mu != null && r.upper_bound_mu != null
                            ? `[${r.lower_bound_mu.toFixed(0)} - ${r.upper_bound_mu.toFixed(0)}]`
                            : '—'}
                        </td>
                        <td className="py-2.5 px-3 text-right font-bold text-slate-800 dark:text-slate-200">
                          {isReconciled ? `${r.actual_demand_mu.toFixed(1)} MU` : '—'}
                        </td>
                        <td className={`py-2.5 px-3 text-right font-semibold ${
                          !isReconciled ? 'text-slate-400' :
                          r.error_mu > 0 ? 'text-amber-600 dark:text-amber-400' : 'text-indigo-600 dark:text-indigo-400'
                        }`}>
                          {isReconciled ? `${r.error_mu > 0 ? '+' : ''}${r.error_mu.toFixed(1)}` : '—'}
                        </td>
                        <td className="py-2.5 px-3 text-right font-semibold text-slate-700 dark:text-slate-300">
                          {r.pct_error != null ? `${r.pct_error.toFixed(1)}%` : '—'}
                        </td>
                        <td className="py-2.5 px-3 text-center font-sans">
                          {!isReconciled ? (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-semibold bg-slate-100 dark:bg-slate-800 text-slate-500 border border-slate-300 dark:border-slate-700">
                              <Clock className="w-3 h-3" /> Pending
                            </span>
                          ) : r.within_ci ? (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-bold bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800">
                              <Check className="w-3 h-3 text-emerald-500" /> In Band
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-bold bg-rose-50 dark:bg-rose-950/40 text-rose-700 dark:text-rose-300 border border-rose-300 dark:border-rose-800">
                              <AlertCircle className="w-3 h-3 text-rose-500" /> Divergent
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-executive-borderLight dark:border-executive-borderDark bg-slate-50 dark:bg-slate-900/50 flex items-center justify-between text-xs text-executive-mutedLight dark:text-executive-mutedDark">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-none bg-emerald-500"></span>
            <span>SQLite Embedded Ledger: <code className="font-mono text-[11px]">data/prediction_ledger.db</code></span>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-200 dark:bg-slate-800 hover:bg-slate-300 dark:hover:bg-slate-700 text-slate-800 dark:text-slate-200 text-xs font-semibold transition-colors"
          >
            Close
          </button>
        </div>

      </div>
    </div>
  );
}
