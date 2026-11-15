import React, { useState } from 'react';
import { 
  ResponsiveContainer, 
  ComposedChart, 
  AreaChart,
  Line, 
  Area, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ReferenceLine,
  ReferenceDot
} from 'recharts';
import { 
  TrendingUp, 
  Flame, 
  ShieldCheck, 
  Calendar, 
  AlertCircle, 
  ArrowUpRight, 
  ArrowDownRight,
  Info,
  Layers,
  SunMedium,
  Download,
  Globe,
  MapPin,
  Zap
} from 'lucide-react';

export default function ForecastTab({
  forecastData,
  loading,
  error,
  state,
  horizonDays,
  onChangeHorizon,
  theme,
  metadata
}) {
  const [showTable, setShowTable] = useState(true);

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        {/* Skeleton Horizon & Header */}
        <div className="flex justify-between items-center bg-white dark:bg-executive-cardDark p-4 rounded-none border border-executive-borderLight dark:border-executive-borderDark">
          <div className="h-6 w-48 bg-slate-200 dark:bg-slate-800 rounded-none"></div>
          <div className="h-9 w-40 bg-slate-200 dark:bg-slate-800 rounded-none"></div>
        </div>

        {/* Skeleton KPI Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-28 bg-white dark:bg-executive-cardDark p-4 rounded-none border border-executive-borderLight dark:border-executive-borderDark">
              <div className="h-4 w-20 bg-slate-200 dark:bg-slate-800 rounded-none mb-2"></div>
              <div className="h-7 w-24 bg-slate-200 dark:bg-slate-800 rounded-none"></div>
            </div>
          ))}
        </div>

        {/* Skeleton Chart */}
        <div className="h-96 bg-white dark:bg-executive-cardDark p-6 rounded-none border border-executive-borderLight dark:border-executive-borderDark"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 rounded-none bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900/50 text-rose-800 dark:text-rose-300 flex items-start gap-3">
        <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0 text-rose-600 dark:text-rose-400" />
        <div>
          <h4 className="font-bold text-sm">Failed to generate demand forecast</h4>
          <p className="text-xs mt-1">{error}</p>
        </div>
      </div>
    );
  }

  if (!forecastData) return null;

  const { summary, timeline = [], warnings = [] } = forecastData;

  // Chart theme colors from THEME.md
  const isDark = theme === 'dark';
  const colors = {
    demandLine: isDark ? '#38BDF8' : '#1E3A8A',
    ciFill: isDark ? 'rgba(56, 189, 248, 0.16)' : 'rgba(30, 58, 138, 0.12)',
    ciStroke: isDark ? '#0284C7' : '#3B82F6',
    tempMean: isDark ? '#FFB703' : '#EA580C',
    cddBar: isDark ? 'rgba(255, 77, 109, 0.35)' : 'rgba(225, 29, 72, 0.28)',
    cddBarHover: isDark ? '#FF4D6D' : '#E11D48',
    grid: isDark ? '#1E293B' : '#F1F5F9',
    text: isDark ? '#94A3B8' : '#64748B',
    peakMarker: isDark ? '#FF4D6D' : '#E11D48'
  };

  // Find peak timeline item
  const peakItem = timeline.find(t => t.date === summary.peak_demand_date);

  const isNational = Boolean(summary?.is_national || state === 'All-India');

  // Export CSV
  const handleExportCSV = () => {
    if (!timeline.length) return;
    const headers = isNational
      ? ['Date', 'National Forecast Demand (MU)', 'Lower Bound (MU)', 'Upper Bound (MU)', 'Northern NR (MU)', 'Western WR (MU)', 'Southern SR (MU)', 'Eastern ER (MU)', 'North-Eastern NER (MU)', 'Mean Temp (C)', 'Max Temp (C)', 'CDD', 'Extreme Heat', 'Holiday']
      : ['Date', 'Forecast Demand (MU)', 'Lower Bound (MU)', 'Upper Bound (MU)', 'Mean Temp (C)', 'Max Temp (C)', 'CDD', 'Extreme Heat', 'Holiday'];

    const rows = timeline.map(t => {
      if (isNational) {
        return [
          t.date,
          t.forecasted_demand_mu,
          t.lower_bound_mu ?? '',
          t.upper_bound_mu ?? '',
          t.regional_shares?.nr ?? '',
          t.regional_shares?.wr ?? '',
          t.regional_shares?.sr ?? '',
          t.regional_shares?.er ?? '',
          t.regional_shares?.ner ?? '',
          t.temperature_mean_c,
          t.temperature_max_c,
          t.cdd,
          t.extreme_heat ? 'Yes' : 'No',
          t.is_holiday ? 'Yes' : 'No'
        ];
      }
      return [
        t.date,
        t.forecasted_demand_mu,
        t.lower_bound_mu ?? '',
        t.upper_bound_mu ?? '',
        t.temperature_mean_c,
        t.temperature_max_c,
        t.cdd,
        t.extreme_heat ? 'Yes' : 'No',
        t.is_holiday ? 'Yes' : 'No'
      ];
    });

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `electricity_forecast_${state}_${horizonDays}d.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Prepare 5-Region stacked data for All-India view
  const regionalChartData = isNational
    ? timeline.map(t => ({
        date: t.date,
        total: t.forecasted_demand_mu,
        NR: t.regional_shares?.nr || 0,
        WR: t.regional_shares?.wr || 0,
        SR: t.regional_shares?.sr || 0,
        ER: t.regional_shares?.er || 0,
        NER: t.regional_shares?.ner || 0,
      }))
    : [];

  return (
    <div className="space-y-6">
      
      {/* Top Controls & State Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white dark:bg-executive-cardDark p-4 rounded-none border border-executive-borderLight dark:border-executive-borderDark shadow-sm">
        <div>
          <div className="flex items-center gap-2">
            {isNational ? (
              <>
                <Globe className="w-5 h-5 text-executive-navy dark:text-executive-cyan" />
                <h2 className="text-xl font-black tracking-tight text-executive-textLight dark:text-executive-textDark">
                  All-India National Grid
                </h2>
                <span className="text-xs px-2.5 py-0.5 rounded-none font-semibold bg-executive-navy/10 dark:bg-executive-cyan/20 text-executive-navy dark:text-executive-cyan border border-executive-navy/20 dark:border-executive-cyan/30">
                  National Aggregate • {horizonDays}-Day Outlook
                </span>
              </>
            ) : (
              <>
                <h2 className="text-xl font-black tracking-tight text-executive-textLight dark:text-executive-textDark">
                  {state}
                </h2>
                <span className="text-xs px-2.5 py-0.5 rounded-none font-medium bg-slate-100 dark:bg-slate-800 text-executive-mutedLight dark:text-executive-mutedDark">
                  {horizonDays}-Day Outlook
                </span>
              </>
            )}
          </div>
          <p className="text-xs text-executive-mutedLight dark:text-executive-mutedDark mt-0.5">
            {isNational
              ? 'Hierarchical bottom-up summation across 36 Indian States & UTs with diversified portfolio uncertainty envelope'
              : `${forecastData?.model_name || metadata?.model_type || 'Production ML'} predictions with 90% confidence uncertainty envelopes and thermal load features`}
          </p>
        </div>

        {/* Horizon Selector */}
        <div className="flex items-center gap-1.5 p-1 bg-slate-100 dark:bg-slate-900 rounded-none border border-slate-200 dark:border-slate-800 self-start sm:self-auto">
          {[7, 14, 30].map((days) => (
            <button
              key={days}
              onClick={() => onChangeHorizon(days)}
              className={`px-3 py-1.5 text-xs font-semibold rounded-none transition-all ${
                horizonDays === days
                  ? 'bg-white dark:bg-executive-cardDark text-executive-navy dark:text-executive-cyan shadow-sm border border-slate-200 dark:border-slate-700'
                  : 'text-executive-mutedLight dark:text-executive-mutedDark hover:text-executive-textLight dark:hover:text-executive-textDark'
              }`}
            >
              {days} Days
            </button>
          ))}
        </div>
      </div>

      {/* Warnings & Domain Context */}
      {warnings && warnings.length > 0 && (
        <div className="space-y-2">
          {warnings.map((warn, i) => (
            <div 
              key={i} 
              className="px-4 py-3 rounded-none bg-amber-50 dark:bg-amber-950/30 border border-amber-200/80 dark:border-amber-900/50 text-amber-900 dark:text-amber-200 text-xs flex items-start gap-2.5"
            >
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0 text-amber-600 dark:text-amber-400" />
              <span>{warn}</span>
            </div>
          ))}
        </div>
      )}

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        
        {/* Average Demand */}
        <div className="p-4 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between text-executive-mutedLight dark:text-executive-mutedDark">
            <span className="text-xs font-medium">{isNational ? 'Avg Daily National Load' : 'Avg Daily Demand'}</span>
            <TrendingUp className="w-4 h-4 text-executive-navy dark:text-executive-cyan" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black text-executive-textLight dark:text-executive-textDark">
              {summary.avg_daily_demand_mu}
              <span className="text-xs font-normal text-executive-mutedLight dark:text-executive-mutedDark ml-1">MU</span>
            </div>
            <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">
              {isNational ? 'National total across India' : `Across ${horizonDays} days`}
            </p>
          </div>
        </div>

        {/* Peak Demand */}
        <div className="p-4 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between text-executive-mutedLight dark:text-executive-mutedDark">
            <span className="text-xs font-medium">{isNational ? 'National Peak Forecast' : 'Peak Forecast'}</span>
            <ArrowUpRight className="w-4 h-4 text-rose-500" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black text-rose-600 dark:text-rose-400">
              {summary.peak_demand_mu}
              <span className="text-xs font-normal text-executive-mutedLight dark:text-executive-mutedDark ml-1">MU</span>
            </div>
            <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">On {summary.peak_demand_date}</p>
          </div>
        </div>

        {/* Min Demand */}
        <div className="p-4 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between text-executive-mutedLight dark:text-executive-mutedDark">
            <span className="text-xs font-medium">{isNational ? 'National Base Load' : 'Min Demand'}</span>
            <ArrowDownRight className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black text-executive-textLight dark:text-executive-textDark">
              {summary.min_demand_mu}
              <span className="text-xs font-normal text-executive-mutedLight dark:text-executive-mutedDark ml-1">MU</span>
            </div>
            <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">Trough baseline</p>
          </div>
        </div>

        {/* Extreme Heat Risk Days */}
        <div className="p-4 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between text-executive-mutedLight dark:text-executive-mutedDark">
            <span className="text-xs font-medium">Heat Risk Alerts</span>
            <Flame className={`w-4 h-4 ${summary.high_risk_days_count > 0 ? 'text-executive-orange dark:text-executive-gold' : 'text-slate-400'}`} />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black text-executive-orange dark:text-executive-gold">
              {summary.high_risk_days_count}
              <span className="text-xs font-normal text-executive-mutedLight dark:text-executive-mutedDark ml-1">Days &gt; 31.8°C</span>
            </div>
            <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">
              {isNational && summary.extreme_heat_states?.length > 0
                ? `${summary.extreme_heat_states.length} states impacted`
                : 'Severe thermal surge load'}
            </p>
          </div>
        </div>

        {/* State Model Uncertainty (RMSE) */}
        <div className="p-4 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark shadow-sm flex flex-col justify-between col-span-2 lg:col-span-1">
          <div className="flex items-center justify-between text-executive-mutedLight dark:text-executive-mutedDark">
            <span className="text-xs font-medium">{isNational ? 'Portfolio Sigma (90%)' : 'Model CI (90%)'}</span>
            <ShieldCheck className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black text-executive-textLight dark:text-executive-textDark">
              ±{summary.state_rmse ? summary.state_rmse : '3.20'}
              <span className="text-xs font-normal text-executive-mutedLight dark:text-executive-mutedDark ml-1">MU</span>
            </div>
            <p className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5">
              {isNational ? `Margin: ±${(summary.state_rmse * 1.645).toFixed(1)} MU` : 'State-specific validation error'}
            </p>
          </div>
        </div>

      </div>

      {/* Main Interactive Forecast Chart */}
      <div className="py-4 sm:py-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6">
          <div>
            <h3 className="text-base font-bold text-executive-textLight dark:text-executive-textDark flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-executive-navy dark:text-executive-cyan" />
              Electricity Demand & Weather Overlay
            </h3>
            <p className="text-xs text-executive-mutedLight dark:text-executive-mutedDark">
              Forecasted demand (MU) with 90% confidence bounds, Cooling Degree Days (CDD), and ambient temperature (°C)
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <button
              onClick={handleExportCSV}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-none border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Export CSV
            </button>
          </div>
        </div>

        <div className="h-80 sm:h-96 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={timeline} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="ciGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={isDark ? '#38BDF8' : '#1E3A8A'} stopOpacity={0.25} />
                  <stop offset="95%" stopColor={isDark ? '#38BDF8' : '#1E3A8A'} stopOpacity={0.02} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} vertical={false} />

              <XAxis 
                dataKey="date" 
                tick={{ fill: colors.text, fontSize: 11 }} 
                stroke={colors.grid}
                tickFormatter={(val) => {
                  const d = new Date(val);
                  return `${d.getDate()} ${d.toLocaleString('default', { month: 'short' })}`;
                }}
              />

              {/* Left Y Axis: Demand in MU */}
              <YAxis 
                yAxisId="demand" 
                tick={{ fill: colors.text, fontSize: 11 }} 
                stroke={colors.grid}
                domain={['auto', 'auto']}
                label={{ value: 'Demand (MU)', angle: -90, position: 'insideLeft', fill: colors.text, fontSize: 11 }}
              />

              {/* Right Y Axis: Temperature in C and CDD */}
              <YAxis 
                yAxisId="weather" 
                orientation="right" 
                tick={{ fill: colors.text, fontSize: 11 }} 
                stroke={colors.grid}
                domain={[0, 'auto']}
                label={{ value: 'Temp (°C) / CDD', angle: 90, position: 'insideRight', fill: colors.text, fontSize: 11 }}
              />

              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ paddingTop: '10px', fontSize: '12px' }} />

              {/* CDD Bar with sharp corners */}
              <Bar 
                yAxisId="weather" 
                dataKey="cdd" 
                name="Cooling Degree Days (CDD)" 
                fill={colors.cddBar} 
                radius={[0, 0, 0, 0]}
                barSize={16}
              />

              {/* Upper & Lower Confidence Envelopes */}
              <Line 
                yAxisId="demand" 
                type="monotone" 
                dataKey="upper_bound_mu" 
                name="Upper Bound (90% CI)" 
                stroke={colors.ciStroke} 
                strokeDasharray="4 4" 
                strokeWidth={1} 
                dot={false} 
              />
              <Line 
                yAxisId="demand" 
                type="monotone" 
                dataKey="lower_bound_mu" 
                name="Lower Bound (90% CI)" 
                stroke={colors.ciStroke} 
                strokeDasharray="4 4" 
                strokeWidth={1} 
                dot={false} 
              />

              {/* Main Forecasted Demand Line */}
              <Line 
                yAxisId="demand" 
                type="monotone" 
                dataKey="forecasted_demand_mu" 
                name="Forecast Demand (MU)" 
                stroke={colors.demandLine} 
                strokeWidth={3} 
                dot={{ r: 4, fill: colors.demandLine, strokeWidth: 2, stroke: '#fff' }}
                activeDot={{ r: 7, fill: colors.demandLine }}
              />

              {/* Temperature Mean Line */}
              <Line 
                yAxisId="weather" 
                type="monotone" 
                dataKey="temperature_mean_c" 
                name="Mean Temperature (°C)" 
                stroke={colors.tempMean} 
                strokeWidth={2} 
                dot={false} 
              />

              {/* Peak Marker Reference Dot */}
              {peakItem && (
                <ReferenceDot 
                  yAxisId="demand" 
                  x={peakItem.date} 
                  y={peakItem.forecasted_demand_mu} 
                  r={6} 
                  fill={colors.peakMarker} 
                  stroke="#ffffff"
                  strokeWidth={2}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* National 5-Region Breakdown & Drivers (Only in All-India Mode) */}
      {isNational && (
        <div className="space-y-6">
          {/* 5-Region Stacked Area Chart */}
          <div className="bg-white dark:bg-executive-cardDark p-4 sm:p-6 rounded-none border border-executive-borderLight dark:border-executive-borderDark shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
              <div>
                <h3 className="text-base font-bold text-executive-textLight dark:text-executive-textDark flex items-center gap-2">
                  <Layers className="w-4 h-4 text-executive-navy dark:text-executive-cyan" />
                  5-Region Hierarchical Demand Composition (NR, WR, SR, ER, NER)
                </h3>
                <p className="text-xs text-executive-mutedLight dark:text-executive-mutedDark">
                  Stacked daily power demand showing each region's real-time contribution towards India's ~5,300 MU national load
                </p>
              </div>
            </div>

            <div className="h-72 sm:h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={regionalChartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} vertical={false} />
                  <XAxis 
                    dataKey="date" 
                    tick={{ fill: colors.text, fontSize: 11 }} 
                    stroke={colors.grid}
                    tickFormatter={(val) => {
                      const d = new Date(val);
                      return `${d.getDate()} ${d.toLocaleString('default', { month: 'short' })}`;
                    }}
                  />
                  <YAxis 
                    tick={{ fill: colors.text, fontSize: 11 }} 
                    stroke={colors.grid}
                    label={{ value: 'Regional Load (MU)', angle: -90, position: 'insideLeft', fill: colors.text, fontSize: 11 }}
                  />
                  <Tooltip content={<RegionalTooltip />} />
                  <Legend wrapperStyle={{ paddingTop: '10px', fontSize: '12px' }} />
                  <Area type="monotone" dataKey="NR" name="Northern Region (NR)" stackId="1" stroke="#0284C7" fill="#0284C7" fillOpacity={0.65} />
                  <Area type="monotone" dataKey="WR" name="Western Region (WR)" stackId="1" stroke="#D97706" fill="#D97706" fillOpacity={0.65} />
                  <Area type="monotone" dataKey="SR" name="Southern Region (SR)" stackId="1" stroke="#059669" fill="#059669" fillOpacity={0.65} />
                  <Area type="monotone" dataKey="ER" name="Eastern Region (ER)" stackId="1" stroke="#7C3AED" fill="#7C3AED" fillOpacity={0.65} />
                  <Area type="monotone" dataKey="NER" name="North-Eastern (NER)" stackId="1" stroke="#E11D48" fill="#E11D48" fillOpacity={0.65} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Regional Shares & Top Contributor States Cards Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            
            {/* Top 5 Driver States */}
            <div className="bg-white dark:bg-executive-cardDark p-5 rounded-none border border-executive-borderLight dark:border-executive-borderDark shadow-sm">
              <div className="flex items-center justify-between pb-3 border-b border-executive-borderLight dark:border-executive-borderDark">
                <h4 className="text-xs font-bold uppercase tracking-wider text-executive-textLight dark:text-executive-textDark flex items-center gap-2">
                  <MapPin className="w-3.5 h-3.5 text-executive-navy dark:text-executive-cyan" />
                  Top 5 Power Consuming States
                </h4>
                <span className="text-[11px] font-medium text-executive-mutedLight dark:text-executive-mutedDark">
                  % of All-India Load
                </span>
              </div>
              <div className="mt-4 space-y-3">
                {summary.top_contributors?.map((item, idx) => (
                  <div key={item.state} className="space-y-1">
                    <div className="flex justify-between items-center text-xs">
                      <div className="flex items-center gap-2">
                        <span className="w-5 h-5 rounded-none bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 font-bold flex items-center justify-center text-[10px]">
                          {idx + 1}
                        </span>
                        <span className="font-semibold text-executive-textLight dark:text-executive-textDark">
                          {item.state}
                        </span>
                        <span className="text-[10px] px-1.5 py-0.2 rounded-none bg-slate-100 dark:bg-slate-800 text-slate-500 font-medium">
                          {item.region}
                        </span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="font-bold text-executive-navy dark:text-executive-cyan">
                          {item.avg_demand_mu} MU/day
                        </span>
                        <span className="font-black text-slate-600 dark:text-slate-300 w-12 text-right">
                          {item.share_pct}%
                        </span>
                      </div>
                    </div>
                    {/* Progress Bar */}
                    <div className="w-full bg-slate-100 dark:bg-slate-800 h-1.5 rounded-none overflow-hidden">
                      <div 
                        className="bg-executive-navy dark:bg-executive-cyan h-full transition-all"
                        style={{ width: `${Math.min(100, item.share_pct * 5)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 5-Region Aggregate Breakdown */}
            <div className="bg-white dark:bg-executive-cardDark p-5 rounded-none border border-executive-borderLight dark:border-executive-borderDark shadow-sm">
              <div className="flex items-center justify-between pb-3 border-b border-executive-borderLight dark:border-executive-borderDark">
                <h4 className="text-xs font-bold uppercase tracking-wider text-executive-textLight dark:text-executive-textDark flex items-center gap-2">
                  <Globe className="w-3.5 h-3.5 text-emerald-500" />
                  Regional Grid Slices (5 Regions)
                </h4>
                <span className="text-[11px] font-medium text-executive-mutedLight dark:text-executive-mutedDark">
                  Avg Daily Demand
                </span>
              </div>
              <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
                {[
                  { key: 'nr', name: 'Northern Region (NR)', color: 'text-sky-600 dark:text-sky-400', border: 'border-sky-500/30' },
                  { key: 'wr', name: 'Western Region (WR)', color: 'text-amber-600 dark:text-amber-400', border: 'border-amber-500/30' },
                  { key: 'sr', name: 'Southern Region (SR)', color: 'text-emerald-600 dark:text-emerald-400', border: 'border-emerald-500/30' },
                  { key: 'er', name: 'Eastern Region (ER)', color: 'text-purple-600 dark:text-purple-400', border: 'border-purple-500/30' },
                  { key: 'ner', name: 'North-Eastern (NER)', color: 'text-rose-600 dark:text-rose-400', border: 'border-rose-500/30' }
                ].map((reg) => {
                  const val = summary.regional_breakdown?.[reg.key] || 0;
                  const pct = summary.avg_daily_demand_mu > 0 ? (val / summary.avg_daily_demand_mu * 100).toFixed(1) : 0;
                  return (
                    <div key={reg.key} className={`p-3 rounded-none bg-slate-50 dark:bg-slate-900/40 border ${reg.border} flex flex-col justify-between`}>
                      <span className="text-[11px] font-medium text-slate-500 dark:text-slate-400">{reg.name}</span>
                      <div className="mt-1 flex items-baseline justify-between">
                        <span className={`text-base font-black ${reg.color}`}>{val.toFixed(1)} <span className="text-xs font-normal text-slate-400">MU</span></span>
                        <span className="text-xs font-bold text-slate-700 dark:text-slate-200">{pct}%</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

          </div>
        </div>
      )}

      {/* Daily Breakdown Table Section */}
      <div className="bg-white dark:bg-executive-cardDark rounded-none border border-executive-borderLight dark:border-executive-borderDark shadow-sm overflow-hidden">
        <div 
          onClick={() => setShowTable(!showTable)}
          className="flex items-center justify-between px-6 py-4 cursor-pointer hover:bg-slate-50/60 dark:hover:bg-slate-900/30 transition-colors border-b border-executive-borderLight dark:border-executive-borderDark"
        >
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-executive-navy dark:text-executive-cyan" />
            <h3 className="text-sm font-bold text-executive-textLight dark:text-executive-textDark">
              Daily Forecast Schedule & Thermal Metrics ({timeline.length} Days)
            </h3>
          </div>
          <span className="text-xs font-semibold text-executive-navy dark:text-executive-cyan">
            {showTable ? 'Hide Table' : 'Show Table'}
          </span>
        </div>

        {showTable && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 dark:bg-slate-900/60 border-b border-slate-200/80 dark:border-slate-800 text-slate-500 dark:text-slate-400 font-semibold">
                <tr>
                  <th className="py-3 px-4">Date</th>
                  <th className="py-3 px-4">{isNational ? 'National Total' : 'Forecast Demand'}</th>
                  <th className="py-3 px-4">90% CI Envelope</th>
                  {isNational && (
                    <>
                      <th className="py-3 px-4">NR (MU)</th>
                      <th className="py-3 px-4">WR (MU)</th>
                      <th className="py-3 px-4">SR (MU)</th>
                      <th className="py-3 px-4">ER (MU)</th>
                      <th className="py-3 px-4">NER (MU)</th>
                    </>
                  )}
                  <th className="py-3 px-4">Mean Temp</th>
                  <th className="py-3 px-4">Max Temp</th>
                  <th className="py-3 px-4">CDD</th>
                  <th className="py-3 px-4">Extreme Heat</th>
                  <th className="py-3 px-4">Holiday</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80 text-executive-textLight dark:text-executive-textDark">
                {timeline.map((row) => {
                  const isPeak = row.date === summary.peak_demand_date;
                  return (
                    <tr 
                      key={row.date} 
                      className={`hover:bg-slate-50/80 dark:hover:bg-slate-900/40 transition-colors ${
                        isPeak ? 'bg-rose-50/40 dark:bg-rose-950/20' : ''
                      }`}
                    >
                      <td className="py-3 px-4 font-medium flex items-center gap-2">
                        <span>{row.date}</span>
                        {isPeak && (
                          <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-none bg-rose-100 dark:bg-rose-900/60 text-rose-700 dark:text-rose-300">
                            PEAK
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-4 font-bold text-executive-navy dark:text-executive-cyan">
                        {row.forecasted_demand_mu.toFixed(2)} MU
                      </td>
                      <td className="py-3 px-4 text-slate-500 dark:text-slate-400">
                        {row.lower_bound_mu !== null && row.upper_bound_mu !== null
                          ? `${row.lower_bound_mu.toFixed(1)} – ${row.upper_bound_mu.toFixed(1)} MU`
                          : '—'}
                      </td>
                      {isNational && (
                        <>
                          <td className="py-3 px-4 text-sky-600 dark:text-sky-400 font-medium">
                            {row.regional_shares?.nr?.toFixed(1) ?? '—'}
                          </td>
                          <td className="py-3 px-4 text-amber-600 dark:text-amber-400 font-medium">
                            {row.regional_shares?.wr?.toFixed(1) ?? '—'}
                          </td>
                          <td className="py-3 px-4 text-emerald-600 dark:text-emerald-400 font-medium">
                            {row.regional_shares?.sr?.toFixed(1) ?? '—'}
                          </td>
                          <td className="py-3 px-4 text-purple-600 dark:text-purple-400 font-medium">
                            {row.regional_shares?.er?.toFixed(1) ?? '—'}
                          </td>
                          <td className="py-3 px-4 text-rose-600 dark:text-rose-400 font-medium">
                            {row.regional_shares?.ner?.toFixed(1) ?? '—'}
                          </td>
                        </>
                      )}
                      <td className="py-3 px-4">
                        {row.temperature_mean_c.toFixed(1)} °C
                      </td>
                      <td className="py-3 px-4 font-medium text-amber-600 dark:text-amber-400">
                        {row.temperature_max_c.toFixed(1)} °C
                      </td>
                      <td className="py-3 px-4">
                        {row.cdd.toFixed(2)}
                      </td>
                      <td className="py-3 px-4">
                        {row.extreme_heat === 1 ? (
                          <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-950/50 px-2 py-0.5 rounded-none border border-rose-200 dark:border-rose-900/60">
                            <Flame className="w-3 h-3" /> Surge (&gt;31.8°C)
                          </span>
                        ) : (
                          <span className="text-slate-400 text-[11px]">Normal</span>
                        )}
                      </td>
                      <td className="py-3 px-4">
                        {row.is_holiday === 1 ? (
                          <span className="text-[11px] font-medium text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-950/40 px-2 py-0.5 rounded-none border border-purple-200 dark:border-purple-800">
                            Holiday
                          </span>
                        ) : (
                          <span className="text-slate-400 text-[11px]">No</span>
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

    </div>
  );
}

// Custom Tooltip for Recharts
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;

  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <div className="bg-white/95 dark:bg-slate-900/95 backdrop-blur-md p-3.5 rounded-none border border-slate-200 dark:border-slate-800 shadow-xl text-xs space-y-2 min-w-[200px]">
      <div className="font-bold border-b border-slate-200 dark:border-slate-800 pb-1.5 flex items-center justify-between">
        <span>{data.date}</span>
        {data.extreme_heat === 1 && (
          <span className="text-[10px] text-rose-500 font-semibold flex items-center gap-0.5">
            <Flame className="w-3 h-3" /> Heat Surge
          </span>
        )}
      </div>

      <div className="space-y-1">
        <div className="flex justify-between items-center">
          <span className="text-slate-500 dark:text-slate-400 font-medium">Demand Forecast:</span>
          <span className="font-bold text-blue-600 dark:text-sky-400">
            {data.forecasted_demand_mu?.toFixed(2)} MU
          </span>
        </div>

        {data.lower_bound_mu !== null && data.upper_bound_mu !== null && (
          <div className="flex justify-between items-center text-[11px] text-slate-500">
            <span>90% CI Range:</span>
            <span>{data.lower_bound_mu?.toFixed(1)} – {data.upper_bound_mu?.toFixed(1)} MU</span>
          </div>
        )}

        <div className="flex justify-between items-center">
          <span className="text-slate-500 dark:text-slate-400">Mean Temp:</span>
          <span className="font-semibold text-amber-600 dark:text-amber-400">{data.temperature_mean_c?.toFixed(1)} °C</span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-slate-500 dark:text-slate-400">Max Temp:</span>
          <span className="font-semibold text-rose-500">{data.temperature_max_c?.toFixed(1)} °C</span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-slate-500 dark:text-slate-400">Cooling Deg Days (CDD):</span>
          <span className="font-semibold">{data.cdd?.toFixed(2)}</span>
        </div>

        {data.is_holiday === 1 && (
          <div className="pt-1 text-[11px] text-purple-600 dark:text-purple-400 font-medium">
            Calendar Holiday
          </div>
        )}
      </div>
    </div>
  );
}

// Tooltip for 5-Region Stacked Area Chart
function RegionalTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <div className="bg-white/95 dark:bg-slate-900/95 backdrop-blur-md p-3.5 rounded-none border border-slate-200 dark:border-slate-800 shadow-xl text-xs space-y-2 min-w-[220px]">
      <div className="font-bold border-b border-slate-200 dark:border-slate-800 pb-1.5 flex items-center justify-between">
        <span>{data.date}</span>
        <span className="text-blue-600 dark:text-sky-400 font-black">
          Total: {data.total?.toFixed(1)} MU
        </span>
      </div>
      <div className="space-y-1">
        <div className="flex justify-between items-center text-sky-600 dark:text-sky-400">
          <span>Northern (NR):</span>
          <span className="font-semibold">{data.NR?.toFixed(1)} MU</span>
        </div>
        <div className="flex justify-between items-center text-amber-600 dark:text-amber-400">
          <span>Western (WR):</span>
          <span className="font-semibold">{data.WR?.toFixed(1)} MU</span>
        </div>
        <div className="flex justify-between items-center text-emerald-600 dark:text-emerald-400">
          <span>Southern (SR):</span>
          <span className="font-semibold">{data.SR?.toFixed(1)} MU</span>
        </div>
        <div className="flex justify-between items-center text-purple-600 dark:text-purple-400">
          <span>Eastern (ER):</span>
          <span className="font-semibold">{data.ER?.toFixed(1)} MU</span>
        </div>
        <div className="flex justify-between items-center text-rose-600 dark:text-rose-400">
          <span>North-Eastern (NER):</span>
          <span className="font-semibold">{data.NER?.toFixed(1)} MU</span>
        </div>
      </div>
    </div>
  );
}

