import React, { useState, useEffect } from 'react';
import { 
  ResponsiveContainer, 
  ComposedChart, 
  Line, 
  BarChart,
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  Cell
} from 'recharts';
import { 
  Flame, 
  Sun, 
  Wind, 
  ThermometerSun, 
  Sparkles, 
  BarChart3, 
  Layers, 
  CloudRain, 
  AlertCircle,
  TrendingUp,
  Snowflake,
  SunDim
} from 'lucide-react';
import { api } from '../services/api';

const FEATURE_COLORS = {
  Temperature: '#EA580C',
  Humidity: '#0D9488',
  Calendar: '#7C3AED',
  Weather: '#F59E0B',
  Historical: '#1E3A8A',
  Location: '#64748B',
  Other: '#94A3B8'
};

const SEASON_ICONS = {
  Summer: Sun,
  Monsoon: CloudRain,
  Autumn: Wind,
  Winter: Snowflake
};

export default function WeatherImpactTab({
  state,
  horizonDays,
  useWeatherApi,
  theme
}) {
  const [mode, setMode] = useState('forecast'); // 'forecast' | 'historical'
  const [weatherData, setWeatherData] = useState(null);
  const [featureData, setFeatureData] = useState([]);
  const [seasonalData, setSeasonalData] = useState(null);
  const [weatherOnly, setWeatherOnly] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const isDark = theme === 'dark';

  // Fetch weather impact, feature importance, and seasonal data
  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    setError(null);

    Promise.all([
      api.getWeatherImpact({ state, mode, horizonDays, useWeatherApi }),
      api.getFeatureImportance({ topN: 12, weatherOnly }),
      api.getSeasonalComparison(state)
    ])
      .then(([wImpact, fImp, sComp]) => {
        if (!isMounted) return;
        setWeatherData(wImpact);
        setFeatureData(fImp?.features || []);
        setSeasonalData(sComp?.seasons || null);
        setLoading(false);
      })
      .catch((err) => {
        if (!isMounted) return;
        setError(err.message || 'Failed to load weather analytics');
        setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [state, mode, horizonDays, useWeatherApi, weatherOnly]);

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-20 bg-white dark:bg-executive-cardDark rounded-none border border-executive-borderLight dark:border-executive-borderDark"></div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="h-32 bg-white dark:bg-executive-cardDark rounded-none border border-executive-borderLight dark:border-executive-borderDark"></div>
          <div className="h-32 bg-white dark:bg-executive-cardDark rounded-none border border-executive-borderLight dark:border-executive-borderDark"></div>
        </div>
        <div className="h-96 bg-white dark:bg-executive-cardDark rounded-none border border-executive-borderLight dark:border-executive-borderDark"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 rounded-none bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900/50 text-rose-800 dark:text-rose-300 flex items-start gap-3">
        <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0 text-rose-600 dark:text-rose-400" />
        <div>
          <h4 className="font-bold text-sm">Failed to retrieve weather analytics</h4>
          <p className="text-xs mt-1">{error}</p>
        </div>
      </div>
    );
  }

  const correlations = weatherData?.correlations || {};
  const series = weatherData?.series || [];

  const tempCorr = correlations.temperature_demand_corr;
  const cddCorr = correlations.cdd_demand_corr;

  const getCorrBadge = (val) => {
    if (val === null || val === undefined) return { label: 'N/A', color: 'bg-slate-100 text-slate-600' };
    if (val >= 0.7) return { label: 'Strong Positive (+)', color: 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300' };
    if (val >= 0.4) return { label: 'Moderate Positive (+)', color: 'bg-amber-100 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300' };
    if (val <= -0.4) return { label: 'Inverse Trend (-)', color: 'bg-blue-100 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300' };
    return { label: 'Mild Correlation', color: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300' };
  };

  const tempBadge = getCorrBadge(tempCorr);
  const cddBadge = getCorrBadge(cddCorr);

  return (
    <div className="space-y-6">
      
      {/* Top Controls & Mode Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white dark:bg-executive-cardDark p-4 rounded-none border border-executive-borderLight dark:border-executive-borderDark shadow-sm">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-xl font-black tracking-tight text-executive-textLight dark:text-executive-textDark">
              {state} — Thermal & Weather Sensitivity
            </h2>
          </div>
          <p className="text-xs text-executive-mutedLight dark:text-executive-mutedDark mt-0.5">
            Analysis of ambient temperature, Cooling Degree Days (CDD), and heatwave load surges
          </p>
        </div>

        {/* Mode Selector */}
        <div className="flex items-center gap-1.5 p-1 bg-slate-100 dark:bg-slate-900 rounded-none border border-slate-200 dark:border-slate-800">
          <button
            onClick={() => setMode('forecast')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-none transition-all ${
              mode === 'forecast'
                ? 'bg-white dark:bg-executive-cardDark text-executive-navy dark:text-executive-cyan shadow-sm border border-slate-200 dark:border-slate-700'
                : 'text-executive-mutedLight dark:text-executive-mutedDark'
            }`}
          >
            Forecast Outlook ({horizonDays}d)
          </button>
          <button
            onClick={() => setMode('historical')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-none transition-all ${
              mode === 'historical'
                ? 'bg-white dark:bg-executive-cardDark text-executive-navy dark:text-executive-cyan shadow-sm border border-slate-200 dark:border-slate-700'
                : 'text-executive-mutedLight dark:text-executive-mutedDark'
            }`}
          >
            Historical Dataset
          </button>
        </div>
      </div>

      {/* Natural Language AI Insight Card */}
      {weatherData?.natural_language_insight && (
        <div className="p-5 rounded-none bg-gradient-to-r from-amber-500/10 via-orange-500/10 to-rose-500/10 dark:from-amber-950/30 dark:via-orange-950/30 dark:to-rose-950/30 border border-orange-200/80 dark:border-orange-900/40">
          <div className="flex items-center gap-2 text-executive-orange dark:text-executive-gold font-bold text-xs uppercase tracking-wider mb-3">
            <Sparkles className="w-4 h-4" />
            <span>Thermal Load Insight Engine</span>
          </div>
          <FormattedInsightText text={weatherData.natural_language_insight} />
        </div>
      )}

      {/* Correlation KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        {/* Temp vs Demand Corr */}
        <div className="p-5 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark shadow-sm flex items-center justify-between">
          <div className="space-y-1">
            <div className="text-xs font-semibold text-executive-mutedLight dark:text-executive-mutedDark flex items-center gap-1.5">
              <ThermometerSun className="w-4 h-4 text-executive-orange dark:text-executive-gold" />
              Temperature ↔ Demand Correlation
            </div>
            <div className="text-3xl font-black text-executive-textLight dark:text-executive-textDark">
              {tempCorr !== null && tempCorr !== undefined ? (tempCorr > 0 ? `+${tempCorr.toFixed(3)}` : tempCorr.toFixed(3)) : 'N/A'}
            </div>
            <span className={`inline-block text-[11px] font-semibold px-2 py-0.5 rounded-none ${tempBadge.color}`}>
              {tempBadge.label}
            </span>
          </div>
          <div className="w-12 h-12 rounded-none bg-orange-500/10 text-orange-600 dark:text-orange-400 flex items-center justify-center flex-shrink-0">
            <Sun className="w-6 h-6" />
          </div>
        </div>

        {/* CDD vs Demand Corr */}
        <div className="p-5 rounded-none bg-white dark:bg-executive-cardDark border border-executive-borderLight dark:border-executive-borderDark shadow-sm flex items-center justify-between">
          <div className="space-y-1">
            <div className="text-xs font-semibold text-executive-mutedLight dark:text-executive-mutedDark flex items-center gap-1.5">
              <Flame className="w-4 h-4 text-rose-500" />
              Cooling Degree Days (CDD) ↔ Demand Correlation
            </div>
            <div className="text-3xl font-black text-rose-600 dark:text-rose-400">
              {cddCorr !== null && cddCorr !== undefined ? (cddCorr > 0 ? `+${cddCorr.toFixed(3)}` : cddCorr.toFixed(3)) : 'N/A'}
            </div>
            <span className={`inline-block text-[11px] font-semibold px-2 py-0.5 rounded-none ${cddBadge.color}`}>
              {cddBadge.label}
            </span>
          </div>
          <div className="w-12 h-12 rounded-none bg-rose-500/10 text-rose-600 dark:text-rose-400 flex items-center justify-center flex-shrink-0">
            <Flame className="w-6 h-6" />
          </div>
        </div>

      </div>

      {/* Multi-Metric Time Series Overlay Chart */}
      <div className="py-4 sm:py-6">
        <div className="mb-6">
          <h3 className="text-base font-bold text-executive-textLight dark:text-executive-textDark flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-executive-navy dark:text-executive-cyan" />
            Demand & Thermal Drivers Overlay
          </h3>
          <p className="text-xs text-executive-mutedLight dark:text-executive-mutedDark">
            Comparison of predicted electricity demand (MU), 7-day historical rolling baseline, and ambient temperatures
          </p>
        </div>

        <div className="h-80 sm:h-96 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={series} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1E293B' : '#F1F5F9'} vertical={false} />
              
              <XAxis 
                dataKey="date" 
                tick={{ fill: isDark ? '#94A3B8' : '#64748B', fontSize: 11 }} 
                stroke={isDark ? '#1E293B' : '#F1F5F9'}
                tickFormatter={(val) => {
                  const d = new Date(val);
                  return `${d.getDate()} ${d.toLocaleString('default', { month: 'short' })}`;
                }}
              />

              <YAxis 
                yAxisId="demand" 
                tick={{ fill: isDark ? '#94A3B8' : '#64748B', fontSize: 11 }} 
                stroke={isDark ? '#1E293B' : '#F1F5F9'}
                domain={['auto', 'auto']}
                label={{ value: 'Demand (MU)', angle: -90, position: 'insideLeft', fill: isDark ? '#94A3B8' : '#64748B', fontSize: 11 }}
              />

              <YAxis 
                yAxisId="temp" 
                orientation="right" 
                tick={{ fill: isDark ? '#94A3B8' : '#64748B', fontSize: 11 }} 
                stroke={isDark ? '#1E293B' : '#F1F5F9'}
                domain={['auto', 'auto']}
                label={{ value: 'Temperature (°C)', angle: 90, position: 'insideRight', fill: isDark ? '#94A3B8' : '#64748B', fontSize: 11 }}
              />

              <Tooltip content={<WeatherTooltip />} />
              <Legend wrapperStyle={{ paddingTop: '10px', fontSize: '12px' }} />

              {/* Baseline 7D Line */}
              {series.some(s => s.baseline_7d_mu !== null && s.baseline_7d_mu !== undefined) && (
                <Line 
                  yAxisId="demand" 
                  type="monotone" 
                  dataKey="baseline_7d_mu" 
                  name="7-Day Historical Baseline (MU)" 
                  stroke="#64748B" 
                  strokeDasharray="4 4" 
                  strokeWidth={2} 
                  dot={false} 
                />
              )}

              {/* Forecast Demand Line */}
              <Line 
                yAxisId="demand" 
                type="monotone" 
                dataKey="forecasted_demand_mu" 
                name="Electricity Demand (MU)" 
                stroke={isDark ? '#38BDF8' : '#1E3A8A'} 
                strokeWidth={3} 
                dot={{ r: 3, fill: isDark ? '#38BDF8' : '#1E3A8A' }} 
              />

              {/* Mean Temp Line */}
              <Line 
                yAxisId="temp" 
                type="monotone" 
                dataKey="temperature_mean_c" 
                name="Mean Temperature (°C)" 
                stroke={isDark ? '#FFB703' : '#EA580C'} 
                strokeWidth={2} 
                dot={false} 
              />

              {/* Max Temp Line */}
              <Line 
                yAxisId="temp" 
                type="monotone" 
                dataKey="temperature_max_c" 
                name="Max Temperature (°C)" 
                stroke={isDark ? '#FF4D6D' : '#E11D48'} 
                strokeWidth={1.5} 
                strokeDasharray="2 2"
                dot={false} 
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Two Column Grid: Feature Importance & Seasonal Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Feature Importance */}
        <div className="bg-white dark:bg-executive-cardDark p-5 sm:p-6 rounded-none border border-executive-borderLight dark:border-executive-borderDark shadow-sm flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 mb-4">
              <div>
                <h3 className="text-base font-bold text-executive-textLight dark:text-executive-textDark flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-executive-navy dark:text-executive-cyan" />
                  ML Feature Importance Breakdown
                </h3>
                <p className="text-xs text-executive-mutedLight dark:text-executive-mutedDark">
                  Model weight contribution ranked by Gain score
                </p>
              </div>

              {/* Weather Filter Button */}
              <button
                onClick={() => setWeatherOnly(!weatherOnly)}
                className={`text-xs px-2.5 py-1 rounded-none border font-medium transition-colors ${
                  weatherOnly
                    ? 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300 border-amber-300 dark:border-amber-700'
                    : 'bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-slate-800'
                }`}
              >
                {weatherOnly ? 'Weather Features' : 'All Features'}
              </button>
            </div>

            {/* Feature Legend */}
            <div className="flex flex-wrap gap-2 mb-4">
              {Object.entries(FEATURE_COLORS).map(([cat, col]) => (
                <span key={cat} className="inline-flex items-center gap-1 text-[10px] font-medium text-slate-600 dark:text-slate-400">
                  <span className="w-2 h-2 rounded-none" style={{ backgroundColor: col }} />
                  {cat}
                </span>
              ))}
            </div>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={featureData} layout="vertical" margin={{ top: 5, right: 30, left: 70, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={isDark ? '#1E293B' : '#F1F5F9'} horizontal={false} />
                  <XAxis type="number" tick={{ fill: isDark ? '#94A3B8' : '#64748B', fontSize: 10 }} />
                  <YAxis type="category" dataKey="name" tick={{ fill: isDark ? '#94A3B8' : '#64748B', fontSize: 10 }} width={80} />
                  <Tooltip 
                    formatter={(val, name, item) => [`${(val * 100).toFixed(2)}%`, `Category: ${item.payload.category}`]}
                  />
                  <Bar dataKey="importance" radius={[0, 0, 0, 0]}>
                    {featureData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={FEATURE_COLORS[entry.category] || FEATURE_COLORS.Other} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Seasonal Breakdown */}
        <div className="bg-white dark:bg-executive-cardDark p-5 sm:p-6 rounded-none border border-executive-borderLight dark:border-executive-borderDark shadow-sm">
          <div className="mb-4">
            <h3 className="text-base font-bold text-executive-textLight dark:text-executive-textDark flex items-center gap-2">
              <Layers className="w-4 h-4 text-executive-navy dark:text-executive-cyan" />
              Seasonal Profile Breakdown ({state})
            </h3>
            <p className="text-xs text-executive-mutedLight dark:text-executive-mutedDark">
              Historical multi-year averages across Indian meteorological seasons
            </p>
          </div>

          {seasonalData ? (
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(seasonalData).map(([seasonName, stats]) => {
                const Icon = SEASON_ICONS[seasonName] || Sun;
                return (
                  <div 
                    key={seasonName}
                    className="p-3.5 rounded-none border border-executive-borderLight dark:border-executive-borderDark bg-slate-50/50 dark:bg-slate-900/40 space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-executive-textLight dark:text-executive-textDark">
                        {seasonName}
                      </span>
                      <Icon className="w-4 h-4 text-executive-orange dark:text-executive-gold" />
                    </div>

                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between text-slate-500 dark:text-slate-400">
                        <span>Mean Temp:</span>
                        <span className="font-semibold text-executive-textLight dark:text-executive-textDark">{stats.temp_mean}°C</span>
                      </div>
                      <div className="flex justify-between text-slate-500 dark:text-slate-400">
                        <span>Max Temp:</span>
                        <span className="font-semibold text-rose-500">{stats.temp_max}°C</span>
                      </div>
                      <div className="flex justify-between text-slate-500 dark:text-slate-400">
                        <span>Mean CDD:</span>
                        <span className="font-semibold text-executive-textLight dark:text-executive-textDark">{stats.cdd_mean}</span>
                      </div>
                      <div className="flex justify-between text-slate-500 dark:text-slate-400">
                        <span>Extreme Heat:</span>
                        <span className={`font-semibold ${stats.extreme_heat_pct > 15 ? 'text-rose-500' : 'text-slate-500'}`}>
                          {stats.extreme_heat_pct}%
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-xs text-slate-400 italic py-8 text-center">
              Seasonal statistics not available for {state}
            </div>
          )}
        </div>

      </div>

    </div>
  );
}

function WeatherTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <div className="bg-white/95 dark:bg-slate-900/95 backdrop-blur-md p-3 rounded-none border border-slate-200 dark:border-slate-800 shadow-xl text-xs space-y-1.5 min-w-[180px]">
      <div className="font-bold border-b border-slate-200 dark:border-slate-800 pb-1">
        {data.date}
      </div>
      <div className="flex justify-between">
        <span className="text-slate-500">Demand:</span>
        <span className="font-bold text-blue-600 dark:text-sky-400">{data.forecasted_demand_mu?.toFixed(2)} MU</span>
      </div>
      {data.baseline_7d_mu !== null && data.baseline_7d_mu !== undefined && (
        <div className="flex justify-between text-slate-500">
          <span>7d Baseline:</span>
          <span>{data.baseline_7d_mu?.toFixed(2)} MU</span>
        </div>
      )}
      <div className="flex justify-between">
        <span className="text-slate-500">Mean Temp:</span>
        <span className="font-semibold text-amber-500">{data.temperature_mean_c?.toFixed(1)} °C</span>
      </div>
      <div className="flex justify-between">
        <span className="text-slate-500">Max Temp:</span>
        <span className="font-semibold text-rose-500">{data.temperature_max_c?.toFixed(1)} °C</span>
      </div>
      <div className="flex justify-between">
        <span className="text-slate-500">CDD:</span>
        <span className="font-semibold">{data.cdd?.toFixed(2)}</span>
      </div>
    </div>
  );
}

function FormattedInsightText({ text }) {
  if (!text) return null;

  // Split insight paragraphs by **Header**:
  const sections = text.split(/(?=\*\*[^*]+\*\*:?)/g).filter(p => p.trim());

  if (sections.length > 1) {
    return (
      <div className="space-y-3">
        {sections.map((section, idx) => {
          const match = section.match(/^\*\*([^*]+)\*\*:?\s*(.*)$/s);
          if (match) {
            const title = match[1];
            const body = match[2];
            return (
              <div key={idx} className="flex items-start gap-2.5 text-xs sm:text-sm">
                <span className="w-1.5 h-1.5 rounded-none bg-executive-navy dark:bg-executive-cyan mt-1.5 flex-shrink-0" />
                <div className="leading-relaxed">
                  <span className="font-bold text-executive-navy dark:text-executive-cyan mr-1.5">
                    {title}:
                  </span>
                  <span className="text-executive-textLight dark:text-executive-textDark">
                    {renderInlineMarkdown(body)}
                  </span>
                </div>
              </div>
            );
          }
          return (
            <p key={idx} className="text-xs sm:text-sm leading-relaxed text-executive-textLight dark:text-executive-textDark">
              {renderInlineMarkdown(section)}
            </p>
          );
        })}
      </div>
    );
  }

  return (
    <div className="text-xs sm:text-sm leading-relaxed text-executive-textLight dark:text-executive-textDark">
      {renderInlineMarkdown(text)}
    </div>
  );
}

function renderInlineMarkdown(text) {
  if (!text) return null;
  const tokens = [];
  const regex = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      tokens.push(text.slice(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith('**') && token.endsWith('**')) {
      tokens.push(
        <strong key={match.index} className="font-bold text-executive-navy dark:text-executive-cyan">
          {token.slice(2, -2)}
        </strong>
      );
    } else if (token.startsWith('*') && token.endsWith('*')) {
      tokens.push(
        <em key={match.index} className="italic text-slate-500 dark:text-slate-400">
          {token.slice(1, -1)}
        </em>
      );
    }
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    tokens.push(text.slice(lastIndex));
  }

  return tokens;
}

