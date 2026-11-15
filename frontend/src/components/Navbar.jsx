import React from 'react';
import { Sun, Moon } from 'lucide-react';

const REGION_LABELS = {
  nr: 'Northern Region (NR)',
  wr: 'Western Region (WR)',
  sr: 'Southern Region (SR)',
  er: 'Eastern Region (ER)',
  ner: 'North-Eastern Region (NER)',
  other: 'Other Territories'
};

export default function Navbar({
  states = [],
  selectedState,
  onSelectState,
  useWeatherApi,
  onToggleWeatherApi,
  health,
  metadata,
  theme,
  onToggleTheme,
  activeTab,
  onSelectTab,
  onOpenHealthModal,
  onOpenLedgerModal
}) {
  // Group states by region
  const groupedStates = React.useMemo(() => {
    const groups = {};
    states.forEach((st) => {
      const reg = st.region || 'other';
      if (!groups[reg]) groups[reg] = [];
      groups[reg].push(st);
    });
    // Sort each group
    Object.keys(groups).forEach(reg => {
      groups[reg].sort((a, b) => a.name.localeCompare(b.name));
    });
    return groups;
  }, [states]);

  const isHealthy = health?.status === 'healthy';

  const tabs = [
    { id: 'forecast', label: 'Demand Forecast' },
    { id: 'weather', label: 'Weather Impact' },
    { id: 'assistant', label: 'AI Assistant' },
  ];

  return (
    <header className="sticky top-0 z-40 bg-white/95 dark:bg-executive-cardDark/95 backdrop-blur-md border-b border-executive-borderLight dark:border-executive-borderDark transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-4">
          
          {/* Logo & Platform Name */}
          <div className="flex items-center flex-shrink-0">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-base sm:text-lg tracking-tight text-executive-textLight dark:text-executive-textDark">
                  Loadly
                </span>
                <span className="text-xs px-2 py-0.5 rounded-none font-semibold bg-executive-navy/10 dark:bg-executive-cyan/15 text-executive-navy dark:text-executive-cyan border border-executive-navy/20 dark:border-executive-cyan/30 hidden sm:inline-block">
                  Stage 1.5
                </span>
              </div>
              <p className="text-[11px] font-medium text-executive-mutedLight dark:text-executive-mutedDark hidden md:block">
                Intelligent Power Demand Forecasting & Grid Analytics
              </p>
            </div>
          </div>

          {/* Center Tabs for Desktop */}
          <nav className="hidden md:flex items-center h-9 p-1 bg-slate-100 dark:bg-slate-900/60 rounded-none border border-slate-200/80 dark:border-slate-800">
            {tabs.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => onSelectTab(tab.id)}
                  className={`flex items-center justify-center h-full px-3.5 text-xs font-semibold whitespace-nowrap rounded-none transition-all ${
                    isActive
                      ? 'bg-white dark:bg-executive-cardDark text-executive-navy dark:text-executive-cyan shadow-sm border border-slate-200 dark:border-slate-700'
                      : 'text-executive-mutedLight dark:text-executive-mutedDark hover:text-executive-textLight dark:hover:text-executive-textDark hover:bg-white/50 dark:hover:bg-slate-800/40'
                  }`}
                >
                  {tab.label}
                </button>
              );
            })}
          </nav>

          {/* Right Controls: State Selector, Weather Toggle, Health, Theme */}
          <div className="flex items-center gap-2.5">
            
            {/* State Selector */}
            <div className="relative">
              <div className="flex items-center h-9 px-3 bg-slate-100 dark:bg-slate-900/80 border border-executive-borderLight dark:border-executive-borderDark rounded-none text-xs font-semibold focus-within:ring-2 focus-within:ring-executive-navy/30 dark:focus-within:ring-executive-cyan/30">
                <select
                  value={selectedState}
                  onChange={(e) => onSelectState(e.target.value)}
                  className="h-full bg-transparent border-none text-executive-textLight dark:text-executive-textDark focus:outline-none cursor-pointer pr-1 text-xs font-semibold whitespace-nowrap"
                >
                  <option value="All-India" className="bg-white dark:bg-slate-900 text-executive-navy dark:text-executive-cyan font-bold">
                    All-India (National Total)
                  </option>
                  {Object.entries(groupedStates).map(([regionKey, stateItems]) => (
                    <optgroup key={regionKey} label={REGION_LABELS[regionKey] || regionKey.toUpperCase()}>
                      {stateItems.map((st) => (
                        <option key={st.name} value={st.name} className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100">
                          {st.name}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              </div>
            </div>

            {/* Weather API Toggle */}
            <button
              onClick={() => onToggleWeatherApi(!useWeatherApi)}
              title={useWeatherApi ? 'Using Live Open-Meteo Weather Forecast' : 'Using Climatology Historical Weather'}
              className={`hidden sm:flex items-center justify-center h-9 px-3 rounded-none text-xs font-medium border whitespace-nowrap transition-colors ${
                useWeatherApi
                  ? 'bg-amber-50 dark:bg-amber-950/40 border-amber-300 dark:border-amber-800/60 text-amber-800 dark:text-amber-300'
                  : 'bg-slate-100 dark:bg-slate-900 border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400'
              }`}
            >
              <span>{useWeatherApi ? 'Live Weather' : 'Climatology'}</span>
            </button>

            {/* Audit Ledger & Verification Button */}
            <button
              onClick={onOpenLedgerModal}
              title="Click to view Prediction Ledger & Verification Audit"
              className="flex items-center justify-center h-9 px-3 rounded-none text-xs font-medium border border-executive-navy/30 dark:border-executive-cyan/40 bg-executive-navy/5 dark:bg-executive-cyan/10 text-executive-navy dark:text-executive-cyan hover:bg-executive-navy/10 dark:hover:bg-executive-cyan/20 whitespace-nowrap transition-all"
            >
              <span>Ledger</span>
            </button>

            {/* Health Status Pill */}
            <button
              onClick={onOpenHealthModal}
              title="Click to view System Health & Architecture Specs"
              className={`flex items-center justify-center h-9 px-3 rounded-none text-xs font-medium border whitespace-nowrap transition-all ${
                isHealthy
                  ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-300 dark:border-emerald-800/60 text-emerald-800 dark:text-emerald-300 hover:bg-emerald-100 dark:hover:bg-emerald-900/50'
                  : 'bg-amber-50 dark:bg-amber-950/40 border-amber-300 dark:border-amber-800/60 text-amber-800 dark:text-amber-300 hover:bg-amber-100 dark:hover:bg-amber-900/50'
              }`}
            >
              <span>{isHealthy ? 'Model Ready' : 'Degraded'}</span>
            </button>

            {/* Dark/Light Theme Toggle */}
            <button
              onClick={onToggleTheme}
              title={theme === 'dark' ? 'Switch to Executive Daylight (Light Mode)' : 'Switch to Executive Midnight (Dark Mode)'}
              className="flex items-center justify-center h-9 w-9 rounded-none text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 bg-slate-100 dark:bg-slate-900 hover:bg-slate-200 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-800 transition-colors"
            >
              {theme === 'dark' ? (
                <Sun className="w-4 h-4 text-executive-gold" />
              ) : (
                <Moon className="w-4 h-4 text-executive-navy" />
              )}
            </button>

          </div>
        </div>

        {/* Mobile Navigation Tabs */}
        <div className="flex md:hidden items-center justify-around py-2 border-t border-slate-200/80 dark:border-slate-800">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onSelectTab(tab.id)}
                className={`flex items-center justify-center h-8 px-3 text-xs font-semibold rounded-none ${
                  isActive
                    ? 'bg-executive-navy dark:bg-executive-cyan text-white dark:text-slate-950'
                    : 'text-executive-mutedLight dark:text-executive-mutedDark'
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
}
