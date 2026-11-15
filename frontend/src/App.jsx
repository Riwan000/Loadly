import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import ForecastTab from './components/ForecastTab';
import WeatherImpactTab from './components/WeatherImpactTab';
import AssistantTab from './components/AssistantTab';
import SystemHealthModal from './components/SystemHealthModal';
import LedgerModal from './components/LedgerModal';
import ElectricityTowerBg from './components/ElectricityTowerBg';
import { api } from './services/api';
import { Zap, ShieldCheck } from 'lucide-react';

const FALLBACK_STATES = [
  { name: 'Maharashtra', region: 'wr', lat: 19.7515, lon: 75.7139 },
  { name: 'Gujarat', region: 'wr', lat: 22.2587, lon: 71.1924 },
  { name: 'Tamil Nadu', region: 'sr', lat: 11.1271, lon: 78.6569 },
  { name: 'Karnataka', region: 'sr', lat: 15.3173, lon: 75.7139 },
  { name: 'Kerala', region: 'sr', lat: 10.8505, lon: 76.2711 },
  { name: 'Uttar Pradesh', region: 'nr', lat: 26.8467, lon: 80.9462 },
  { name: 'Punjab', region: 'nr', lat: 31.1471, lon: 75.3412 },
  { name: 'Rajasthan', region: 'nr', lat: 27.0238, lon: 74.2179 },
  { name: 'NCT of Delhi', region: 'nr', lat: 28.7041, lon: 77.1025 },
  { name: 'West Bengal', region: 'er', lat: 22.9868, lon: 87.8550 },
  { name: 'Bihar', region: 'er', lat: 25.0961, lon: 85.3131 },
  { name: 'Odisha', region: 'er', lat: 20.9517, lon: 85.0985 },
  { name: 'Assam', region: 'ner', lat: 26.2006, lon: 92.9376 }
];

export default function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'dark';
  });
  const [states, setStates] = useState(FALLBACK_STATES);
  const [selectedState, setSelectedState] = useState('Maharashtra');
  const [horizonDays, setHorizonDays] = useState(7);
  const [useWeatherApi, setUseWeatherApi] = useState(true);
  const [activeTab, setActiveTab] = useState('forecast'); // 'forecast' | 'weather' | 'assistant'
  
  const [health, setHealth] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [isHealthModalOpen, setIsHealthModalOpen] = useState(false);
  const [isLedgerModalOpen, setIsLedgerModalOpen] = useState(false);

  const [forecastData, setForecastData] = useState(null);
  const [forecastLoading, setForecastLoading] = useState(true);
  const [forecastError, setForecastError] = useState(null);

  // Apply theme class to document element
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  // Initial load: states, health, metadata
  useEffect(() => {
    api.getStates()
      .then(res => {
        if (res?.states && res.states.length > 0) {
          setStates(res.states);
        }
      })
      .catch(err => console.warn('Using fallback state list:', err));

    api.getHealth()
      .then(res => setHealth(res))
      .catch(err => console.warn('Could not fetch initial health:', err));

    api.getModelMetadata()
      .then(res => setMetadata(res))
      .catch(err => console.warn('Could not fetch model metadata:', err));
  }, []);

  // Fetch forecast data when state, horizon, or weather source changes
  useEffect(() => {
    let isMounted = true;
    setForecastLoading(true);
    setForecastError(null);

    api.getForecast({
      state: selectedState,
      horizonDays,
      useWeatherApi
    })
      .then(data => {
        if (!isMounted) return;
        setForecastData(data);
        setForecastLoading(false);
      })
      .catch(err => {
        if (!isMounted) return;
        setForecastError(err.message || 'Error generating forecast');
        setForecastLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [selectedState, horizonDays, useWeatherApi]);

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors relative overflow-x-hidden">
      
      {/* Background Electricity Transmission Tower in Left Margin */}
      <ElectricityTowerBg />

      {/* Navigation & Controls */}
      <Navbar
        states={states}
        selectedState={selectedState}
        onSelectState={setSelectedState}
        useWeatherApi={useWeatherApi}
        onToggleWeatherApi={setUseWeatherApi}
        health={health}
        metadata={metadata}
        theme={theme}
        onToggleTheme={toggleTheme}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        onOpenHealthModal={() => setIsHealthModalOpen(true)}
        onOpenLedgerModal={() => setIsLedgerModalOpen(true)}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 relative z-10">
        {activeTab === 'forecast' && (
          <ForecastTab
            forecastData={forecastData}
            loading={forecastLoading}
            error={forecastError}
            state={selectedState}
            horizonDays={horizonDays}
            onChangeHorizon={setHorizonDays}
            theme={theme}
          />
        )}

        {activeTab === 'weather' && (
          <WeatherImpactTab
            state={selectedState}
            horizonDays={horizonDays}
            useWeatherApi={useWeatherApi}
            theme={theme}
          />
        )}

        {activeTab === 'assistant' && (
          <AssistantTab
            state={selectedState}
            horizonDays={horizonDays}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-executive-borderLight dark:border-executive-borderDark py-6 bg-white/50 dark:bg-executive-cardDark/50 text-xs text-executive-mutedLight dark:text-executive-mutedDark transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-executive-navy dark:text-executive-cyan" />
            <span className="font-semibold text-executive-textLight dark:text-executive-textDark">
              Loadly
            </span>
            <span>— Stage 1.5: Forecast + Explain</span>
          </div>
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setIsLedgerModalOpen(true)} 
              className="hover:underline flex items-center gap-1 text-executive-navy dark:text-executive-cyan font-semibold"
            >
              <Zap className="w-3.5 h-3.5" /> Verification Ledger
            </button>
            <span>•</span>
            <button 
              onClick={() => setIsHealthModalOpen(true)} 
              className="hover:underline flex items-center gap-1"
            >
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" /> Model Accuracy & Status
            </button>
            <span>•</span>
            <span>FastAPI + XGBoost + Open-Meteo</span>
          </div>
        </div>
      </footer>

      {/* System Health & Architecture Modal */}
      <SystemHealthModal
        isOpen={isHealthModalOpen}
        onClose={() => setIsHealthModalOpen(false)}
        health={health}
        metadata={metadata}
      />

      {/* Prediction Ledger & Verification Audit Modal */}
      <LedgerModal
        isOpen={isLedgerModalOpen}
        onClose={() => setIsLedgerModalOpen(false)}
        selectedState={selectedState}
      />

    </div>
  );
}
