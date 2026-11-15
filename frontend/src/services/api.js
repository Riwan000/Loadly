/**
 * API service for communicating with the Electricity Demand Forecaster FastAPI backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

async function handleResponse(res) {
  if (!res.ok) {
    let errorDetail = 'API request failed';
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || errJson.message || JSON.stringify(errJson);
    } catch {
      errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    }
    throw new Error(errorDetail);
  }
  return res.json();
}

export const api = {
  // Config & Metadata
  async getStates() {
    const res = await fetch(`${API_BASE_URL}/api/config/states`);
    return handleResponse(res);
  },

  async getModelMetadata() {
    const res = await fetch(`${API_BASE_URL}/api/model/metadata`);
    return handleResponse(res);
  },

  // Forecast
  async getForecast({ state, horizonDays = 7, useWeatherApi = true }) {
    const res = await fetch(`${API_BASE_URL}/api/forecast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        state,
        horizon_days: horizonDays,
        use_weather_api: useWeatherApi
      })
    });
    return handleResponse(res);
  },

  // Weather Impact Analytics
  async getWeatherImpact({ state, mode = 'forecast', horizonDays = 14, useWeatherApi = true }) {
    const res = await fetch(`${API_BASE_URL}/api/analytics/weather-impact`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        state,
        mode,
        horizon_days: horizonDays,
        use_weather_api: useWeatherApi
      })
    });
    return handleResponse(res);
  },

  async getFeatureImportance({ topN = 15, weatherOnly = false } = {}) {
    const params = new URLSearchParams({
      top_n: topN.toString(),
      weather_only: weatherOnly.toString()
    });
    const res = await fetch(`${API_BASE_URL}/api/analytics/feature-importance?${params.toString()}`);
    return handleResponse(res);
  },

  async getSeasonalComparison(state) {
    const params = new URLSearchParams({ state });
    const res = await fetch(`${API_BASE_URL}/api/analytics/seasonal-comparison?${params.toString()}`);
    return handleResponse(res);
  },

  // AI Energy Assistant
  async sendChatMessage({ message, state, horizonDays = 7, chatHistory = [] }) {
    const res = await fetch(`${API_BASE_URL}/api/assistant/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        state,
        horizon_days: horizonDays,
        chat_history: chatHistory
      })
    });
    return handleResponse(res);
  },

  async getSuggestions(state) {
    const params = new URLSearchParams({ state: state || 'Kerala' });
    const res = await fetch(`${API_BASE_URL}/api/assistant/suggestions?${params.toString()}`);
    return handleResponse(res);
  },

  // Health Check
  async getHealth() {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    return handleResponse(res);
  },

  // Prediction Ledger & Verification
  async getLedgerRecords({ state, limit = 50, reconciledOnly = false } = {}) {
    const params = new URLSearchParams();
    if (state) params.append('state', state);
    params.append('limit', limit.toString());
    params.append('reconciled_only', reconciledOnly.toString());
    const res = await fetch(`${API_BASE_URL}/api/ledger/records?${params.toString()}`);
    return handleResponse(res);
  },

  async getLedgerMetrics({ state, daysWindow = 30 } = {}) {
    const params = new URLSearchParams();
    if (state) params.append('state', state);
    params.append('days_window', daysWindow.toString());
    const res = await fetch(`${API_BASE_URL}/api/ledger/metrics?${params.toString()}`);
    return handleResponse(res);
  },

  async ingestActuals(records) {
    const res = await fetch(`${API_BASE_URL}/api/ledger/actuals`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ records })
    });
    return handleResponse(res);
  },

  async triggerIngestion({ targetDate = null, catchupDays = 1 } = {}) {
    const res = await fetch(`${API_BASE_URL}/api/ingestion/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_date: targetDate,
        catchup_days: catchupDays
      })
    });
    return handleResponse(res);
  }
};
