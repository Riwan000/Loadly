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

// ---------------------------------------------------------------------------
// Operator admin key
// The key is entered by the operator at runtime and kept in sessionStorage only
// (cleared when the tab closes). It must never be bundled or read from VITE_ env.
// ---------------------------------------------------------------------------
const ADMIN_KEY_STORAGE_KEY = 'loadly.adminKey';
const ADMIN_KEY_HEADER = 'X-Admin-Key';

export const ADMIN_ERROR = Object.freeze({
  MISSING: 'missing',
  UNAUTHORIZED: 'unauthorized',
  DISABLED: 'disabled'
});

export class AdminAuthError extends Error {
  constructor(message, code) {
    super(message);
    this.name = 'AdminAuthError';
    this.code = code;
  }
}

// In-memory fallback for when sessionStorage is blocked; lost on page reload.
let memoryAdminKey = '';

export const adminKeyStore = {
  get() {
    try {
      return window.sessionStorage.getItem(ADMIN_KEY_STORAGE_KEY) || memoryAdminKey;
    } catch {
      return memoryAdminKey;
    }
  },
  set(key) {
    memoryAdminKey = '';
    try {
      window.sessionStorage.setItem(ADMIN_KEY_STORAGE_KEY, key);
    } catch {
      memoryAdminKey = key;
    }
  },
  clear() {
    memoryAdminKey = '';
    try {
      window.sessionStorage.removeItem(ADMIN_KEY_STORAGE_KEY);
    } catch {
      // Storage unavailable: the in-memory copy above is already cleared.
    }
  }
};

async function adminFetch(path, { headers = {}, ...options } = {}) {
  const key = adminKeyStore.get();
  if (!key) {
    throw new AdminAuthError('An operator admin key is required for this action.', ADMIN_ERROR.MISSING);
  }
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: { ...headers, [ADMIN_KEY_HEADER]: key }
  });
  if (res.status === 401) {
    adminKeyStore.clear();
    throw new AdminAuthError('Admin key was rejected. Please re-enter it.', ADMIN_ERROR.UNAUTHORIZED);
  }
  if (res.status === 503) {
    throw new AdminAuthError(
      'Admin actions are disabled on this server (LOADLY_ADMIN_KEY is not configured).',
      ADMIN_ERROR.DISABLED
    );
  }
  return handleResponse(res);
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

  // Automated data update status (public, read-only)
  async getDataUpdateStatus() {
    const res = await fetch(`${API_BASE_URL}/api/data/update-status`);
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
    return adminFetch('/api/ledger/actuals', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ records })
    });
  },

  async triggerIngestion({ targetDate = null, catchupDays = 1 } = {}) {
    return adminFetch('/api/ingestion/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_date: targetDate,
        catchup_days: catchupDays
      })
    });
  },

  // MLOps Drift Detection & Adaptive Retraining
  async getDriftStatus({ state = null, windowDays = 14 } = {}) {
    const params = new URLSearchParams();
    if (state) params.append('state', state);
    params.append('window_days', windowDays.toString());
    const res = await fetch(`${API_BASE_URL}/api/mlops/drift-status?${params.toString()}`);
    return handleResponse(res);
  },

  async triggerRetrain({ triggerReason = 'manual_calibration', force = false } = {}) {
    return adminFetch('/api/mlops/retrain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        trigger_reason: triggerReason,
        force
      })
    });
  },

  async triggerRollback() {
    return adminFetch('/api/mlops/rollback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
  },

  async getRetrainingHistory({ limit = 15 } = {}) {
    const params = new URLSearchParams({ limit: limit.toString() });
    const res = await fetch(`${API_BASE_URL}/api/mlops/history?${params.toString()}`);
    return handleResponse(res);
  }
};
