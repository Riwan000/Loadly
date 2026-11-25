# ============================================================================
# WEATHER API FUNCTIONS
# ============================================================================
import logging
from datetime import date, timedelta

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Open-Meteo forecast request: `forecast_days` ahead of today plus `past_days` behind it
# (the API allows up to 92). Explicit start_date/end_date are avoided because their
# allowed range edge differs from forecast_days and a violation is a hard HTTP 400.
OPENMETEO_FORECAST_DAYS = 16
OPENMETEO_MAX_PAST_DAYS = 92
OPENMETEO_PAST_DAYS_SLACK = 1

from config import STATE_COORDINATES
from data_loading import load_historical_data


def _slice_daily_window(daily_data, window_start, window_end):
    """
    Keep the rows of an Open-Meteo `daily` block dated within [window_start, window_end],
    cut at the first row without temperatures. The API pads the last requested day with
    nulls (the 16th forecast day is often empty); those days are left to climatology.
    """
    keep = []
    for i, t in enumerate(daily_data["time"]):
        if not window_start <= pd.Timestamp(t).date() <= window_end:
            continue
        if daily_data["temperature_2m_max"][i] is None or daily_data["temperature_2m_min"][i] is None:
            break
        keep.append(i)
    return {key: [values[i] for i in keep] for key, values in daily_data.items()}


def get_weather_forecast_openmeteo(state_name, start_date, num_days):
    """
    Get weather forecast using Open-Meteo API (FREE, no API key required).

    Open-Meteo is a free weather API that provides up to 16 days of forecast
    data without requiring registration or API keys.

    Parameters:
        state_name (str): Name of the Indian state
        start_date (datetime.date): First day of the window; may be in the recent past
        num_days (int): Number of days wanted (callers fill any shortfall with climatology).

    Returns:
        pd.DataFrame: Weather data matching model's expected format, starting exactly
        at start_date (ValueError otherwise). Rows beyond today + 15 are
        not available, so fewer than num_days rows may come back
    """
    if state_name not in STATE_COORDINATES:
        raise ValueError(f"Coordinates not found for state: {state_name}")

    coords = STATE_COORDINATES[state_name]
    lat, lon = coords["lat"], coords["lon"]

    window_start = pd.Timestamp(start_date).date()
    window_end = window_start + timedelta(days=num_days - 1)
    # One day of slack: Open-Meteo anchors its window to the Asia/Kolkata date, which can be
    # a day ahead of the host's date (UTC hosts, 00:00-05:30 IST). Extra rows are sliced off.
    past_days = max(0, (date.today() - window_start).days) + OPENMETEO_PAST_DAYS_SLACK
    if past_days > OPENMETEO_MAX_PAST_DAYS:
        raise ValueError(f"start_date {window_start} is older than the Open-Meteo past range")

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        daily_params = "temperature_2m_max,temperature_2m_min,dewpoint_2m_mean,windspeed_10m_max,winddirection_10m_dominant,shortwave_radiation_sum,cloudcover_mean"

        from urllib.parse import urlencode
        base_params = {
            "latitude": lat,
            "longitude": lon,
            "past_days": past_days,
            "forecast_days": OPENMETEO_FORECAST_DAYS,
            "timezone": "Asia/Kolkata"
        }
        query_parts = [urlencode(base_params), f"daily={daily_params}"]
        full_url = f"{url}?{'&'.join(query_parts)}"

        response = requests.get(full_url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "daily" not in data:
            raise ValueError("Invalid API response: missing 'daily' key")

        daily_data = _slice_daily_window(data["daily"], window_start, window_end)

        temp_mean = [(max + min) / 2 for max, min in zip(
            daily_data["temperature_2m_max"],
            daily_data["temperature_2m_min"]
        )]

        forecast_df = pd.DataFrame({
            "date": pd.to_datetime(daily_data["time"]),
            "2m_temperature_max": daily_data["temperature_2m_max"],
            "2m_temperature_min": daily_data["temperature_2m_min"],
            "2m_temperature_mean": temp_mean,
        })

        if len(forecast_df) == 0 or forecast_df["date"].iloc[0].date() != window_start:
            raise ValueError(
                f"Open-Meteo returned dates starting {daily_data['time'][:1]} "
                f"instead of {window_start}")

        # Convert temperatures from Celsius to Kelvin
        forecast_df["2m_temperature_max"] = forecast_df["2m_temperature_max"] + 273.15
        forecast_df["2m_temperature_min"] = forecast_df["2m_temperature_min"] + 273.15
        forecast_df["2m_temperature_mean"] = forecast_df["2m_temperature_mean"] + 273.15

        if "dewpoint_2m_mean" in daily_data:
            forecast_df["2m_dewpoint_temperature_mean"] = [
                d + 273.15 if d is not None else None
                for d in daily_data["dewpoint_2m_mean"]
            ]
            forecast_df["2m_dewpoint_temperature_min"] = forecast_df["2m_dewpoint_temperature_mean"]
            forecast_df["2m_dewpoint_temperature_max"] = forecast_df["2m_dewpoint_temperature_mean"]
        else:
            forecast_df["2m_dewpoint_temperature_mean"] = forecast_df["2m_temperature_mean"] - 5
            forecast_df["2m_dewpoint_temperature_min"] = forecast_df["2m_dewpoint_temperature_mean"]
            forecast_df["2m_dewpoint_temperature_max"] = forecast_df["2m_dewpoint_temperature_mean"]

        if "windspeed_10m_max" in daily_data and "winddirection_10m_dominant" in daily_data:
            wind_speed = daily_data["windspeed_10m_max"]
            wind_dir_rad = np.radians(daily_data["winddirection_10m_dominant"])

            forecast_df["10m_u_component_of_wind_mean"] = [
                -speed * np.sin(rad) for speed, rad in zip(wind_speed, wind_dir_rad)
            ]
            forecast_df["10m_v_component_of_wind_mean"] = [
                -speed * np.cos(rad) for speed, rad in zip(wind_speed, wind_dir_rad)
            ]
            forecast_df["10m_u_component_of_wind_min"] = forecast_df["10m_u_component_of_wind_mean"]
            forecast_df["10m_u_component_of_wind_max"] = forecast_df["10m_u_component_of_wind_mean"]
            forecast_df["10m_v_component_of_wind_min"] = forecast_df["10m_v_component_of_wind_mean"]
            forecast_df["10m_v_component_of_wind_max"] = forecast_df["10m_v_component_of_wind_mean"]
        else:
            for col in ["10m_u_component_of_wind_min", "10m_u_component_of_wind_mean",
                        "10m_u_component_of_wind_max", "10m_v_component_of_wind_min",
                        "10m_v_component_of_wind_mean", "10m_v_component_of_wind_max"]:
                forecast_df[col] = 0.0

        if "shortwave_radiation_sum" in daily_data:
            forecast_df["surface_solar_radiation_downwards_mean"] = [
                (val / 86400) if val is not None else 0
                for val in daily_data["shortwave_radiation_sum"]
            ]
            forecast_df["surface_solar_radiation_downwards_min"] = 0.0
            forecast_df["surface_solar_radiation_downwards_max"] = forecast_df["surface_solar_radiation_downwards_mean"] * 2
        else:
            forecast_df["surface_solar_radiation_downwards_mean"] = 0.0
            forecast_df["surface_solar_radiation_downwards_min"] = 0.0
            forecast_df["surface_solar_radiation_downwards_max"] = 0.0

        if "cloudcover_mean" in daily_data:
            forecast_df["total_cloud_cover_mean"] = [
                val / 100.0 if val is not None else 0
                for val in daily_data["cloudcover_mean"]
            ]
            forecast_df["total_cloud_cover_min"] = 0.0
            forecast_df["total_cloud_cover_max"] = forecast_df["total_cloud_cover_mean"]
        else:
            forecast_df["total_cloud_cover_mean"] = 0.0
            forecast_df["total_cloud_cover_min"] = 0.0
            forecast_df["total_cloud_cover_max"] = 0.0

        # UTCI approximation using Steadman heat index (accounts for humidity + wind chill)
        # Full UTCI requires complex biometeorological model; this uses a simpler thermal index
        def steadman_heat_index(t_celsius, rh_percent, wind_ms):
            c1, c2, c3, c4, c5, c6, c7, c8, c9 = -42.379, 2.04901523, 10.14333127, -0.22475541, -0.00683783, -0.05481717, 0.00122874, 0.00085282, -0.00000199
            t_fahrenheit = (t_celsius * 9/5) + 32
            rh_squared = rh_percent * rh_percent
            hi = c1 + c2*t_fahrenheit + c3*rh_percent + c4*t_fahrenheit*rh_percent + c5*t_fahrenheit**2 + c6*rh_percent**2 + c7*t_fahrenheit**2*rh_percent + c8*t_fahrenheit*rh_percent**2 + c9*t_fahrenheit**2*rh_percent**2
            hi_celsius = (hi - 32) * 5/9
            wind_chill_factor = 1.0 - (0.3 * wind_ms / 10.0)
            return hi_celsius * wind_chill_factor

        forecast_df["utci_mean"] = forecast_df.apply(lambda row: steadman_heat_index(row["2m_temperature_mean"], row.get("relative_humidity_2m_mean", 50), row.get("wind_speed_10m_mean", 3)), axis=1)
        forecast_df["utci_min"] = forecast_df.apply(lambda row: steadman_heat_index(row["2m_temperature_min"], row.get("relative_humidity_2m_mean", 50), row.get("wind_speed_10m_min", 1)), axis=1)
        forecast_df["utci_max"] = forecast_df.apply(lambda row: steadman_heat_index(row["2m_temperature_max"], row.get("relative_humidity_2m_mean", 50), row.get("wind_speed_10m_max", 5)), axis=1)

        return forecast_df

    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")
    except KeyError as e:
        raise Exception(f"Missing data in API response: {str(e)}")
    except Exception as e:
        raise Exception(f"Error fetching weather data: {str(e)}")


def get_weather_forecast_climatology(state_name, start_date, num_days):
    """
    Get weather forecast using historical climatology (average of same dates in past years).

    This is a fallback method when weather API is unavailable.

    Parameters:
        state_name (str): Name of the Indian state
        start_date (datetime.date): Start date for forecast
        num_days (int): Number of days to forecast

    Returns:
        pd.DataFrame: Weather data with historical averages
    """
    historical_df = load_historical_data(state_name)
    if historical_df is None:
        logger.warning(f"No historical weather found for {state_name} — using reference defaults.")
        historical_df = load_historical_data('Kerala')

    if historical_df is None:
        # Generate minimal synthetic fallback
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        historical_df = pd.DataFrame({
            'date': dates,
            '2m_temperature_min': 295.0,
            '2m_temperature_mean': 300.0,
            '2m_temperature_max': 305.0,
            '2m_dewpoint_temperature_min': 290.0,
            '2m_dewpoint_temperature_mean': 295.0,
            '2m_dewpoint_temperature_max': 300.0,
            '10m_u_component_of_wind_mean': 0.0,
            '10m_v_component_of_wind_mean': 0.0,
            'surface_solar_radiation_downwards_mean': 150.0,
            'total_cloud_cover_mean': 50.0,
            'utci_mean': 300.0
        })

    forecast_dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    forecast_data = []

    for date in forecast_dates:
        month_day = (date.month, date.day)
        historical_same_date = historical_df[
            (historical_df['date'].dt.month == month_day[0]) &
            (historical_df['date'].dt.day == month_day[1])
        ]

        if len(historical_same_date) > 0:
            avg_weather = {
                'date': date,
                '2m_temperature_min': historical_same_date['2m_temperature_min'].mean(),
                '2m_temperature_mean': historical_same_date['2m_temperature_mean'].mean(),
                '2m_temperature_max': historical_same_date['2m_temperature_max'].mean(),
                '2m_dewpoint_temperature_min': historical_same_date['2m_dewpoint_temperature_min'].mean(),
                '2m_dewpoint_temperature_mean': historical_same_date['2m_dewpoint_temperature_mean'].mean(),
                '2m_dewpoint_temperature_max': historical_same_date['2m_dewpoint_temperature_max'].mean(),
                '10m_u_component_of_wind_mean': historical_same_date['10m_u_component_of_wind_mean'].mean(),
                '10m_v_component_of_wind_mean': historical_same_date['10m_v_component_of_wind_mean'].mean(),
                'surface_solar_radiation_downwards_mean': historical_same_date['surface_solar_radiation_downwards_mean'].mean(),
                'total_cloud_cover_mean': historical_same_date['total_cloud_cover_mean'].mean(),
                'utci_mean': historical_same_date['utci_mean'].mean(),
            }
            avg_weather['10m_u_component_of_wind_min'] = avg_weather['10m_u_component_of_wind_mean']
            avg_weather['10m_u_component_of_wind_max'] = avg_weather['10m_u_component_of_wind_mean']
            avg_weather['10m_v_component_of_wind_min'] = avg_weather['10m_v_component_of_wind_mean']
            avg_weather['10m_v_component_of_wind_max'] = avg_weather['10m_v_component_of_wind_mean']
            avg_weather['surface_solar_radiation_downwards_min'] = 0.0
            avg_weather['surface_solar_radiation_downwards_max'] = avg_weather['surface_solar_radiation_downwards_mean'] * 2
            avg_weather['total_cloud_cover_min'] = 0.0
            avg_weather['total_cloud_cover_max'] = avg_weather['total_cloud_cover_mean']
            avg_weather['utci_min'] = avg_weather['2m_temperature_min']
            avg_weather['utci_max'] = avg_weather['2m_temperature_max']
            forecast_data.append(avg_weather)
        else:
            monthly_avg = historical_df[historical_df['date'].dt.month == date.month].mean()
            avg_weather = {
                'date': date,
                '2m_temperature_min': monthly_avg.get('2m_temperature_min', 295.0),
                '2m_temperature_mean': monthly_avg.get('2m_temperature_mean', 300.0),
                '2m_temperature_max': monthly_avg.get('2m_temperature_max', 305.0),
                '2m_dewpoint_temperature_min': monthly_avg.get('2m_dewpoint_temperature_min', 290.0),
                '2m_dewpoint_temperature_mean': monthly_avg.get('2m_dewpoint_temperature_mean', 295.0),
                '2m_dewpoint_temperature_max': monthly_avg.get('2m_dewpoint_temperature_max', 300.0),
                '10m_u_component_of_wind_mean': monthly_avg.get('10m_u_component_of_wind_mean', 0.0),
                '10m_v_component_of_wind_mean': monthly_avg.get('10m_v_component_of_wind_mean', 0.0),
                'surface_solar_radiation_downwards_mean': monthly_avg.get('surface_solar_radiation_downwards_mean', 0.0),
                'total_cloud_cover_mean': monthly_avg.get('total_cloud_cover_mean', 0.0),
                'utci_mean': monthly_avg.get('utci_mean', 300.0),
            }
            for key in ['10m_u_component_of_wind', '10m_v_component_of_wind']:
                avg_weather[f'{key}_min'] = avg_weather[f'{key}_mean']
                avg_weather[f'{key}_max'] = avg_weather[f'{key}_mean']
            avg_weather['surface_solar_radiation_downwards_min'] = 0.0
            avg_weather['surface_solar_radiation_downwards_max'] = avg_weather['surface_solar_radiation_downwards_mean'] * 2
            avg_weather['total_cloud_cover_min'] = 0.0
            avg_weather['total_cloud_cover_max'] = avg_weather['total_cloud_cover_mean']
            avg_weather['utci_min'] = avg_weather['2m_temperature_min']
            avg_weather['utci_max'] = avg_weather['2m_temperature_max']
            forecast_data.append(avg_weather)

    return pd.DataFrame(forecast_data)


def get_weather_forecast(state_name, start_date, num_days, use_api=True):
    """
    Main function to get weather forecast with automatic fallback and blending.

    Days Open-Meteo cannot serve (beyond today + 15) are filled with climatology
    estimates; if the API fails or misaligns its dates, everything is climatology.
    start_date may be in the recent past (used to bridge gaps in observed demand).

    Parameters:
        state_name (str): Name of the Indian state
        start_date (datetime.date): Start date for forecast
        num_days (int): Number of days to forecast
        use_api (bool): If True, try API first. If False, use climatology directly.

    Returns:
        pd.DataFrame: Weather forecast data
    """
    if not use_api:
        return get_weather_forecast_climatology(state_name, start_date, num_days)
    try:
        df_api = get_weather_forecast_openmeteo(state_name, start_date, num_days)
    except Exception as e:
        logger.warning(f"Weather API failed: {str(e)}. Using historical averages...")
        return get_weather_forecast_climatology(state_name, start_date, num_days)

    api_len = len(df_api)
    if api_len >= num_days:
        return df_api
    next_date = (pd.Timestamp(start_date) + pd.Timedelta(days=api_len)).date()
    df_clim = get_weather_forecast_climatology(state_name, next_date, num_days - api_len)
    return pd.concat([df_api, df_clim], ignore_index=True)
