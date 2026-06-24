import pandas as pd
import requests


WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Freezing drizzle (light)",
    57: "Freezing drizzle (dense)",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Freezing rain (light)",
    67: "Freezing rain (heavy)",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm (slight/moderate)",
    96: "Thunderstorm with hail (slight)",
    99: "Thunderstorm with hail (heavy)",
}

def get_hourly_forecast(current_location, latitude, longitude, date: str) -> str:
    if not date:
        return "Date parameter is required"
    if current_location:
        latitude, longitude, city, country = get_current_location(True)
        noti = f"Get weather forecast for current location: {city}, {country}:\n"
    elif latitude and longitude:
        noti = f"Get weather forecast for specified coordinates: Latitude {latitude}, Longitude {longitude}:\n"
    else:
        latitude, longitude, city, country = get_current_location(True)
        noti = f"Longitude or latitude information are missing\nReturn current location: {city}, {country}:\n"

    forecast_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m,weathercode,cloudcover"
        f"&forecast_days=7&timezone=auto"
    )
    forecast_response = requests.get(forecast_url)
    if forecast_response.status_code != 200:
        return f"Forecast request failed: {forecast_response.status_code}"
    data = forecast_response.json()

    df = pd.DataFrame({
        "Time": data["hourly"]["time"],
        "temperature (°C)": data["hourly"]["temperature_2m"],
        "Humidity (%)": data["hourly"]["relative_humidity_2m"],
        "Windspeed (m/s)": data["hourly"]["windspeed_10m"],
        "weathercode": data["hourly"]["weathercode"],
        "Cloudcover (%)": data["hourly"]["cloudcover"]
    })
    df["Time"] = pd.to_datetime(df["Time"]).dt.strftime("%H:%M:%S")
    df["date"] = pd.to_datetime(data["hourly"]["time"]).date.astype(str)

    df["Weather"] = df["weathercode"].map(WEATHER_CODES).fillna("Unknown")
    df = df.drop(columns=["weathercode"])
    
    if date in set(df["date"]):
        result_df = df[df["date"] == date].drop(columns=["date"])
        return f"{noti}Weather forecast in {date}:\n{result_df.to_csv(index=False)}"
    else:
        return f"{noti}Date {date} not available in forecast range"


# Helper ======================================================================================

def get_current_location(return_value = False):
    response = requests.get("https://ipinfo.io/json")
    if response.status_code == 200:
        data = response.json()
        lat, lon = map(float, data["loc"].split(","))
        if return_value:
            return lat, lon, data.get('city'), data.get('country')
        return f"Current location:\nLatitude: {lat}, Longitude: {lon}, city: {data.get('city')}, country: {data.get('country')}"
    else:
        return "Failed to get location"