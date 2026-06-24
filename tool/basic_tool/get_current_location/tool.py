import requests


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