import requests
city = "Bengaluru"

url= "https://geocoding-api.open-meteo.com/v1/search"
params = {
            'name': f"{city}, India",
            'count': 1,
        }
response = requests.get(url, params)
data = response.json()
admin3 = data['results'][0]
print(admin3)


