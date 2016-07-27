import pandas as pd
import requests


city = 'CEADR PARK'
country = 'US'

g_lat = []
g_lng = []

g_url = 'https://maps.googleapis.com/maps/api/geocode/json?'
g_key = 'AIzaSyA5TX9wDRR60A1wrv_pwRJyleBz3NZ25g0'
g_add = 'address=' + city
g_add2 = '&components=country:' + country
g_add3 = '&key='
query_url = g_url + g_add + g_add2 + g_add3 + g_key

response = requests.get(query_url)
results = response.json()['results']
x_comp = results[0]['address_components']

x_city_name = x_comp[0]['long_name'].upper()
x_county = x_comp[1]['long_name'].upper()
x_state_name = x_comp[2]['long_name'].upper()
x_state_abbr = x_comp[2]['short_name'].upper()
x_country_full = x_comp[3]['long_name'].upper()
x_country_iso = x_comp[3]['short_name'].upper()

print(x_comp)
x_geo = results[0]['geometry']['location']
g_lat.append(x_geo['lat'])
g_lng.append(x_geo['lng'])

print(g_lat, g_lng)
print(x_city_name, x_county, x_state_name, x_state_abbr, x_country_full, x_country_iso)
