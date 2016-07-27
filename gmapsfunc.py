import requests
import time

g_url = 'https://maps.googleapis.com/maps/api/geocode/json?'
g_key = 'AIzaSyA5TX9wDRR60A1wrv_pwRJyleBz3NZ25g0'


def geo_loc(city, country):
    address_1 = g_url + 'address=' + city
    address_2 = '&components=country:' + country + '&key='
    query_url = address_1 + address_2 + g_key
    # submit and get the response
    try:
        response = requests.get(query_url)
        print(response)
        entity = response.json()['results']
        print(entity)
        try:
            results = response.json()['results']
            print(results)
            # retrieve information about the match
            x_comp = results[0]['address_components']
            x_city_name = x_comp[0]['long_name'].upper()
            x_county = x_comp[1]['long_name'].upper()
            x_state_name = x_comp[2]['long_name'].upper()
            x_state_abbr = x_comp[2]['short_name'].upper()
            x_country_full = x_comp[3]['long_name'].upper()
            x_country_iso = x_comp[3]['short_name'].upper()
            # retrieve the lat/long
            x_geo = results[0]['geometry']['location']
            g_lat = x_geo['lat']
            g_long = x_geo['lng']
            return (x_city_name, x_county, x_state_name, x_state_abbr, x_country_full, x_country_iso,
                    g_lat, g_long)
        except IndexError:
            x_city_name = ''
            x_county = ''
            x_state_name = ''
            x_state_abbr = ''
            x_country_full = ''
            x_country_iso = ''
            g_lat = None
            g_long = None
            return (x_city_name, x_county, x_state_name, x_state_abbr, x_country_full, x_country_iso,
                    g_lat, g_long)
            pass
    except requests.exceptions.Timeout:
        time.sleep(5)
        geo_loc(city, country)
    except requests.exceptions.RequestException as e:
        print(e)
        x_city_name = ''
        x_county = ''
        x_state_name = ''
        x_state_abbr = ''
        x_country_full = ''
        x_country_iso = ''
        g_lat = None
        g_long = None
        return (x_city_name, x_county, x_state_name, x_state_abbr, x_country_full, x_country_iso,
                g_lat, g_long)
        pass

g_city = []
g_county = []
g_state = []
g_state_abbr = []
g_country = []
g_country_iso = []
g_lat = []
g_lng = []
r_city, r_county, r_state, r_state_a, r_country, r_iso, r_lat, r_lng = geo_loc('NEW YORK', 'US')
g_city.append(r_city)
g_county.append(r_county)
g_state.append(r_state)
g_state_abbr.append(r_state_a)
g_country.append(r_country)
g_country_iso.append(r_iso)
g_lat.append(r_lat)
g_lng.append(r_lng)
r_city, r_county, r_state, r_state_a, r_country, r_iso, r_lat, r_lng = geo_loc('CEDAR PARK', 'US')
g_city.append(r_city)
g_county.append(r_county)
g_state.append(r_state)
g_state_abbr.append(r_state_a)
g_country.append(r_country)
g_country_iso.append(r_iso)
g_lat.append(r_lat)
g_lng.append(r_lng)
r_city, r_county, r_state, r_state_a, r_country, r_iso, r_lat, r_lng = geo_loc('BOSTON', 'US')
g_city.append(r_city)
g_county.append(r_county)
g_state.append(r_state)
g_state_abbr.append(r_state_a)
g_country.append(r_country)
g_country_iso.append(r_iso)
g_lat.append(r_lat)
g_lng.append(r_lng)
r_city, r_county, r_state, r_state_a, r_country, r_iso, r_lat, r_lng = geo_loc('BOSTON', 'ZZ')
g_city.append(r_city)
g_county.append(r_county)
g_state.append(r_state)
g_state_abbr.append(r_state_a)
g_country.append(r_country)
g_country_iso.append(r_iso)
g_lat.append(r_lat)
g_lng.append(r_lng)

print(g_city, g_county, g_state, g_state_abbr, g_country, g_country_iso, g_lat, g_lng)