import pandas as pd


# place the country mapping data into a dataframe
country_map = pd.read_csv('C:/Problem3_Data/CountryMap.csv', sep='|', header=0,
                          index_col=0, engine='python')
# place the geo_data into a dataframe
geo_data = pd.read_csv('C:/Problem3_Data/GeoMerge.csv', sep='|', header=0,
                       index_col=0, engine='python', skipinitialspace=True)
# drop the count field as it is not needed.
geo_input = geo_data.drop('count', axis=1)
# merge the reference table data
final_output = pd.merge(geo_input, country_map, how='left', on='CountryCode')
# reorder the fields
final_output = final_output[['CityName', 'CountryCode', 'CountryName',
                             'g_city', 'g_county', 'g_state', 'g_state_abbr',
                             'g_country', 'g_country_code', 'lat', 'lng']]


# apply a match success evaluation field
def match_check(row):
    """Checks to see if the geoloc from Google's api matches the raw input"""
    if row['CountryCode'] == row['g_country_code']:
        match = 1
    else:
        match = 0
    return match

# create the match success check field and apply it to the data frame.
final_output['Match_Success'] = final_output.apply(match_check, axis=1)
# rename the fields
final_output.columns = ['Input_City', 'Input_CountryCode', 'Output_CountryName',
                        'g_CityName', 'g_County', 'g_State', 'g_State_Abbreviation',
                        'g_CountryName', 'g_CountryCode', 'g_latitude', 'g_longitude',
                        'g_Match_Success']
# verification of correct formatting
print(final_output[:10])
# save the output file
final_output.to_csv('C:/Problem3_Data/FinalOutput.csv', sep='|', header=0, index=False,
                    encoding='utf-8')

