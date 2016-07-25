import pandas as pd
import re
# Set the input path
input_path = 'C:/Problem3_Data'
country_input = input_path + '/Problem 3 Input Data - Country Map.txt'
raw_input = input_path + '/Problem 3 Input Data.txt'
cleaned_data = input_path + '/cleansed.txt'
country_map = pd.read_table(country_input, sep='|', header=0, engine='python')


# Function for checking and verifying the change to a field
def filter_action(frame, col_name, target_val):
    print(frame.loc[frame[col_name] == target_val])


# Verification of formatting issues with a known pipe-existing field (Tanzania)
filter_action(country_map, 'Country Code', 'TZ')
# The input data frame contains pipes '|' within the field 'Country Name'.  Replace them with ','
country_map['Country Name'] = country_map['Country Name'].str.replace('|', ',')
# Verification of changing the entry
filter_action(country_map, 'Country Code', 'TZ')

# The data source contains improperly formatted unicode characters.
# Example: row 97523: 'BIENK#|98|#WKA'|'PL' which is not a unicode definition for what it should
# be (U+00F3).  There are 23 total times this appears in this data set.
raw_input_data = open(raw_input)
new_file = open(cleaned_data, 'w')
for line in raw_input_data:
    line = re.sub(r'#\|98\|#', 'O', line.rstrip())
    new_file.write(line + '\n')
new_file.close()
# Now open the cleansed data set in a data frame.

input_data = pd.read_table(cleaned_data, sep="|", header=0, engine='python')

print(input_data[:10])