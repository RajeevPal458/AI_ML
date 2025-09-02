import numpy as np
import pandas as pd

# pandas practive of reading and writing data files csv, excel, json
# read csv file
df = pd.read_csv('../testFiles/weather.csv')
print("DataFrame from CSV:")
print(df)
print()
# write to csv file
df.to_csv('../testFiles/weather_output.csv', index=False)
print("DataFrame written to CSV:")
print(df)
print()

# write to csv file
df.to_csv('../testFiles/weather_output_partial.csv', index=False, header=False,sep='\t', mode='a', encoding='utf-8',na_rep = 'nothing',columns=['MinTemp', 'MaxTemp', 'Rainfall'])
print("DataFrame written to CSV:")
print(df)
print()




# read excel file
df = pd.read_excel('../testFiles/ChallengeBakes.xlsx')
print("DataFrame from Excel:")
print(df)
print()
# write to excel file
df.to_excel('../testFiles/ChallengeBakes_output.xlsx', index=False)
print("DataFrame written to Excel:")
print(df)
print()
# read json file
df = pd.read_json('../testFiles/weather.json')
print("DataFrame from JSON:")
print(df)
print()

import pandas as pd

# Sample nested JSON data
data = [
    {
        'date': '2025-05-15',
        'location': 'New York',
        'weather': [
            {'temp': 20, 'condition': 'Sunny'},
            {'temp': 22, 'condition': 'Partly Cloudy'}
        ]
    },
    {
        'date': '2025-05-16',
        'location': 'Los Angeles',
        'weather': [
            {'temp': 25, 'condition': 'Sunny'},
            {'temp': 27, 'condition': 'Hot'}
        ]
    }
]

# Normalize the nested 'weather' records
df_normalized = pd.json_normalize(data, record_path='weather', meta=['date', 'location'])

print("Normalized DataFrame from JSON:")
print(df_normalized)
print()

# write to json file
df.to_json('../testFiles/weather_output.json', orient='records')
print("DataFrame written to JSON:")
print(df)
print()



