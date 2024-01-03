import pandas as pd

def time_str_to_seconds(time_str):
    if pd.isnull(time_str) or time_str == "DNQ":
        return None
    
    try:
        minutes, seconds = time_str.split(":")
        return int(minutes) * 60 + float(seconds)
    
    except ValueError:
        return None
    
readfilename = "F1_data.csv"
df = pd.read_csv(readfilename)

for q in ['Q1', 'Q2', 'Q3']:
    df[q + '_seconds'] = df[q].apply(time_str_to_seconds)
    
df['Avg Qual Time'] = df[['Q1_seconds', 'Q2_seconds', 'Q3_seconds']].mean(axis=1, skipna=True)
        
df.loc[df['Year'] >= 2006, 'Qual Time'] = df['Avg Qual Time']

df.drop(['Q1_seconds', 'Q2_seconds', 'Q3_seconds', 'Avg Qual Time'], axis=1, inplace=True)

writefilename = "F1_new.csv"
df.to_csv(writefilename, index=False)