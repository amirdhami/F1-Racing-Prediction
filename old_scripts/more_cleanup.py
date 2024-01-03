import pandas as pd

readfilename = "F1_racetime.csv"
df = pd.read_csv(readfilename)

# drop "time in seconds" column
df = df.drop(columns=['Time in Seconds'])

df['Race Laps'] = df['Race Laps'].fillna(0).astype(int)

df.loc[df['Adjusted Time'].isnull(), 'Adjusted Time'] = df['Time/Retired']

def standardize_time(time_str):
    if ':' in time_str: # find min:sec.ms format
        mins, sec = time_str.split(':')
        return float(mins)*60 + float(sec)
    # otherwise don't touch it
    return time_str

df['Qual Time'] = df['Qual Time'].apply(lambda x: standardize_time(x) if isinstance(x, str) and ':' in x else x)

writefilename = "F1_standardized_1.csv"
df.to_csv(writefilename, index=False)