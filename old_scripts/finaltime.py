import pandas as pd

readfilename = "F1_qualtime.csv"
df = pd.read_csv(readfilename)

def race_time_to_seconds(time_str):
    if pd.isnull(time_str) or time_str in ["DNF", "DNS", "NC"] or 'lap' in time_str:
        return None  # Return None for non-time strings
        
    # Handle relative time format "+21.969s"
    if '+' in time_str:
        try:
            return float(time_str.strip('+s'))
        except ValueError:
            return None
    
    # Handle first place time format "1:34:50.616"
    if ':' in time_str:
        time_parts = time_str.split(":")
        if len(time_parts) == 3:  # Format is hours:minutes:seconds
            hrs, mins, sec = [float(part) for part in time_parts]
            return hrs * 3600 + mins * 60 + sec
        elif len(time_parts) == 2:  # Format is minutes:seconds
            mins, sec = [float(part) for part in time_parts]
            return mins * 60 + sec

    return None

df['Time in Seconds'] = df['Time/Retired'].apply(race_time_to_seconds)

# Filter out rows where 'Race Pos' is "1" (string) and create a dictionary to map each race to the winner's time
winner_times = df[df['Race Pos'] == "1"].set_index(['Year', 'Race Name'])['Time in Seconds'].to_dict()

# calc avg lap times 
avg_lap_times = df[df['Race Pos'] == "1"].set_index(['Year', 'Race Name'])
avg_lap_times['Avg Lap Time'] = avg_lap_times['Time in Seconds'] / avg_lap_times['Race Laps']
avg_lap_times_dict = avg_lap_times['Avg Lap Time'].to_dict()


def time_to_feature(row):
    if pd.isnull(row['Time/Retired']) or row['Time/Retired'] in ["DNF", "DNS", "NC"]:
        return row['Time/Retired']
    
    year_race_key = (row['Year'], row['Race Name'])
    
    # Cases where racer is multiple laps behind
    if 'lap' in row['Time/Retired']:
        laps_behind = int(row['Time/Retired'].split('+')[1].split()[0])
        avg_lap_time = avg_lap_times_dict.get(year_race_key)
        if avg_lap_time is not None and year_race_key in winner_times:
            winner_time = winner_times[year_race_key]
            return winner_time + avg_lap_time * laps_behind
        
    # Cases where racer is multiple seconds behind
    elif '+' in row['Time/Retired']:
        seconds_behind = float(row['Time/Retired'].split('+')[1].strip('s'))
        if year_race_key in winner_times:
            winner_time = winner_times[year_race_key]
            return winner_time + seconds_behind
    
    return race_time_to_seconds(row['Time/Retired'])
        
df['Adjusted Time'] = df.apply(time_to_feature, axis=1)

writefilename = "F1_racetime.csv"
df.to_csv(writefilename, index=False)