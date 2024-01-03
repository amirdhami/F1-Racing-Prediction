import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

pd.set_option('display.max_columns', None)

results_df = pd.read_csv('F1-raw-data/results.csv')

# merge constructor name
constructors_df = pd.read_csv('F1-raw-data/constructors.csv')
merge1_df = pd.merge(results_df, constructors_df[['constructorId', 'constructorRef']], on='constructorId')
merge1_df.rename(columns={'constructorRef': 'constructorName'}, inplace=True)

# place constructorName right after constructorId
column_order = merge1_df.columns.tolist()
constructor_index = column_order.index('constructorId')
column_order.insert(constructor_index + 1, column_order.pop(column_order.index('constructorName')))
merge1_df = merge1_df[column_order]

# merge race circuit, year, and round
races_df = pd.read_csv('F1-raw-data/races.csv')
merge2_pt1_df = pd.merge(merge1_df, races_df[['raceId', 'year', 'round', 'circuitId', 'date']], on='raceId')

# now grab circuit based on circuitId
circuits_df = pd.read_csv('F1-raw-data/circuits.csv')
merge2_df = pd.merge(merge2_pt1_df, circuits_df[['circuitId', 'circuitRef']], on='circuitId')
merge2_df.rename(columns={'circuitRef': 'circuitName'}, inplace=True)

# place all info after raceId, drop circuitId
column_order = merge2_df.columns.tolist()
constructor_index = column_order.index('raceId')
column_order.insert(constructor_index + 1, column_order.pop(column_order.index('circuitId')))
column_order.insert(constructor_index + 2, column_order.pop(column_order.index('circuitName')))
column_order.insert(constructor_index+3, column_order.pop(column_order.index('year')))
column_order.insert(constructor_index+4, column_order.pop(column_order.index('date')))
column_order.insert(constructor_index+5, column_order.pop(column_order.index('round')))
merge2_df = merge2_df[column_order]

# merge driver name
drivers_df = pd.read_csv('F1-raw-data/drivers.csv')
merge3_df = pd.merge(merge2_df, drivers_df[['driverId', 'driverRef']], on='driverId')
merge3_df.rename(columns={'driverRef': 'driverName'}, inplace=True)

# place driverName right after driverId
column_order = merge3_df.columns.tolist()
constructor_index = column_order.index('driverId')
column_order.insert(constructor_index + 1, column_order.pop(column_order.index('driverName')))
merge3_df = merge3_df[column_order]

# just some rename work for clarity
merge3_df.rename(columns={'date':'raceDate', 'round':'raceRound', 'number':'carNumber', 'grid':'startPos'}, inplace=True)
merge3_df.rename(columns={'position':'finishPos', 'positionOrder':'finalRank', 'rank':'fastestLapPos'}, inplace=True)

# add status from statusId
status_df = pd.read_csv('F1-raw-data/status.csv')
merge4_df = pd.merge(merge3_df, status_df[['statusId', 'status']], on='statusId')
merge4_df.rename(columns={'status': 'finishStatus'}, inplace=True)
column_order = merge4_df.columns.tolist()
constructor_index = column_order.index('statusId')
column_order.insert(constructor_index + 1, column_order.pop(column_order.index('finishStatus')))
merge4_df = merge4_df[column_order]

# add qualifying data
qualifying_df = pd.read_csv('F1-raw-data/qualifying.csv')
merge5_df = pd.merge(merge4_df, qualifying_df[['raceId', 'driverId', 'constructorId', 'position', 'q1', 'q2', 'q3']], on=['raceId', 'driverId', 'constructorId'])
merge5_df.rename(columns={'position': 'qualPos', 'q1': 'q1Time', 'q2': 'q2Time', 'q3': 'q3Time'}, inplace=True)

#
# build historical features
#

# HISTORICAL FEATURES #1

# historical placement metrics
# first, points and wins going into a race:
# rename existing points column to racePoints to avoid conflicts
merge5_df.rename(columns={'points': 'racePoints'}, inplace=True)
driver_standings_df = pd.read_csv('F1-raw-data/driver_standings.csv')
merge6_df = pd.merge(merge5_df, driver_standings_df[['raceId', 'driverId', 'points', 'wins']], on=['raceId', 'driverId'])
merge6_df.rename(columns={'points': 'pointsGoingIn', 'wins': 'winsGoingIn'}, inplace=True)
# now subtract racePoints from pointsGoingIn to get pointsGoingIn
merge6_df['pointsGoingIn'] = merge6_df['pointsGoingIn'] - merge6_df['racePoints']
# now subtract if the driver won this race from winsGoingIn to get winsGoingIn
merge6_df['winsGoingIn'] = merge6_df['winsGoingIn'] - merge6_df['finishPos'].apply(lambda x: 1 if x == 1 else 0)

# performance in past 5 races 
merge6_df.sort_values(by=['driverId', 'raceDate'], inplace=True)
merge6_df['recentPlacement'] = merge6_df.groupby('driverId')['finalRank'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
# for first race, use their qualifying pos as the recentPlacement
merge6_df['recentPlacement'] = np.where(merge6_df.groupby('driverId').cumcount() == 0, merge6_df['qualPos'], merge6_df['recentPlacement'])

# avg performance of driver & team for this season so far
# Ensure data is sorted by driverId, constructorId, and raceDate
merge6_df.sort_values(by=['driverId', 'constructorId', 'raceDate'], inplace=True)

# Function to calculate the average placement excluding the current race
def calculate_avg_placement(group):
    group['seasonAvgPlace'] = group['finalRank'].expanding().mean().shift(1)
    return group

# Apply the function to calculate 'seasonAvgPlace' and reset index
merge6_df = merge6_df.groupby(['driverId', 'year']).apply(calculate_avg_placement).reset_index(drop=True)

# For the first race of the season, use the qualifying position
merge6_df['seasonAvgPlace'] = np.where(merge6_df.groupby(['driverId', 'year']).cumcount() == 0, merge6_df['qualPos'], merge6_df['seasonAvgPlace'])

# Calculate teamAvgPlace by averaging seasonAvgPlace for each team and race
team_avg = merge6_df.groupby(['constructorId', 'raceId'])['seasonAvgPlace'].mean().reset_index()
team_avg.rename(columns={'seasonAvgPlace': 'teamAvgPlace'}, inplace=True)

# Merge this team average back into the main DataFrame
merge6_df = pd.merge(merge6_df, team_avg, on=['constructorId', 'raceId'])

# avg performance of driver & team for this circuit
# Function to calculate the average circuit placement for drivers excluding the current race
merge6_df.sort_values(by=['driverId', 'circuitName', 'raceDate'], inplace=True)

def calculate_driver_circuit_avg_placement(group):
    group['driverCircuitAvgPlace'] = group['finalRank'].expanding().mean().shift(1)
    return group

# Apply the function for driver circuit average placement
merge6_df = merge6_df.groupby(['driverId', 'circuitName']).apply(calculate_driver_circuit_avg_placement).reset_index(drop=True)
# for the first time they race a track, use qualifying pos
merge6_df['driverCircuitAvgPlace'] = np.where(merge6_df.groupby(['driverId', 'circuitName']).cumcount() == 0, merge6_df['qualPos'], merge6_df['driverCircuitAvgPlace'])

merge6_df.sort_values(by=['constructorId', 'circuitName', 'raceDate'], inplace=True)

# Function to calculate the team's average circuit placement
def calculate_team_circuit_avg_placement(group):
    group['teamCircuitAvgPlace'] = group['finalRank'].expanding().mean().shift(1)
    return group

# Apply the function for team circuit average placement
merge6_df = merge6_df.groupby(['constructorId', 'circuitName']).apply(calculate_team_circuit_avg_placement).reset_index(drop=True)
merge6_df['teamCircuitAvgPlace'] = np.where(merge6_df.groupby(['constructorId', 'circuitName']).cumcount() == 0, merge6_df['qualPos'], merge6_df['teamCircuitAvgPlace'])

# Reset index to flatten the DataFrame after groupby operations
merge6_df.reset_index(drop=True, inplace=True)

# place these new features after finalRank
column_order = merge6_df.columns.tolist()
constructor_index = column_order.index('finalRank')
column_order.insert(constructor_index + 1, column_order.pop(column_order.index('seasonAvgPlace')))
column_order.insert(constructor_index + 2, column_order.pop(column_order.index('teamAvgPlace')))
merge6_df = merge6_df[column_order]

#
# HISTORICAL FEATURE #2 
#

# next, age of driver at time of race:
merge6_df = pd.merge(merge6_df, drivers_df[['driverId', 'dob']], on='driverId')
merge6_df['dob'] = pd.to_datetime(merge6_df['dob'])
merge6_df['raceDate'] = pd.to_datetime(merge6_df['raceDate'])

# calc ageAtRace
def calculate_age(dob, race_date):
    age = relativedelta(race_date, dob)
    return age.years
merge6_df['ageAtRace'] = merge6_df.apply(lambda row: calculate_age(row['dob'], row['raceDate']), axis=1)

# place dob, ageAtRace right after driverName
column_order = merge6_df.columns.tolist()
constructor_index = column_order.index('driverName')
column_order.insert(constructor_index + 1, column_order.pop(column_order.index('dob')))
column_order.insert(constructor_index + 2, column_order.pop(column_order.index('ageAtRace')))
merge6_df = merge6_df[column_order]

#
# HISTORICAL FEATURE #3
#
# pit stop performance


#
# HISTORICAL FEATURE #4
#
# overtake ability (how many positions they typically outperform their starting position by)
merge6_df['positionChange'] = merge6_df['startPos'] - merge6_df['finalRank']

# Ensure data is sorted by driverId and raceDate
merge6_df.sort_values(by=['driverId', 'raceDate'], inplace=True)

# Function to calculate the average overtake ability for the season excluding the current race
def calculate_season_overtake_ability(group):
    group['seasonOvertake'] = group['positionChange'].expanding().mean().shift(1)
    return group

# Apply the function to calculate 'seasonOvertake'
merge6_df = merge6_df.groupby(['driverId', 'year']).apply(calculate_season_overtake_ability).reset_index(drop=True)

# Function to calculate the average overtake ability for the career excluding the current race
def calculate_career_overtake_ability(group):
    group['careerOvertake'] = group['positionChange'].expanding().mean().shift(1)
    return group

# Apply the function to calculate 'careerOvertake'
merge6_df = merge6_df.groupby(['driverId']).apply(calculate_career_overtake_ability).reset_index(drop=True)

# For the first race of the season/career, set 'seasonOvertake' and 'careerOvertake' to default value of 0
merge6_df['seasonOvertake'] = np.where(merge6_df.groupby(['driverId', 'year']).cumcount() == 0, 0, merge6_df['seasonOvertake'])
merge6_df['careerOvertake'] = np.where(merge6_df.groupby('driverId').cumcount() == 0, 0, merge6_df['careerOvertake'])

# Reset index to flatten the DataFrame after groupby operations
merge6_df.reset_index(drop=True, inplace=True)

#
# HISTORICAL FEATURE #5
#
# consistency in finishing pos for driver & team


merge6_df.sort_values(by=['constructorId', 'circuitName', 'driverId', 'raceDate'], inplace=True)
merge6_df.to_csv('raw_data.csv', index=False)

#
# PIT STOP DATA
#

# Read the pit stops data
pit_stops_df = pd.read_csv('F1-raw-data/pit_stops.csv')

# Calculate the average number of pit stops and average pit time for each race
pit_stop_stats = pit_stops_df.groupby(['raceId', 'driverId']).agg(
    avgNumPitStops=('stop', 'count'), 
    avgPitTime=('milliseconds', lambda x: x.astype(float).mean())
).reset_index()

# Merge pit stop data into the main DataFrame
merge6_df = pd.merge(merge6_df, pit_stop_stats, on=['raceId', 'driverId'], how='left')

# Ensure data is sorted by driverId, year, and raceDate for the rolling calculations
merge6_df.sort_values(by=['driverId', 'year', 'raceDate'], inplace=True)

# Function to calculate the rolling average for number of pit stops and average pit time
def calculate_pit_stop_rolling_averages(group):
    group['avgNumPitStops'] = group['avgNumPitStops'].expanding().mean().shift(1).fillna(0)
    group['avgPitTime'] = group['avgPitTime'].expanding().mean().shift(1).fillna(0)
    return group

# Apply the function for each driver and season
merge6_df = merge6_df.groupby(['driverId', 'year']).apply(calculate_pit_stop_rolling_averages)

# Reset index to flatten the DataFrame after groupby operations
merge6_df.reset_index(drop=True, inplace=True)

# ... (rest of your code remains the same)

merge6_df=merge6_df.query("year >= 2011")
merge6_df.sort_values(by=['driverId', 'raceDate'], inplace=True)
merge6_df.to_csv('raw_data_with_pit_stops.csv', index=False)