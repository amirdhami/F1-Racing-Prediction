# README IS CURRENTLY OUTDATED, LOOK AT raw_data.csv AND construct_raw.py FOR PROJECT DETAILS

## CURRENT CONTENTS OF THE REPO:

# raw_data
- contains the 3 raw data files
- these were all directly pulled from the kaggle site

# F1_data.csv
- contains the consolidated data
- starting point for my transforms

# F1_qualtime.csv
- contains the consolidated data
- "Qual Time" column now contains avg time of qual phases that the driver competed in if the race was new qual format (3-round quals started in 2006)
- "Qual Time" may contain "DNS" or "DNF" if a driver failed to complete phase 1 of the qual for any reason

# F1_racetime.csv
- most recent and processed version of data--please use this as a baseline for further modification
- contains prev data from F1_qualtime.csv
- additional columns are "Time in Seconds" and "Adjusted Time"
- "Time in Seconds" is an intermediate for calculations
- "Adjusted Time" is an adjusted form of "Time/Retired" in seconds for all drivers
- "Adjusted Time" contains either a time in seconds (for drivers who did not complete all laps in time, the avg time for the 1st place driver to complete a lap is added per lap they were behind) or "DNF" if the driver did not finish
- "Adjusted Time" has some holes due to disqualifications--these are not yet accounted for

# qualtime.py
- transformed "F1_data.csv" into "F1_qualtime.csv"
- note that manually filled in holes where qual was unfinished (DNF, DNS, etc.) will be erased if you rerun this script

# finaltime.py
- transformed "F1_qualtime.csv" into "F1_racetime.csv"
