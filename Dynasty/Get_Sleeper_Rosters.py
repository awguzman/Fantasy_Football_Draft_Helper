#   Script to import Sleeper roster data after user inputs their league number. Will generate a DataFrame giving roster names per team.

import pandas as pd
import requests
import nfl_data_py as nfl

# turn off truncation of rows and columns if needed.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Store Sleeper URL for league roster data.
league_id = input('Enter Sleeper platform league number (This is found in the sleeper url):')
sleeper_url = "https://api.sleeper.app/v1/league/" + league_id + "/rosters"

# Request league data from Sleeper and store in a DataFrame
sleeper_request = requests.get(sleeper_url).json()
rosters_df = pd.DataFrame.from_dict(sleeper_request, orient='columns')
rosters_df = rosters_df[['owner_id', 'players']]
rosters_df = rosters_df.rename(columns={'players': 'sleeper_ids'})

# Load in the player data map from Sleeper.
# map_df = pd.read_csv('Sleeper_Player_Map.csv')[['player_id', 'team', 'full_name']]
map_df = nfl.import_ids()[['name', 'sleeper_id', 'fantasypros_id']]

# Use the sleeper player map to translate ID's into full names and create a new column listing all the players names.
rosters_df.insert(2, 'fantasypros_ids', None)
rosters_df['fantasypros_ids'] = rosters_df['fantasypros_ids'].astype(object)

for _, row in rosters_df.iterrows():
    team_roster = []
    for player_id in row['sleeper_ids']:
        index = map_df.loc[map_df['sleeper_id'] == int(player_id)]
        team_roster.append(index['fantasypros_id'].values[0])
    rosters_df.at[_, 'fantasypros_ids'] = team_roster
