#   Script to import Sleeper roster data after user inputs their league number. Will generate a DataFrame giving roster names per team.

import pandas as pd
import requests

# turn off truncation of rows and columns if needed.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Store Sleeper URL for league roster data.
league_id = input('Enter Sleeper platform league number (This is found in the sleeper url):')
sleeper_url = "https://api.sleeper.app/v1/league/" + league_id + "/rosters"

# Request league data from Sleeper and store in a DataFrame
sleeper_request = requests.get(sleeper_url).json()
roster_df = pd.DataFrame.from_dict(sleeper_request, orient='columns')
roster_df = roster_df[['owner_id', 'players']]

# Load in the player data map from Sleeper.
map_df = pd.read_csv('Sleeper_Player_Map.csv')[['player_id', 'team', 'full_name']]

# Use the sleeper player map to translate ID's into full names and create a new column listing all the players names.
roster_df.insert(2, 'Roster', None)
roster_df['Roster'] = roster_df['Roster'].astype(object)
for _, row in roster_df.iterrows():
    team_roster = []
    for player_id in row['players']:
        player = map_df.loc[map_df['player_id'] == player_id]
        name = player['full_name']
        team_roster.append(name.to_string(index=False))
    roster_df.at[_, 'Roster'] = team_roster
