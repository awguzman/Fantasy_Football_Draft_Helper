#   Creates a .csv file giving mapping data for players id's to player names used by the Sleeper platform.
#   Only needs to be run every once in a while. The .csv file generated will be read by other scripts.

import pandas as pd
import requests

# Requests data needed to translate sleeper player_id to player names.
sleeper_url = "https://api.sleeper.app/v1/players/nfl"
sleeper_request = requests.get(sleeper_url).json()
players_df = pd.DataFrame.from_dict(sleeper_request, orient='index')[['player_id', 'full_name', 'status']]

# Create a .csv file
players_df.to_csv('Sleeper_Player_Map.csv')
