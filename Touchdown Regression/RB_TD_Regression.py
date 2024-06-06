import pandas as pd
import nfl_data_py as nfl
import numpy as np
import warnings; warnings.simplefilter('ignore')

# turn off truncation of rows and columns if needed.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#   Import play-by-play data from 2012 to 2022 seasons (This may take a minute)
seasons = range(2000, 2023)
pbp_df = nfl.import_pbp_data(seasons)

#   Reduce to looking at relavent columns pertaining to rushing touchdowns.
rushing_df = pbp_df[['rush_attempt', 'rush_touchdown', 'yardline_100', 'two_point_attempt']]

#   Get rid of two-point conversion attempts.
rushing_df = rushing_df.loc[(rushing_df['two_point_attempt'] == 0) & (rushing_df['rush_attempt'] == 1)]

#   Compute the probability that a rushing touchdown is achieved at each yard line.
rushing_df_probs = rushing_df.groupby('yardline_100')['rush_touchdown'].value_counts(normalize = -True)

#   Create a new datafrme indexed by yard line including the touchdown probabilities calculated.
rushing_df_probs = pd.DataFrame({
    'probability_of_touchdown': rushing_df_probs.values}, index=rushing_df_probs.index).reset_index()

#   Get rid of non-rushing plays.
rushing_df_probs = rushing_df_probs.loc[rushing_df_probs['rush_touchdown'] == 1]

#   Get rid of rush_touchdown as this is implicit after getting rid of non-rushing plays.
rushing_df_probs = rushing_df_probs.drop('rush_touchdown', axis=1)

#   Import 2023 play-by-play data and isolate players who scored rushing touchdowns.
pbp_2023_df = nfl.import_pbp_data([2023])
pbp_2023_df = pbp_2023_df[['rusher_player_name', 'rusher_player_id', 'posteam', 'rush_touchdown', 'yardline_100']].dropna()

#   Merge 2023 pbp dataframe with probability dataframe along yard line.
exp_df = pbp_2023_df.merge(rushing_df_probs, how='left', on='yardline_100')

#   Group the pbp data by each player and add up the probabilities and actual touchdowns to find the respective totals.
exp_df = exp_df.groupby(['rusher_player_name', 'rusher_player_id', 'posteam'], as_index = False).agg({
    'probability_of_touchdown': np.sum,
    'rush_touchdown': np.sum
}).rename({'probability_of_touchdown': 'Expected Touchdowns'}, axis = 1)
exp_df = exp_df.sort_values(by='Expected Touchdowns', ascending = False)

#   Rename columns.
exp_df = exp_df.rename(columns={
    "rusher_player_name": "Player",
    "posteam": "Team",
    "rusher_player_id":"ID",
    'rush_touchdown': 'Actual Touchdowns'
})

#   Import roster data to drop all non running back players from exp_df.
roster_df = nfl.import_seasonal_rosters([2023])
roster_df = roster_df[['player_id', 'position']].rename(columns = {"player_id": "ID"})
exp_df = exp_df.merge(roster_df, on='ID')
exp_df = exp_df[exp_df['position'] == 'RB'].drop('position', axis=1)
exp_df.drop(columns=['ID'])

#   Compute regression candidacy by take the difference of expected by actual touchdowns.
exp_df['Regression'] = exp_df['Expected Touchdowns'] - exp_df['Actual Touchdowns']

#   Give a binary yes or no answer to regression candidacy.
exp_df.loc[exp_df['Regression'] >= 0, 'Regression Candidate'] = 'No'
exp_df.loc[exp_df['Regression'] < 0, 'Regression Candidate'] = 'Yes'

#   Filter out low expected touchdown players.
exp_df = exp_df[exp_df['Expected Touchdowns'] >= 1]

#   Export to csv
exp_df.to_csv('RB_TD_Regression.csv')
