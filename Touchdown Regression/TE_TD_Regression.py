import pandas as pd
import nfl_data_py as nfl
import numpy as np
import warnings; warnings.simplefilter('ignore')

# turn off truncation of rows and columns if needed.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#   Import play-by-play data from 2000 to 2022 seasons (This may take a minute)
seasons = range(2000, 2023)
pbp_df = nfl.import_pbp_data(seasons)

#   Reduce to looking at relavent columns pertaining to catching touchdowns.
passing_df = pbp_df[['pass_attempt', 'pass_touchdown', 'yards_after_catch', 'two_point_attempt']]

#   Get rid of two-point conversion attempts.
passing_df = passing_df.loc[(passing_df['two_point_attempt'] == 0) & (passing_df['pass_attempt'] == 1)]

#   Compute the probability that a passing touchdown is achieved at each yard line and yards after catch.
passing_df_probs = passing_df.groupby(['yards_after_catch'])['pass_touchdown'].value_counts(normalize = -True)

#   Create a new datafrme indexed by yard line and yards after catch including the touchdown probabilities calculated.
passing_df_probs = pd.DataFrame({
    'probability_of_touchdown': passing_df_probs.values}, index=passing_df_probs.index).reset_index()

#   Get rid of non-passing plays.
passing_df_probs = passing_df_probs.loc[passing_df_probs['pass_touchdown'] == 1]

#   Get rid of pass_touchdown as this is implicit after getting rid of non-passing plays.
passing_df_probs = passing_df_probs.drop('pass_touchdown', axis=1)

#   Import 2023 play-by-play data and isolate players who scored receiving touchdowns.
pbp_2023_df = nfl.import_pbp_data([2023])
pbp_2023_df = pbp_2023_df[['receiver_player_name', 'receiver_player_id', 'posteam', 'pass_touchdown', 'yards_after_catch']].dropna()

#   Merge 2023 pbp dataframe with probability dataframe along yard line.
exp_df = pbp_2023_df.merge(passing_df_probs, how='left', on=['yards_after_catch'])

#   Group the pbp data by each player and add up the probabilities and actual touchdowns to find the respective totals.
exp_df = exp_df.groupby(['receiver_player_name', 'receiver_player_id', 'posteam'], as_index = False).agg({
    'probability_of_touchdown': np.sum,
    'pass_touchdown': np.sum
}).rename({'probability_of_touchdown': 'Expected Touchdowns'}, axis = 1)
exp_df = exp_df.sort_values(by='Expected Touchdowns', ascending = False)

#   Rename columns.
exp_df = exp_df.rename(columns={
    "receiver_player_name": "Player",
    "posteam": "Team",
    "receiver_player_id":"ID",
    'pass_touchdown': 'Actual Touchdowns'
})

#   Import roster data to drop all non wide receiver players from exp_df.
roster_df = nfl.import_seasonal_rosters([2023])
roster_df = roster_df[['player_id', 'position']].rename(columns = {"player_id": "ID"})
exp_df = exp_df.merge(roster_df, on='ID')
exp_df = exp_df[exp_df['position'] == 'TE'].drop('position', axis=1)
exp_df.drop(columns=['ID'])

#   Compute regression candidacy by take the difference of expected by actual touchdowns.
exp_df['Regression'] = exp_df['Expected Touchdowns'] - exp_df['Actual Touchdowns']

#   Give a binary yes or no answer to regression candidacy.
exp_df.loc[exp_df['Regression'] >= 0, 'Regression Candidate'] = 'No'
exp_df.loc[exp_df['Regression'] < 0, 'Regression Candidate'] = 'Yes'

#   Filter out low expected touchdown players.
exp_df = exp_df[exp_df['Expected Touchdowns'] >= 1]

#   Export to csv
exp_df.to_csv('TE_TD_Regression.csv')
