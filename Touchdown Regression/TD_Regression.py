"""
This file
"""

import pandas as pd
import nfl_data_py as nfl
import numpy as np
import matplotlib.pyplot as plt
import datetime
import adjustText as txt

import warnings; warnings.simplefilter('ignore')

# turn off truncation of rows and columns if needed.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def compute_regression(past_seasons: list[int], last_season: int, plot=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to predict touchdown scoring regression candidacy for players based on historical probabilities.
    """

    #   Import play-by-play data from past seasons
    print("Importing play-by-play data...")
    pbp_hist_df = nfl.import_pbp_data(past_seasons)

    #   Import play-by-play data for the most recent season.
    pbp_prev_df = nfl.import_pbp_data([last_season])

    #   Reduce to looking at relevant columns pertaining to rushing or passing touchdowns.
    hist_rush_df = pbp_hist_df[['rush_attempt', 'rush_touchdown', 'yardline_100']].dropna()
    hist_rush_df = hist_rush_df.loc[(hist_rush_df['rush_attempt'] == 1)]

    hist_pass_df = pbp_hist_df[['complete_pass', 'pass_touchdown', 'yardline_100', 'air_yards']].dropna()
    hist_pass_df = hist_pass_df.loc[(hist_pass_df['complete_pass'] == 1)]

    #   Feature engineering: Compute the yardline where a pass is caught
    hist_pass_df['catch_yardline'] = hist_pass_df['yardline_100'] - hist_pass_df['air_yards']

    #   Compute the probability that a rushing touchdown is achieved from each scrimmage yard line.
    hist_rush_probs = hist_rush_df.groupby('yardline_100')['rush_touchdown'].value_counts(normalize = True)
    #   Compute the probability that a passing touchdown is achieved at each catch yard line.
    hist_pass_probs = hist_pass_df.groupby(['catch_yardline'])['pass_touchdown'].value_counts(normalize = True)

    #   Create a new dataframe indexed by yard line including the touchdown probabilities calculated.
    hist_rush_probs = pd.DataFrame({
        'probability_of_touchdown': hist_rush_probs.values}, index=hist_rush_probs.index).reset_index()

    hist_pass_probs = pd.DataFrame({
        'probability_of_touchdown': hist_pass_probs.values}, index=hist_pass_probs.index).reset_index()

    #   Get rid of non-scoring probabilities.
    hist_rush_probs = hist_rush_probs.loc[hist_rush_probs['rush_touchdown'] == 1]
    hist_pass_probs = hist_pass_probs.loc[hist_pass_probs['pass_touchdown'] == 1]

    hist_rush_probs = hist_rush_probs.drop('rush_touchdown', axis=1)
    hist_pass_probs = hist_pass_probs.drop('pass_touchdown', axis=1)

    #   Isolate past season rushing and passing plays.
    prev_rush_df = pbp_prev_df[['rusher_player_name', 'rusher_player_id',
                                'posteam', 'rush_touchdown', 'yardline_100']].dropna()

    prev_pass_df = pbp_prev_df[['receiver_player_name', 'receiver_player_id',
                                'posteam', 'pass_touchdown', 'yardline_100', 'air_yards', 'complete_pass']].dropna()
    prev_pass_df['catch_yardline'] = prev_pass_df['yardline_100'] - prev_pass_df['air_yards'] # Calculate catch yard line.
    prev_pass_df = prev_pass_df[prev_pass_df['complete_pass'] == 1]  # Ensure these are actual targets

    #   Merge the previous seasons plays with the probabilities for scoring along the yard line.
    prev_rush_df = prev_rush_df.merge(hist_rush_probs, how='left', on='yardline_100')
    prev_pass_df = prev_pass_df.merge(hist_pass_probs, how='left', on='catch_yardline')

    #   Group the pbp data by each player and add up the probabilities and actual touchdowns to find the respective totals.
    prev_rush_df = prev_rush_df.groupby(['rusher_player_name', 'rusher_player_id', 'posteam'], as_index = False).agg({
        'probability_of_touchdown': np.sum,
        'rush_touchdown': np.sum
    }).rename({'probability_of_touchdown': 'Expected Touchdowns'}, axis = 1)
    prev_rush_df = prev_rush_df.sort_values(by='Expected Touchdowns', ascending = False)

    prev_pass_df = prev_pass_df.groupby(['receiver_player_name', 'receiver_player_id', 'posteam'], as_index = False).agg({
        'probability_of_touchdown': np.sum,
        'pass_touchdown': np.sum
    }).rename({'probability_of_touchdown': 'Expected Touchdowns'}, axis = 1)
    prev_pass_df = prev_pass_df.sort_values(by='Expected Touchdowns', ascending = False)

    #   Rename columns.
    prev_rush_df = prev_rush_df.rename(columns={
        "rusher_player_name": "Player",
        "posteam": "Team",
        "rusher_player_id":"player_id",
        'rush_touchdown': 'Actual Touchdowns'
    })

    prev_pass_df = prev_pass_df.rename(columns={
        "receiver_player_name": "Player",
        "posteam": "Team",
        "receiver_player_id":"player_id",
        'pass_touchdown': 'Actual Touchdowns'
    })

    #   Compute regression candidacy by take the difference of expected by actual touchdowns.
    prev_rush_df['Regression'] = prev_rush_df['Expected Touchdowns'] - prev_rush_df['Actual Touchdowns']
    prev_pass_df['Regression'] = prev_pass_df['Expected Touchdowns'] - prev_pass_df['Actual Touchdowns']

    #   Give a binary yes or no answer to regression candidacy.
    prev_rush_df.loc[prev_rush_df['Regression'] >= 0, 'Regression Candidate'] = False
    prev_rush_df.loc[prev_rush_df['Regression'] < 0, 'Regression Candidate'] = True

    prev_pass_df.loc[prev_pass_df['Regression'] >= 0, 'Regression Candidate'] = False
    prev_pass_df.loc[prev_pass_df['Regression'] < 0, 'Regression Candidate'] = True

    #   Filter out low expected touchdown players.
    prev_rush_df = prev_rush_df[prev_rush_df['Expected Touchdowns'] >= 1]
    prev_pass_df = prev_pass_df[prev_pass_df['Expected Touchdowns'] >= 1]

    #   Import roster data to drop all non-running back or non-receiver players.
    roster_df = nfl.import_seasonal_rosters([last_season])
    roster_df = roster_df[['player_id', 'position']]

    #   Break up into positional dataframes.
    rb_df = prev_rush_df.merge(roster_df, on='player_id')
    rb_df = rb_df[rb_df['position'] == 'RB'].drop('position', axis=1)

    rec_df = prev_pass_df.merge(roster_df, on='player_id')
    wr_df = rec_df[rec_df['position'] == 'WR'].drop('position', axis=1)
    te_df = rec_df[rec_df['position'] == 'TE'].drop('position', axis=1)

    if plot:
        plot_reg(rb_df, 'RB')
        plot_reg(wr_df, 'WR')
        plot_reg(te_df, 'TE')

    return rb_df, wr_df, te_df

def validate(rb_reg_df: pd.DataFrame, wr_reg_df: pd.DataFrame, te_reg_df: pd.DataFrame, season: int) -> None:
    """
    Validate the performance of the regression predictions based on a known season results.
    """

    actual_df = nfl.import_seasonal_data([season + 1])[['player_id', 'rushing_tds', 'receiving_tds']]
    actual_df = actual_df.rename(columns={'rushing_tds': 'Next Year Rushing Touchdowns',
                                          'receiving_tds': 'Next Year Receiving Touchdowns'})

    rb_reg_df = rb_reg_df.merge(actual_df, on='player_id').drop('Next Year Receiving Touchdowns', axis=1)
    wr_reg_df = wr_reg_df.merge(actual_df, on='player_id').drop('Next Year Rushing Touchdowns', axis=1)
    te_reg_df = te_reg_df.merge(actual_df, on='player_id').drop('Next Year Rushing Touchdowns', axis=1)

    rb_reg_df.loc[rb_reg_df['Next Year Rushing Touchdowns'] < rb_reg_df['Actual Touchdowns'], 'Did Regress'] = True
    rb_reg_df.loc[rb_reg_df['Next Year Rushing Touchdowns'] >= rb_reg_df['Actual Touchdowns'], 'Did Regress'] = False

    wr_reg_df.loc[wr_reg_df['Next Year Receiving Touchdowns'] < wr_reg_df['Actual Touchdowns'], 'Did Regress'] = True
    wr_reg_df.loc[wr_reg_df['Next Year Receiving Touchdowns'] >= wr_reg_df['Actual Touchdowns'], 'Did Regress'] = False

    te_reg_df.loc[te_reg_df['Next Year Receiving Touchdowns'] < te_reg_df['Actual Touchdowns'], 'Did Regress'] = True
    te_reg_df.loc[te_reg_df['Next Year Receiving Touchdowns'] >= te_reg_df['Actual Touchdowns'], 'Did Regress'] = False

    for _, row in rb_reg_df.iterrows():
        if row['Did Regress'] == row['Regression Candidate']:
            rb_reg_df.loc[row.name, 'Was Correct'] = True
        else:
            rb_reg_df.loc[row.name, 'Was Correct'] = False

    rush_correct = rb_reg_df['Was Correct'].sum()
    print(f"The model predicted correctly {rush_correct} RB's out of {len(rb_reg_df)} giving an accuracy of {rush_correct / len(rb_reg_df)}")

    for _, row in wr_reg_df.iterrows():
        if row['Did Regress'] == row['Regression Candidate']:
            wr_reg_df.loc[row.name, 'Was Correct'] = True
        else:
            wr_reg_df.loc[row.name, 'Was Correct'] = False

    pass_correct = wr_reg_df['Was Correct'].sum()
    print(f"The model predicted correctly {pass_correct} WR's out of {len(wr_reg_df)} giving an accuracy of {pass_correct / len(wr_reg_df)}")

    for _, row in te_reg_df.iterrows():
        if row['Did Regress'] == row['Regression Candidate']:
            te_reg_df.loc[row.name, 'Was Correct'] = True
        else:
            te_reg_df.loc[row.name, 'Was Correct'] = False

    te_correct = te_reg_df['Was Correct'].sum()
    print(f"The model predicted correctly {te_correct} TE's out of {len(te_reg_df)} giving an accuracy of {te_correct / len(te_reg_df)}")


def plot_reg(df: pd.DataFrame, pos: str) -> None:
    """Plot regression candidacy and highlight the largest under/over-performers"""
    plt.figure(figsize=(12, 8))

    colors = {True: 'red', False: 'green'} # Define colors for candidates

    # Plotting points for 'Yes' and 'No' separately.
    for candidate, color in colors.items():
        subset = df[df['Regression Candidate'] == candidate]
        plt.scatter(subset['Expected Touchdowns'],
                    subset['Actual Touchdowns'],
                    label='Negative Regression Candidates' if candidate else 'Positive Regression Candidates',
                    color=color,
                    alpha=0.7,  # Adjust transparency
                    s=50)  # Adjust marker size

    # Add the y=x line (Regression Barrier)
    all_td_values = pd.concat([df['Expected Touchdowns'], df['Actual Touchdowns']])
    max_val = all_td_values.max() # Determine the plot limits to draw the line across the entire range

    plt.plot([0, max_val], [0, max_val],
             color='grey',
             linestyle='--',
             linewidth=2)

    #  Add Labels, Title, and Legend
    plt.title(f'Touchdown Regression for {pos}\'s', fontsize=16)
    plt.xlabel('Expected Touchdowns', fontsize=14)
    plt.ylabel('Actual Touchdowns', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)  # grid for readability

    #   Annotate the top 5 most positive and negative regression candidates.
    players = [] # List to store text objects for adjustText
    sorted_df = df.sort_values(by='Regression', ascending=False)
    players_to_annotate = pd.concat([
        sorted_df[sorted_df['Regression Candidate'] == True].tail(5), # Top positive regression
        sorted_df[sorted_df['Regression Candidate'] == False].head(5)   # Top negative regression
    ])

    for index, row in players_to_annotate.iterrows():
        players.append(plt.text(row['Expected Touchdowns'] + 0.2, # x-offset for text
                 row['Actual Touchdowns'], # y-offset for tex
                 row['Player'],
                 fontsize=8))

    txt.adjust_text(players)
    plt.show()

if __name__ == "__main__":
    past_seasons = list(range(2013, 2024))
    last_season = 2024

    rb_reg_df, wr_reg_df, te_reg_df = compute_regression(past_seasons, last_season, plot=True)
    if last_season <= (datetime.datetime.now().year - 2):
        validate(rb_reg_df, wr_reg_df, te_reg_df, last_season)