# A draft board creater for my best ball fantasy football league using data from FantasyPros.com. Outputs a .csv file.

import pandas as pd

# turn off truncation of rows and columns if needed.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Store html url's for each position measuring projected season stats for the 2023 season.
qb_url = "https://www.fantasypros.com/nfl/projections/qb.php?week=draft"
rb_url = "https://www.fantasypros.com/nfl/projections/rb.php?week=draft"
wr_url = "https://www.fantasypros.com/nfl/projections/wr.php?week=draft"
te_url = "https://www.fantasypros.com/nfl/projections/te.php?week=draft"

# Scrape Fantasy Pro's projected player stats and store in DataFrames according to position.
qb_df = pd.read_html(qb_url, header=1)[0]
rb_df = pd.read_html(rb_url, header=1)[0]
wr_df = pd.read_html(wr_url, header=1)[0]
te_df = pd.read_html(te_url, header=1)[0]

# Pick out columns of each positional DataFrame that are important for fantasy point calculation.
qb_stats = ['Player', 'YDS', 'TDS', 'INTS', 'YDS.1', 'TDS.1', 'FL']
rb_stats = ['Player', 'YDS', 'TDS', 'REC', 'YDS.1', 'TDS.1', 'FL']
wr_stats = ['Player', 'REC', 'YDS', 'TDS', 'YDS.1', 'TDS.1', 'FL']
te_stats = ['Player', 'REC', 'YDS', 'TDS', 'FL']

# Filter columns of each positional DataFrame according to the columns picked out above.
qb_df = qb_df[qb_stats]
rb_df = rb_df[rb_stats]
wr_df = wr_df[wr_stats]
te_df = te_df[te_stats]

# Set scoring weights for fantasy point calculations.
# Free to change to match a given league.
scoring_weights = {
    'receptions': 0.5,  # half-PPR
    'receiving_yds': 0.1,
    'receiving_td': 6,
    'rushing_yds': 0.1,
    'rushing_td': 6,
    'passing_yds': 0.05,
    'passing_td': 4,
    'int': -1,
    'fumble': -1
}

# Compute fantasy points and add a column in the positional DataFrames, then filter out raw stats.
qb_df['Fantasy Points'] = (
    qb_df['YDS']*scoring_weights['passing_yds'] +
    qb_df['TDS']*scoring_weights['passing_td'] +
    qb_df['INTS']*scoring_weights['int'] +
    qb_df['YDS.1']*scoring_weights['rushing_yds'] +
    qb_df['TDS.1']*scoring_weights['rushing_td'] +
    qb_df['FL']*scoring_weights['fumble']
)

rb_df['Fantasy Points'] = (
    rb_df['YDS']*scoring_weights['rushing_yds'] +
    rb_df['TDS']*scoring_weights['rushing_td'] +
    rb_df['REC']*scoring_weights['receptions'] +
    rb_df['YDS.1']*scoring_weights['receiving_yds'] +
    rb_df['TDS.1']*scoring_weights['receiving_td'] +
    rb_df['FL']*scoring_weights['fumble']
)

wr_df['Fantasy Points'] = (
    wr_df['REC']*scoring_weights['receptions'] +
    wr_df['YDS']*scoring_weights['receiving_yds'] +
    wr_df['TDS']*scoring_weights['receiving_td'] +
    wr_df['YDS.1']*scoring_weights['rushing_yds'] +
    wr_df['TDS.1']*scoring_weights['rushing_td'] +
    wr_df['FL']*scoring_weights['fumble']
)

te_df['Fantasy Points'] = (
    te_df['REC']*scoring_weights['receptions'] +
    te_df['YDS']*scoring_weights['receiving_yds'] +
    te_df['TDS']*scoring_weights['receiving_td'] +
    te_df['FL']*scoring_weights['fumble']
)

qb_df = qb_df[['Player', 'Fantasy Points']]
rb_df = rb_df[['Player', 'Fantasy Points']]
wr_df = wr_df[['Player', 'Fantasy Points']]
te_df = te_df[['Player', 'Fantasy Points']]
pos_df = [qb_df, rb_df, wr_df, te_df]
board_df = pd.concat(pos_df).sort_values(by='Fantasy Points', ascending=False)


# Scrape Fantasy Pro's Average Draft Position (ADP) data and store in a DataFrames according to position.
adp_url = "https://www.fantasypros.com/nfl/adp/best-ball-overall.php"
adp_df = pd.read_html(adp_url, header=0)[0]

# Reformat the player name index in each DataFrame to match each other for merging.
adp_df = adp_df.rename({'Player Team (Bye)': 'Player', 'Rank': 'ADP'}, axis=1)
adp_df = adp_df[['Player', 'POS', 'ADP']]
teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAC', 'KC',
         'LV', 'LAR', 'LAC', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS']
byes = ['(5)', '(6)', '(7)', '(8)', '(9)', '(10)', '(11)', '(12)', '(13)', '(14)']


def player_cleanup(df):  # Function to get rid of any team, bye week, or injury data from the player name index.
    for _, row in df.iterrows():
        player = row['Player'].split(' ')
        for i in range(len(player)):
            if player[i] in teams:
                player[i] = ''
            elif player[i] in byes:
                player[i] = ''
            elif player[i] == 'O':
                player[i] = ''
        player = ' '.join(player).strip()
        df.update(df['Player'].replace({row['Player']: player}))
    return df


adp_df = player_cleanup(adp_df)
board_df = player_cleanup(board_df)

# Merge ADP Dataframe with projected points DataFrame before sorting by ADP.
board_df = board_df.merge(adp_df, how='left', on='Player')[['Player', 'POS', 'Fantasy Points', 'ADP']].sort_values(by='ADP')
board_df = board_df.dropna(subset=['ADP'])

# Determine the ADP cutoff players to be used later in computing Value over Replacement (VOR) points.
# Free to change the specific cutoff number.
board_df_cutoff = board_df[:75]

replacement_players = {
    'QB': '',
    'RB': '',
    'WR': '',
    'TE': ''
}

# Find the last player by position before the cutoff point and store them as replacement players.
for _, row in board_df_cutoff.iterrows():

    position = row['POS'][:2]   # Cutoff the numerical part of the position data
    player = row['Player']

    if position in replacement_players:
        replacement_players[position] = player

# Store the computed Fantasy Points for each replacement player.
replacement_values = {}

for position, player_name in replacement_players.items():
    player = board_df.loc[board_df['Player'] == player_name]

    replacement_values[position] = player['Fantasy Points'].tolist()[0]

# Compute the Value over Replacement score and store as a new column in the dataFrame.
board_df['VOR'] = board_df.apply(
    lambda row: row['Fantasy Points'] - replacement_values.get(row['POS'][:2]), axis=1
)

# Compute and store the Z score of the VOR points.
board_df['Z Score'] = board_df['VOR'].apply(lambda x: (x - board_df['VOR'].mean()) / board_df['VOR'].std())

# Create a .csv file of the final draft board.
board_df.to_csv('Best_Ball_Draft_Board.csv')