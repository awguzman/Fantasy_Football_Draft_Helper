# A draft board creator for a Dynasty fantasy football league hosted through Sleeper.com using data from FantasyPros.com.
# Must run Get_Sleeper_Map.py before running this code (Only need to run the affermentioned script after every years draft after the initial run.
# Outputs a .csv file.

import Get_Sleeper_Rosters
import pandas as pd

# turn off truncation of rows and columns if needed.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Store html url's for each position measuring projected season stats.
qb_url = "https://www.fantasypros.com/nfl/projections/qb.php?week=draft"
rb_url = "https://www.fantasypros.com/nfl/projections/rb.php?week=draft"
wr_url = "https://www.fantasypros.com/nfl/projections/wr.php?week=draft"
te_url = "https://www.fantasypros.com/nfl/projections/te.php?week=draft"
dynasty_url = "https://www.fantasypros.com/nfl/adp/dynasty-overall.php"

# Scrape Fantasy Pro's projected player stats and store in DataFrames according to position.
qb_df = pd.read_html(qb_url, header=1)[0]
rb_df = pd.read_html(rb_url, header=1)[0]
wr_df = pd.read_html(wr_url, header=1)[0]
te_df = pd.read_html(te_url, header=1)[0]

# Scrape Fantasy Pro's dynasty ADP data and store in a DataFrame.
dynasty_df = pd.read_html(dynasty_url, header=0)[0]
dynasty_df = dynasty_df.rename({'Player Team (Bye)': 'Player', 'AVG': 'Avg ADP'}, axis=1)

# Reformat the player names in the Dynasty ADP DataFrame to match other DataFrame formats.
byes = ['(5)', '(6)', '(7)', '(8)', '(9)', '(10)', '(11)', '(12)', '(13)', '(14)']

def player_cleanup(df):  # Function to get rid of any team, bye week, or injury data from the player name index.
    for _, row in df.iterrows():
        player = row['Player'].split(' ')
        for i in range(len(player)):
            if player[i] in byes:
                player[i] = ''
        player = ' '.join(player).strip()
        df.update(df['Player'].replace({row['Player']: player}))
    return df


dynasty_df = player_cleanup(dynasty_df)

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

# Combine the positional DataFrames into a single DataFrame.
qb_df = qb_df[['Player', 'Fantasy Points']]
rb_df = rb_df[['Player', 'Fantasy Points']]
wr_df = wr_df[['Player', 'Fantasy Points']]
te_df = te_df[['Player', 'Fantasy Points']]
pos_df = [qb_df, rb_df, wr_df, te_df]
board_df = pd.concat(pos_df).sort_values(by='Fantasy Points', ascending=False, ignore_index=True)

# Merge the Dynasty ADP and Fantasy points DataFrames.
board_df = dynasty_df.merge(board_df, how='left', on='Player')[['Player', 'POS', 'Fantasy Points', 'Avg ADP']].sort_values(by='Fantasy Points', ascending=False)

# Import the Get_Sleeper_Rosters.py roster data.
taken_df = Get_Sleeper_Rosters.roster_df

# Take taken players out of the draft board.
for _, row in taken_df.iterrows():
    for player in row['Roster']:
        board_df.drop(board_df.loc[board_df['Player'].str.contains(player)].index, inplace=True)
board_df = board_df.dropna()

# Generate a .csv file.
board_df.to_csv('Dynasty_Draft_Board.csv')
