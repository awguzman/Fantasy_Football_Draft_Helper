# A draft board creator for a Dynasty fantasy football league hosted through Sleeper.com using data from FantasyPros.com.
# Must run Get_Sleeper_Map.py before running this code (Only need to run the affermentioned script after every years draft after the initial run.
# Outputs a .csv file.

import Get_Sleeper_Rosters
import pandas as pd

import requests, re, json

# turn off truncation of rows and columns if needed.
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def compute_projected_points(pos: str) -> pd.DataFrame:
    """ Scrape Fantasypros.com projected stats for fantasy point calculations. """

    if pos not in ['qb', 'rb', 'wr', 'te']:
        raise Exception(f'Invalid position: {pos}. Must be qb, rb, wr, or te. ')

    # Store fantasypros url for projected stats.
    proj_url = f'https://www.fantasypros.com/nfl/projections/{pos}.php?week=draft'

    # Scrape Fantasy Pro's projected player stats and store in a DataFrame.
    proj_df = pd.read_html(proj_url, header=1)[0]

    # Pick out columns that are important for fantasy point calculation.
    if pos == 'qb':
        stats_columns = ['Player', 'YDS', 'TDS', 'INTS', 'YDS.1', 'TDS.1', 'FL']
    elif pos == 'te':
        stats_columns = ['Player', 'REC', 'YDS', 'TDS', 'FL']
    else:
        stats_columns = ['Player', 'YDS', 'TDS', 'REC', 'YDS.1', 'TDS.1', 'FL']

    # Filter columns according to the columns picked out above.
    proj_df = proj_df[stats_columns]

    # Compute projected fantasy points and add a column in the positional DataFrames.
    if pos == 'qb':
        proj_df['Proj. Points'] = (
                proj_df['YDS'] * scoring_weights['passing_yds'] +
                proj_df['TDS'] * scoring_weights['passing_td'] +
                proj_df['INTS'] * scoring_weights['int'] +
                proj_df['YDS.1'] * scoring_weights['rushing_yds'] +
                proj_df['TDS.1'] * scoring_weights['rushing_td'] +
                proj_df['FL'] * scoring_weights['fumble'])

    elif pos == 'te':
        proj_df['Proj. Points'] = (
                proj_df['REC'] * scoring_weights['receptions'] +
                proj_df['YDS'] * scoring_weights['receiving_yds'] +
                proj_df['TDS'] * scoring_weights['receiving_td'] +
                proj_df['FL'] * scoring_weights['fumble'])

    else:
        proj_df['Proj. Points'] = (
                proj_df['YDS'] * scoring_weights['rushing_yds'] +
                proj_df['TDS'] * scoring_weights['rushing_td'] +
                proj_df['REC'] * scoring_weights['receptions'] +
                proj_df['YDS.1'] * scoring_weights['receiving_yds'] +
                proj_df['TDS.1'] * scoring_weights['receiving_td'] +
                proj_df['FL'] * scoring_weights['fumble'])

    # Filter out raw projected stats.
    proj_df = proj_df[['Player', 'Proj. Points']]

    return proj_df

def get_ecr_data(pos:str) -> pd.DataFrame:
    """ Scrape Fantasypros.com ECR Data. """

    if pos not in ['qb', 'rb', 'wr', 'te']:
        raise Exception(f'Invalid position: {pos}. Must be qb, rb, wr, or te. ')

    ecr_url = f'https://www.fantasypros.com/nfl/rankings/dynasty-{pos}.php'

    # Find the ecrData JSON in the html file. Load it in as a DataFrame.
    response = requests.get(ecr_url)
    match = re.search(r'var ecrData = (\{.*?\});', response.text)
    ecr_json = match.group(1)
    ecr_data = json.loads(ecr_json)
    players_list = (ecr_data.get('players', []))
    ecr_df = pd.DataFrame(players_list)

    ecr_df['Player'] = ecr_df['player_name'] + ' ' + ecr_df['player_team_id']
    ecr_columns = ['Player', 'player_age', 'player_bye_week', 'pos_rank', 'rank_std']
    ecr_df = ecr_df[ecr_columns]
    ecr_df = ecr_df.rename({'player_age': 'Age',
                            'player_bye_week': 'Bye',
                            'pos_rank': 'Rank',
                            'rank_std': 'Confidence'}, axis=1)

    return ecr_df

def get_draft_board(pos: str) -> pd.DataFrame:

    if pos not in ['qb', 'rb', 'wr', 'te']:
        raise Exception(f'Invalid position: {pos}. Must be qb, rb, wr, or te. ')

    stats_df = compute_projected_points(pos)
    ecr_df = get_ecr_data(pos)

    # Merge projected points and ecr dataFrames.
    board_df = ecr_df.merge(stats_df, how='left', on='Player')

    # Import the Get_Sleeper_Rosters.py roster data.
    taken_df = Get_Sleeper_Rosters.roster_df

    # Remove taken players from the draft board.
    for _, row in taken_df.iterrows():
        for player in row['Roster']:
            board_df.drop(board_df[board_df['Player'].str.contains(player)].index, inplace=True)

    return board_df

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

for pos in ['qb', 'rb', 'wr', 'te']:
    board_df = get_draft_board(pos)
    # Generate a .csv file.
    board_df.to_csv(f'Dynasty_{pos}_Draft_Board.csv')
