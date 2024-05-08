# Best_Ball_Draft_Helper
This is a repository of python code intended to use data analysis and machine learning techniques to help me draft in my Best Ball League.

We primarily use the Python data analysis library Pandas to perform data analysis on fantasy football data.

Planned additions: 

2024 Draft board

Touchdown regression candidate calculator for all positions

-------------------------------------------------------------------------------------------------------------------------------

2023_Draft_Board.py  -  This holds the first version of my draft board creater using data from the 2023 season. We use Pandas to web scrape projected stats and average draft position (ADP) data from FantasyPros.com to compute projected fantasy points and value over replacement (VOR) points. This is compiled into a .csv file that can be used as a draft board for a Best Ball draft.

2024_Best_Ball_Draft_Board - An updated version of 2023_Draft_Board.py with minor adjustements made to account for changes in how FantasyPros stores data.

RB_TD_Regression.py  -  Running back rushing touchdown regression candidate calculator for the 2023 season. Imports play-by-play data from 2012-2022 via nfl_data_py and computes the expected rushing touchdowns vs actual rushing touchdowns for each running back in the 2023 season. This is compiles into a .csv file.
