# Best_Ball_Draft_Helper
This is a repository of python code intended to use data analysis and machine learning techniques to help me draft in my Best Ball League.

We primarily use the Python data analysis library Pandas to perform data analysis on fantasy football data.

Planned additions: 

Touchdown regression candidate calculator for all positions

-------------------------------------------------------------------------------------------------------------------------------

Best_Ball_Draft_Board.py  -  We use Pandas to web scrape projected stats and average draft position (ADP) data from FantasyPros.com to compute projected fantasy points and value over replacement (VOR) points. This is compiled into a .csv file that can be used as a draft board for a Best Ball draft. Should be good to run every year following schedule release.

RB_TD_Regression.py  -  Running back rushing touchdown regression candidate calculator for the 2023 season. Imports play-by-play data from 2000-2022 via nfl_data_py and computes the expected rushing touchdowns vs actual rushing touchdowns for each running back in the 2023 season. This is compiles into a .csv file.

WR_TD_Regression.py  -  Wide reciever passing touchdown regression candidate calculator for the 2023 season. Imports play-by-play data from 2000-2022 via nfl_data_py and computes the expected receiving touchdowns vs actual touchdowns for each wide reciever in the 2023 season. This is compiles into a .csv file.
