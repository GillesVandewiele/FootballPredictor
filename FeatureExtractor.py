import urllib

from lxml import etree, html
import pandas as pd
import numpy as np
import ast


def get_prior_game_statistics(home_team, away_team, game_date, df, N_GAMES=5):

    print(game_date, 'HOME TEAM: ', home_team, 'AWAY TEAM: ', away_team)
    # Filtering out some dataframes
    home_home_games = df[(df.date < game_date) & (df.home_team == home_team)].tail(N_GAMES)

    home_games = df[(df.date < game_date) & ((df.home_team == home_team) | (df.away_team == home_team))].tail(N_GAMES)

    away_away_games = df[(df.date < game_date) & (df.home_team == away_team)].tail(N_GAMES)

    away_games = df[(df.date < game_date) & ((df.home_team == away_team) | (df.away_team == away_team))].tail(N_GAMES)

    homevsaway_homestadium_games = df[(df.date < game_date) & (df.home_team == home_team) &
                                      (df.away_team == away_team)].tail(N_GAMES)

    homevsaway_games = df[(df.date < game_date) & ((df.home_team == home_team) | (df.away_team == home_team)) &
                          ((df.away_team == away_team) | (df.home_team == away_team))].tail(N_GAMES)

    print(home_games['date'])
    pass


def get_fifa_rating(player_name, season):
    url = "http://sofifa.com/players?keyword=" + urllib.parse.quote_plus(player_name) + "&v=" + season + "&hl=en-US"
    with urllib.request.urlopen(url) as page:
        s = page.read().decode("utf-8")
    tree = html.fromstring(s).getroottree()
    for table_cell in tree.findall('//td'):
        if 'data-title' in table_cell.attrib and table_cell.attrib['data-title'] == "Overall rating":
            rating = table_cell.find('./span').text
            print((url, int(rating)))
            return 1, int(rating)

    print((url, 50))
    return 0, 50


#############################################################################################
# Extract the fifa ratings and write them to csv (the ones not found can be fixed manually) #
#############################################################################################
feature_df = pd.read_csv('features_dirty.csv')

# Try to open (not_)found_players.csv and initialize the dicts
ratings_found = {}
ratings_not_found = {}
try:
    found_ratings_df = pd.read_csv('found_players.csv')
    not_found_ratings_df = pd.read_csv('not_found_players.csv')

    for i in range(len(found_ratings_df)):
        ratings_found[(found_ratings_df.iloc[i, :]['Player'],
                       found_ratings_df.iloc[i, :]['Season'])] = found_ratings_df.iloc[i, :]['Rating']

    for i in range(len(not_found_ratings_df)):
        ratings_not_found[(not_found_ratings_df.iloc[i, :]['Player'],
                           not_found_ratings_df.iloc[i, :]['Season'])] = 50
except:
    pass

home_gk_ratings = []
home_def_ratings = []
home_mid_ratings = []
home_atk_ratings = []
away_gk_ratings = []
away_def_ratings = []
away_mid_ratings = []
away_atk_ratings = []
for i in range(len(feature_df)):
    record = feature_df.iloc[i, :].copy(deep=False)
    season = record['date'][2:4]

    if record['home_gk'] is not np.NaN:
        print(record['home_team'], 'Goalkeepers', record['home_gk'])
        ratings = []
        for player in ast.literal_eval(record['home_gk']):
            # First check in the dataframes if we can find the player (faster)
            if (player, season) in ratings_found:
                ratings.append((player, ratings_found[(player, season)]))
            elif (player, season) in ratings_not_found:
                ratings.append((player, ratings_not_found[(player, season)]))
            else:
                fifa_ratings = get_fifa_rating(player, season)
                if fifa_ratings[0]:
                    ratings_found[(player, season)] = fifa_ratings[1]
                else:
                    ratings_not_found[(player, season)] = fifa_ratings[1]
                ratings.append((player, fifa_ratings[1]))
        home_gk_ratings.append(ratings)
    else:
        home_gk_ratings.append(None)

    if record['home_def'] is not np.NaN:
        print(record['home_team'], 'Defenders', record['home_def'])
        ratings = []
        for player in ast.literal_eval(record['home_def']):
            # First check in the dataframes if we can find the player (faster)
            if (player, season) in ratings_found:
                ratings.append((player, ratings_found[(player, season)]))
            elif (player, season) in ratings_not_found:
                ratings.append((player, ratings_not_found[(player, season)]))
            else:
                fifa_ratings = get_fifa_rating(player, season)
                if fifa_ratings[0]:
                    ratings_found[(player, season)] = fifa_ratings[1]
                else:
                    ratings_not_found[(player, season)] = fifa_ratings[1]
                ratings.append((player, fifa_ratings[1]))
        home_def_ratings.append(ratings)
    else:
        home_def_ratings.append(None)

    if record['home_mid'] is not np.NaN:
        print(record['home_team'], 'Midfielders', record['home_mid'])
        ratings = []
        for player in ast.literal_eval(record['home_mid']):
            # First check in the dataframes if we can find the player (faster)
            if (player, season) in ratings_found:
                ratings.append((player, ratings_found[(player, season)]))
            elif (player, season) in ratings_not_found:
                ratings.append((player, ratings_not_found[(player, season)]))
            else:
                fifa_ratings = get_fifa_rating(player, season)
                if fifa_ratings[0]:
                    ratings_found[(player, season)] = fifa_ratings[1]
                else:
                    ratings_not_found[(player, season)] = fifa_ratings[1]
                ratings.append((player, fifa_ratings[1]))
        home_mid_ratings.append(ratings)
    else:
        home_mid_ratings.append(None)

    if record['home_atk'] is not np.NaN:
        print(record['home_team'], 'Strikers', record['home_atk'])
        ratings = []
        for player in ast.literal_eval(record['home_atk']):
            # First check in the dataframes if we can find the player (faster)
            if (player, season) in ratings_found:
                ratings.append((player, ratings_found[(player, season)]))
            elif (player, season) in ratings_not_found:
                ratings.append((player, ratings_not_found[(player, season)]))
            else:
                fifa_ratings = get_fifa_rating(player, season)
                if fifa_ratings[0]:
                    ratings_found[(player, season)] = fifa_ratings[1]
                else:
                    ratings_not_found[(player, season)] = fifa_ratings[1]
                ratings.append((player, fifa_ratings[1]))
        home_atk_ratings.append(ratings)
    else:
        home_atk_ratings.append(None)

    if record['away_gk'] is not np.NaN:
        print(record['away_team'], 'Goalkeepers', record['away_gk'])
        ratings = []
        for player in ast.literal_eval(record['away_gk']):
            # First check in the dataframes if we can find the player (faster)
            if (player, season) in ratings_found:
                ratings.append((player, ratings_found[(player, season)]))
            elif (player, season) in ratings_not_found:
                ratings.append((player, ratings_not_found[(player, season)]))
            else:
                fifa_ratings = get_fifa_rating(player, season)
                if fifa_ratings[0]:
                    ratings_found[(player, season)] = fifa_ratings[1]
                else:
                    ratings_not_found[(player, season)] = fifa_ratings[1]
                ratings.append((player, fifa_ratings[1]))
        away_gk_ratings.append(ratings)
    else:
        away_gk_ratings.append(None)

    if record['away_def'] is not np.NaN:
        print(record['away_team'], 'Defenders', record['away_def'])
        ratings = []
        for player in ast.literal_eval(record['away_def']):
            # First check in the dataframes if we can find the player (faster)
            if (player, season) in ratings_found:
                ratings.append((player, ratings_found[(player, season)]))
            elif (player, season) in ratings_not_found:
                ratings.append((player, ratings_not_found[(player, season)]))
            else:
                fifa_ratings = get_fifa_rating(player, season)
                if fifa_ratings[0]:
                    ratings_found[(player, season)] = fifa_ratings[1]
                else:
                    ratings_not_found[(player, season)] = fifa_ratings[1]
                ratings.append((player, fifa_ratings[1]))
        away_def_ratings.append(ratings)
    else:
        away_def_ratings.append(None)

    if record['away_mid'] is not np.NaN:
        print(record['away_team'], 'Midfielders', record['away_mid'])
        ratings = []
        for player in ast.literal_eval(record['away_mid']):
            # First check in the dataframes if we can find the player (faster)
            if (player, season) in ratings_found:
                ratings.append((player, ratings_found[(player, season)]))
            elif (player, season) in ratings_not_found:
                ratings.append((player, ratings_not_found[(player, season)]))
            else:
                fifa_ratings = get_fifa_rating(player, season)
                if fifa_ratings[0]:
                    ratings_found[(player, season)] = fifa_ratings[1]
                else:
                    ratings_not_found[(player, season)] = fifa_ratings[1]
                ratings.append((player, fifa_ratings[1]))
        away_mid_ratings.append(ratings)
    else:
        away_mid_ratings.append(None)

    if record['away_atk'] is not np.NaN:
        print(record['away_team'], 'Strikers', record['away_atk'])
        ratings = []
        for player in ast.literal_eval(record['away_atk']):
            # First check in the dataframes if we can find the player (faster)
            if (player, season) in ratings_found:
                ratings.append((player, ratings_found[(player, season)]))
            elif (player, season) in ratings_not_found:
                ratings.append((player, ratings_not_found[(player, season)]))
            else:
                fifa_ratings = get_fifa_rating(player, season)
                if fifa_ratings[0]:
                    ratings_found[(player, season)] = fifa_ratings[1]
                else:
                    ratings_not_found[(player, season)] = fifa_ratings[1]
                ratings.append((player, fifa_ratings[1]))
        away_atk_ratings.append(ratings)
    else:
        away_atk_ratings.append(None)

feature_df['home_gk'] = home_gk_ratings
feature_df['home_def'] = home_def_ratings
feature_df['home_mid'] = home_mid_ratings
feature_df['home_atk'] = home_atk_ratings
feature_df['away_gk'] = away_gk_ratings
feature_df['away_def'] = away_def_ratings
feature_df['away_mid'] = away_mid_ratings
feature_df['away_atk'] = away_atk_ratings
print(feature_df['home_gk'])
print(feature_df['home_def'])
print(feature_df['home_mid'])
print(feature_df['home_atk'])
print(feature_df['away_gk'])
print(feature_df['away_def'])
print(feature_df['away_mid'])
print(feature_df['away_atk'])
ratings_found_lists = []
for player, season in ratings_found:
    ratings_found_lists.append([player, season, ratings_found[(player, season)]])
ratings_found_df = pd.DataFrame(ratings_found_lists, columns=['Player', 'Season', 'Rating'])
ratings_found_df.to_csv('found_players.csv', index=False)


ratings_not_found_list = []
for player, season in ratings_not_found:
    ratings_not_found_list.append([player, season])
ratings_found_df = pd.DataFrame(ratings_not_found_list, columns=['Player', 'Season'])
ratings_found_df.to_csv('not_found_players.csv', index=False)

