import urllib

from lxml import etree, html
import pandas as pd


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
