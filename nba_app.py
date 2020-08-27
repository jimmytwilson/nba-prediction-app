import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import OrderedDict


#title
st.image("https://cdn.nba.net/nba-drupal-prod/2017-08/Twitter-seo-image-NBA-logman.jpg", use_column_width=True)
st.title("NBA Data")
st.markdown(
"""
## This is a Visualisation of NBA Data
""")

#Getting the data in a Pandas Dataframe
@st.cache(persist=True)
def load_data():
    data = pd.read_csv("nba_all_elo.csv")
    return data

data = load_data()


#MACHINE LEARNING
ml_dataframe = data.loc[(data["year_id"] == 2015) | (data["year_id"] == 2014)]

fran_id_data = ml_dataframe["fran_id"]
d = dict([(y,x+1) for x,y in enumerate(sorted(set(fran_id_data)))])
fran_id_list = [d[x] for x in fran_id_data]

opp_fran_data = ml_dataframe["opp_fran"]
j = dict([(y,x+1) for x,y in enumerate(sorted(set(opp_fran_data)))])
opp_fran_list = [j[x] for x in opp_fran_data]

final_dict = {'fran_id':fran_id_list, 'opp_fran':opp_fran_list}

game_result_data = ml_dataframe["game_result"]
a = dict([(y,x+1) for x,y in enumerate(sorted(set(game_result_data)))])
game_result_list = [a[x] for x in game_result_data]
final_targ_dict = {'game_result':game_result_list}

X = pd.DataFrame(final_dict)
y = pd.DataFrame(final_targ_dict)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2,random_state = 4000)

logistic_regression_model = LogisticRegression()

logistic_regression_model.fit(X_train, y_train.values.ravel())

#Commented out code below was used to test the clf
#ml_test_data = OrderedDict([("fran_id", 4), ("opp_fran", 8)])
#ml_test_data = pd.Series(ml_test_data).values.reshape(1,-1)

#print(logistic_regression_model.predict(ml_test_data))
#print(logistic_regression_model.predict_proba(ml_test_data))
#print(logistic_regression_model.score(X_test,y_test))


#Getting all team names
teams = data.loc[data["year_id"] == 2015]
team_list_sorted = sorted(set(teams.fran_id))

st.sidebar.header("Select Input Parameters")
team_select = st.sidebar.selectbox(
    'What team would you like to see?', team_list_sorted)


#Selected team to show the Data
st.write('Team selected:', team_select)

selected_team_data = data[data["fran_id"] == team_select]


#Getting the wins and loses data to display in a line graph for selected team
win_by_year = selected_team_data.loc[data["game_result"] == "W", "year_id"].value_counts().sort_index().rename("Wins")
lose_by_year = selected_team_data.loc[data["game_result"] == "L", "year_id"].value_counts().sort_index().rename("Loses")
number_of_games = selected_team_data["year_id"].value_counts().sort_index().rename("Games")

wins_and_loses = pd.concat([win_by_year, lose_by_year], axis=1)

#Sidebar input for Wins/Loses
start_year = wins_and_loses.index.values.tolist()
sel_start_length = st.sidebar.slider('Choose the starting year:', start_year[0], 2014, start_year[0])
sel_end_length = st.sidebar.slider('Choose the ending year:', sel_start_length, 2015, 2015)

begin_win_lose = start_year.index(sel_start_length)
end_win_lose = start_year.index(sel_end_length)
st.markdown(
"""
#### Wins vs Loses
""")
st.line_chart(wins_and_loses[begin_win_lose:end_win_lose + 1])
st.markdown(
"""
#### Games per season (inc. Playoffs)
""")
st.bar_chart(number_of_games[begin_win_lose:end_win_lose + 1])

#Checkbox for showing the teams raw data
if st.checkbox("Show Team's Raw Data"):
    st.write(selected_team_data)


#Display who will win from 2 teams
team_list_sorted.insert(0, "Choose")

def user_input_features():
    team_select_home_team = st.sidebar.selectbox(
        'Choose the HOME team:', team_list_sorted)
    team_select_away_team = st.sidebar.selectbox(
        'Choose the AWAY team:', team_list_sorted)
    if (team_list_sorted.index(team_select_home_team) == 0) and (team_list_sorted.index(team_select_away_team) == 0):
        return_statement = "Please input teams in the sidebar"
        return return_statement
    if team_list_sorted.index(team_select_home_team) == team_list_sorted.index(team_select_away_team):
        return_statement_same = "Can't have the same team"
        return return_statement_same
    if (team_list_sorted.index(team_select_home_team) > 0) and (team_list_sorted.index(team_select_away_team) > 0):
        ml_test_data = OrderedDict([("fran_id", team_list_sorted.index(team_select_home_team)), ("opp_fran", team_list_sorted.index(team_select_away_team))])
        ml_test_data = pd.Series(ml_test_data).values.reshape(1,-1)
        return ml_test_data
    else:
        return_statement = "Please input teams"
        return return_statement

st.sidebar.header("Predict a match!")
df = user_input_features()
st.header('Win prediction')


if (df == "Can't have the same team") or (df == "Please input teams") or (df == "Please input teams in the sidebar"):
    st.write(df)
else:
    prediction = logistic_regression_model.predict(df)
    prediction_proba = logistic_regression_model.predict_proba(df)

    winner_team = "Select Team"
    if (prediction_proba[0][0] > prediction_proba[0][1]):
        winner_team = team_list_sorted[df[0][0]]
    else:
        winner_team = team_list_sorted[df[0][1]]

    st.subheader('Nice choice!')
    st.write("...and the winners are the:  " + winner_team)
    
    if winner_team == "Bucks":
        st.image("https://1000logos.net/wp-content/uploads/2018/01/Milwaukee-Bucks-Logo-768x432.png", use_column_width=True)
    if winner_team == "Bulls":
        st.image("https://1000logos.net/wp-content/uploads/2016/11/Chicago-Bulls-Logo-768x693.png", use_column_width=True)
    if winner_team == "Cavaliers":
        st.image("https://1000logos.net/wp-content/uploads/2017/08/CAVS-Logo-768x455.png", use_column_width=True)
    if winner_team == "Celtics":
        st.image("https://1000logos.net/wp-content/uploads/2016/10/Boston-Celtics-Logo.png", use_column_width=True)
    if winner_team == "Clippers":
        st.image("https://1000logos.net/wp-content/uploads/2017/12/Los-Angeles-Clippers-Logo-768x432.png", use_column_width=True)
    if winner_team == "Grizzlies":
        st.image("https://1000logos.net/wp-content/uploads/2018/05/Memphis-Grizzlies-Logo-768x432.png", use_column_width=True)
    if winner_team == "Hawks":
        st.image("https://1000logos.net/wp-content/uploads/2017/12/atlanta-hawks-logo-768x432.png", use_column_width=True)
    if winner_team == "Heat":
        st.image("https://1000logos.net/wp-content/uploads/2017/04/Miami-Heat-Logo-768x514.png", use_column_width=True)
    if winner_team == "Hornets":
        st.image("https://1000logos.net/wp-content/uploads/2018/05/Charlotte-Hornets-Logo-768x432.png", use_column_width=True)
    if winner_team == "Jazz":
        st.image("https://1000logos.net/wp-content/uploads/2018/05/Utah-Jazz-Logo-768x432.png", use_column_width=True)
    if winner_team == "Kings":
        st.image("https://1000logos.net/wp-content/uploads/2017/12/sacramento-kings-logo-768x432.png", use_column_width=True)
    if winner_team == "Knicks":
        st.image("https://1000logos.net/wp-content/uploads/2017/10/new-york-knicks-logo-768x590.png", use_column_width=True)
    if winner_team == "Lakers":
        st.image("https://1000logos.net/wp-content/uploads/2017/03/Lakers-Logo.png", use_column_width=True)
    if winner_team == "Magic":
        st.image("https://1000logos.net/wp-content/uploads/2017/08/Orlando-Magic-Logo.png", use_column_width=True)
    if winner_team == "Mavericks":
        st.image("https://1000logos.net/wp-content/uploads/2018/05/Dallas-Mavericks-Logo-768x432.png", use_column_width=True)
    if winner_team == "Nets":
        st.image("https://1000logos.net/wp-content/uploads/2018/05/Brooklyn-Nets-Logo-768x432.png", use_column_width=True)
    if winner_team == "Nuggets":
        st.image("https://1000logos.net/wp-content/uploads/2018/05/Denver-Nuggets-Logo-768x432.png", use_column_width=True)
    if winner_team == "Pacers":
        st.image("https://1000logos.net/wp-content/uploads/2018/05/Indiana-Pacers-Logo-768x432.png", use_column_width=True)
    if winner_team == "Pelicans":
        st.image("https://1000logos.net/wp-content/uploads/2018/05/New-Orleans-Pelicans-Logo-768x432.png", use_column_width=True)
    if winner_team == "Pistons":
        st.image("https://1000logos.net/wp-content/uploads/2017/08/Detroit-Pistons-Logo-768x563.png", use_column_width=True)
    if winner_team == "Raptors":
        st.image("https://1000logos.net/wp-content/uploads/2018/05/Toronto-Raptors-Logo-768x432.png", use_column_width=True)
    if winner_team == "Rockets":
        st.image("https://1000logos.net/wp-content/uploads/2017/10/houston-rockets-logo-768x384.png", use_column_width=True)
    if winner_team == "Sixers":
        st.image("https://1000logos.net/wp-content/uploads/2016/10/Philadelphia-76ers-logo.png", use_column_width=True)
    if winner_team == "Spurs":
        st.image("https://1000logos.net/wp-content/uploads/2017/11/San-Antonio-Spurs-Logo-768x424.png", use_column_width=True)
    if winner_team == "Suns":
        st.image("https://1000logos.net/wp-content/uploads/2018/05/Phoenix-Suns-Logo-768x432.png", use_column_width=True)
    if winner_team == "Thunder":
        st.image("https://1000logos.net/wp-content/uploads/2018/05/Oklahoma-City-Thunder-Logo-768x432.png", use_column_width=True)
    if winner_team == "Timberwolves":
        st.image("https://1000logos.net/wp-content/uploads/2017/07/Timberwolves-Logo-768x689.png", use_column_width=True)
    if winner_team == "Trailblazers":
        st.image("https://1000logos.net/wp-content/uploads/2018/05/Portland-Trail-Blazers-Logo-768x432.png", use_column_width=True)
    if winner_team == "Warriors":
        st.image("https://1000logos.net/wp-content/uploads/2018/01/golden-state-warriors-logo-768x432.png", use_column_width=True)
    if winner_team == "Wizards":
        st.image("https://1000logos.net/wp-content/uploads/2018/05/Washington-Wizards-Logo-768x432.png", use_column_width=True)
    

    my_dict = {'Home': [prediction_proba[0][0] * 100], 'Away': [prediction_proba[0][1] * 100]}
    prob_df = pd.DataFrame.from_dict(my_dict).rename(index={0: '%'})

    st.subheader('Prediction Probability')
    st.write(prob_df)

    #Below showing the prediction
    #st.subheader('Prediction')
    #st.write(prediction)

    #Balloons every render because they are fun
    st.balloons()


#Final checkbox to show all the raw data. Click with caution, takes ages to render
if st.checkbox('Show all Raw Data'):
    'data', data