# Visit http://127.0.0.1:8050/ in your web browser.

## Installs
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pymssql
import pickle
import pandas as pd
import numpy as np
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

## Get Data
# Connect to SQL server
database = "group2"
username = "group2user"
password = "everythingIsAwesome!"
server = "database2108.database.windows.net"

conn = pymssql.connect(server, username, password, database)
cursor = conn.cursor()

# Create dataframes from server tables
census_table = "SELECT * FROM dbo.census_data" # Census data
census = pd.read_sql(census_table, conn)

stock_table = "SELECT * FROM dbo.stock_data" # Stock data
stock = pd.read_sql(stock_table, conn)

stock['Date'] = pd.to_datetime(stock['Date'], format= '%Y-%m-%d')
monthly_stock = stock.groupby(['Ticker_ID', pd.Grouper(key = 'Date', freq = 'M')]).mean() # Monthly averages
industry_stock = stock.groupby(['Ticker_ID', pd.Grouper(key = 'Date', freq = 'M')]).sum() # Monthly sum of daily percent changes
industry_stock_average = industry_stock.groupby([pd.Grouper(level = 1, freq = 'M')]).mean() # Monthly average percent change

ticker_name = {'1': 'JPM', '2': 'SNV', '3': 'NYCB', '4': 'DFS', '5': 'BOH', '6': 'GS'} # Change legend values in graph to ticker name
industry_stock_average['Color'] = np.where(industry_stock_average['Perc'] < 0, '#5C5C5C', '#0A0A0A') # Change graph colors

sp_table = "SELECT * FROM dbo.sp_data" # S&P data
sp = pd.read_sql(sp_table, conn)

sp['Date'] = pd.to_datetime(sp['Date'], format= '%Y-%m-%d')
monthly_sp = sp.groupby([pd.Grouper(key = 'Date', freq = 'M')]).sum() # Monthly average percent change

# Machine learning model (see 'Capstone Load ML.ipynb' in Code file for full comments)
query = 'SELECT * FROM dbo.stock_table'
df = pd.read_sql(query, conn)

df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month
df['Day'] = pd.to_datetime(df['Date']).dt.day

df['C_weighted'] = df.groupby('Ticker')['Close'].ewm(alpha=0.5, adjust=True).mean().reset_index(0, drop = True)

df['comparison'] = np.where(df['Perc'] > df['SP_perc'], 1, 0)
df.loc[df['Ticker'] == 'JPM', 'Market'] = 'Large'
df.loc[df['Ticker'] == 'GS', 'Market'] = 'Large'
df.loc[df['Ticker'] == 'NYCB', 'Market'] = 'Small'
df.loc[df['Ticker'] == 'BOH', 'Market'] = 'Small'
df.loc[df['Ticker'] == 'DFS', 'Market'] = 'Mid'
df.loc[df['Ticker'] == 'SNV', 'Market'] = 'Mid'

df['Perc_diff'] = df['Perc'] - df['SP_perc'] # create new column made from difference between Perc and SP_perc
df['Diff1'] = df.groupby(['Ticker'])['Perc_diff'].shift(1) # get the previous day's difference between Perc and SP_perc
df['Diff2'] = df.groupby(['Ticker'])['Perc_diff'].shift(2) # get 2 day's previous difference between Perc and SP_perc
df['Diff3'] = df.groupby(['Ticker'])['Perc_diff'].shift(3) # get 3 day's previous difference between Perc and SP_perc
df['Diff4'] = df.groupby(['Ticker'])['Perc_diff'].shift(4) # get 4 day's previous difference between Perc and SP_perc
df['Diff5'] = df.groupby(['Ticker'])['Perc_diff'].shift(5) # get 5 day's previous difference between Perc and SP_perc

df_ml = df[30:] 

y = df_ml.comparison
X = df_ml.drop(['comparison'], axis = 1)
X = X.drop(['SP_perc'], axis = 1) # drop SP_perc from model

dummies = pd.get_dummies(X['Market'], drop_first = True) # get dummies for Market column
X = pd.merge(X, dummies, left_index = True, right_index = True)

X = X.drop(['Ticker', 'Ticker_ID', 'Market', 'Date'], axis = 1) # drop non-numeric columns
X = X.drop(['High', 'Low', 'Close', 'Open'], axis = 1) 
X = X.drop(['SP_Close', 'SP_High', 'SP_Low', 'SP_Open'], axis = 1)
X = X.drop(['Perc_diff'], axis = 1) # drop Perc_diff from model
X = X.drop(['Volume'], axis = 1) # drop Volume
X = X.drop(['Year'], axis = 1) # drop Year

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

with open('/Users/Nicole/Downloads/Capstone_ML2' , 'rb') as f: # path from local device
    lr = pickle.load(f)

clf = lr.fit(X_train, y_train)
clf.score(X_test, y_test)

y_true = list(y_test)
y_pred = clf.predict(X_test)

logit_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])


## Visualizations
def census_fig():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure = px.histogram(
                        census,
                        title = 'Total Revenue by Industry Category (2017)',
                        x = 'LABELNAME',
                        y = 'RCPTOT',
                    ).update_layout(
                        title_x = 0.5,
                        xaxis_title_text = 'Name',
                        yaxis_title_text = 'Total Revenue (in USD)',
                        xaxis = dict(gridcolor = '#EEEEEE', showticklabels = False),
                        yaxis = dict(gridcolor = '#EEEEEE'),
                        font = dict(size = 10),
                        paper_bgcolor = '#FFFFFF',
                        plot_bgcolor = '#FFFFFF',
                        margin = dict(pad = 10)
                    ).update_traces(
                        marker_color = '#0A0A0A')
                )
            ])
        )
    ])

def stock_fig():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure = px.line(
                        monthly_stock,
                        title = 'Monthly Closing Prices (2007–2021)',
                        x = monthly_stock.index.get_level_values(1),
                        y = 'Close',
                        labels = {'x': 'Date', 'Close': 'Average Closing Price (in USD)', 'color': ' '},
                        color = monthly_stock.index.get_level_values(0),
                        color_discrete_sequence = ['#9B2226', '#EE9B00', '#00333D', '#CA6702', '#94D2BD', '#641718']
                    ).update_layout(
                        title_x = 0.5,
                        xaxis = dict(gridcolor = '#EEEEEE'),
                        yaxis = dict(gridcolor = '#EEEEEE'),
                        legend = dict(orientation = 'h', xanchor = 'right', x = 1, yanchor = 'top', y = 1),
                        font = dict(size = 10),
                        paper_bgcolor = '#FFFFFF',
                        plot_bgcolor = '#FFFFFF',
                        width = 800,
                        margin = dict(pad = 10)
                    ).update_traces(
                        line = dict(width = 1)
                    ).for_each_trace(lambda t: t.update(
                        name = ticker_name[t.name],
                        legendgroup = ticker_name[t.name],
                        hovertemplate = t.hovertemplate.replace(t.name, ticker_name[t.name]))
                    )
                )
            ])
        )
    ])

def stock_fig2():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure = go.Figure(
                    ).add_trace(
                        go.Scatter(
                            name = 'S&P 500',
                            x = monthly_sp.index.get_level_values(0),
                            y = monthly_sp.SP_perc,
                            marker_color = '#EE9B00',
                            line = dict(width = 1))
                    ).add_trace(
                        go.Bar(
                            name = 'Industry Average',
                            x = monthly_stock.index.get_level_values(1),
                            y = industry_stock_average.Perc,
                            marker_color = industry_stock_average['Color'])
                    ).update_layout(
                        title = 'Monthly Percent Change in Closing Prices Compared to the S&P 500 (2007–2021)',
                        xaxis_title = 'Date',
                        yaxis_title = 'Average Percent Change',
                        title_x = 0.5,
                        xaxis = dict(gridcolor = '#EEEEEE'),
                        yaxis = dict(gridcolor = '#EEEEEE', tickformat = ',.0%'),
                        legend = dict(orientation = 'h', xanchor = 'right', x = 1, yanchor = 'top', y = 1),
                        font = dict(size = 10),
                        paper_bgcolor = '#FFFFFF',
                        plot_bgcolor = '#FFFFFF',
                        width = 800,
                        margin = dict(pad = 10)
                    )
                )
            ])
        )
    ])

def ml_fig():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure = px.scatter(
                        title = 'Logistic Regression of Machine Learning Testing Data',
                        x = X_test.Perc,
                        y = y_test,
                        labels = dict(x = 'Percent Change of Closing Price from Previous Day', y = 'Outcome'),
                        trendline = 'lowess'
                    ).update_layout(
                        title_x = 0.5,
                        xaxis = dict(gridcolor = '#EEEEEE'),
                        yaxis = dict(gridcolor = '#EEEEEE'),
                        font = dict(size = 10),
                        paper_bgcolor = '#FFFFFF',
                        plot_bgcolor = '#FFFFFF',
                        margin = dict(pad = 10)
                    ).update_traces(
                        marker_color = '#0A0A0A',
                        line = dict(width = 1)
                    )
                )
            ])
        )
    ])

def ml_fig2():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure = px.line(
                        title = 'Receiver Operating Characteristic',
                        x = fpr,
                        y = tpr,
                        labels = dict(x = 'False Positive Rate', y = 'True Positive Rate'),
                        color_discrete_sequence = ['#EE9B00']
                    ).add_shape(
                        type = 'line',
                        line = dict(width = 1, dash = 'dash'),
                        x0 = 0,
                        x1 = 1, 
                        y0 = 0, 
                        y1 = 1
                    ).update_layout(
                        title_x = 0.5,
                        xaxis = dict(gridcolor = '#EEEEEE'),
                        yaxis = dict(gridcolor = '#EEEEEE'),
                        font = dict(size = 10),
                        paper_bgcolor = '#FFFFFF',
                        plot_bgcolor = '#FFFFFF',
                        margin = dict(pad = 10)
                    ).update_traces(
                        marker = dict(color = '#0A0A0A'),
                        line = dict(width = 1)
                    )
                )
            ])
        )
    ])

# Dashboard
fontawesome_stylesheet = 'https://use.fontawesome.com/releases/v5.8.1/css/all.css'
app = dash.Dash(external_stylesheets = [dbc.themes.BOOTSTRAP, fontawesome_stylesheet])

app.layout = html.Div([
    html.H2('The Finance Industry and the Stock Market', className = 'card-title'),

    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.P('''Studying and learning from financial trends is a key aspect of financial management. 
                    The stock market experiences changes throughout the economic cycle and can be affected by major 
                    events. The finance industry is one of the largest industries trading in the stock market. 
                    Drawing from a small sample of financial institutions with a range of market capitalization 
                    sizes as well as the S&P 500, we use historical stock price data and machine learning to 
                    understand and explain the market and its changes throughout the past and potential changes in 
                    the future.''',
                    className = 'card-text'),
                    
                    html.P('''We sourced our data from the United States Census Bureau and The Wall Street Journal.
                    The census data provides a statistical summary on the finance industry and the data from The 
                    Wall Street Journal provides the opening, closing, high, and low prices for the specified 
                    companies as well as the S&P 500 along with the volume traded each day from July 2, 2007 until 
                    October 27, 2021.''',
                    className = 'card-text')
                ]),
            ),
            width = 4 
        ),
        dbc.Col([
            dbc.Row([
                html.H4('Industry Overview', className = 'card-text')
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.P('''Here is an overview of the finance industry, with the total number of 
                            establishments and total revenue displayed below as well as a breakdown of the total 
                            revenue by sub-category to the right. Hover over a bar to show the sub-category and 
                            its contribution to the total revenue.''', 
                            className = 'card-descriptor')
                        ])
                    )
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4('712,730', className = 'card-text'),
                            html.P('Total Establishments', className = 'card-descriptor')
                        ])
                    )
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4('$4.34B', className = 'card-text'),
                            html.P('Total Revenue', className = 'card-descriptor')
                        ])
                    )
                )
            ]),
        ]),
        dbc.Col([census_fig()], width = 5)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Row([stock_fig()]),
            dbc.Row([stock_fig2()]),
        ]),
        dbc.Col([
            dbc.Row([
                html.H4('State of the Market', className = 'card-text')
            ]),
            dbc.Row(
                dbc.Card(
                    dbc.CardBody([
                        html.P('''Left: We use stock price data to compare companies within the finance industry 
                        with each other as well as with the S&P 500 index. You can filter the selections by clicking
                        on the ticker symbol.''',
                        className = 'card-text'),

                        html.P('''Bottom: Below are the results from our machine learning model to explain how
                        accurately we can predict whether a stock will perform better or worse than the S&P 500.''',
                        className = 'card-text')
                    ])
                )
            ),
            dbc.Row([ml_fig()]),
            dbc.Row([ml_fig2()])
        ]),  
    ]),
])

if __name__ == '__main__':
    app.run_server(debug = True)
