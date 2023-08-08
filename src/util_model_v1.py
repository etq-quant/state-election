import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor

def compute_majority(tdf):
    majority = tdf.apply(lambda row: row.nlargest(1).iloc[-1] - row.nlargest(2).iloc[-1], axis=1)
    return majority
    

def run_election_model(state, lost_faith, bn_to_ph, ph_to_bn):
    # state - one of ['SELANGOR', 'PULAU_PINANG', 'NEGERI_SEMBILAN', 'KEDAH']
    # lost_faith = 0.05
    # bn_to_ph = 0.3
    # ph_to_bn = 0.5
    
    file_mapper = {"SELANGOR": "SELANGOR_2018_DUN_RESULTS", }
    states = ['SELANGOR', 'PULAU_PINANG', 'NEGERI_SEMBILAN', 'KEDAH']
    cols = ['PARLIAMENTARY CODE', 'STATE', 'MALAY (%)', 'CHINESE (%)', 'INDIANS (%)', 'ORANG ASLI (%)', 'BUMIPUTERA SABAH (%)', 'BUMIPUTERA SARAWAK (%)', 'OTHERS (%)']
    df_c18 = pd.read_csv('data/MALAYSIA_2013_PARLIAMENTARY_COMPOSITION.csv')[cols].set_index('PARLIAMENTARY CODE')
    df_c18['MALAY_2018'] = df_c18['MALAY (%)']/100

    cols = ['PARLIAMENTARY CODE', 'STATE', 'MALAY (%)', 'CHINESE (%)', 'INDIANS (%)', 'ORANG ASLI (%)', 'BUMIPUTERA SABAH (%)', 'BUMIPUTERA SARAWAK (%)', 'OTHERS (%)', 'URBAN-RURAL CLASSIFICATION (2022)']
    df_c22 = pd.read_csv('data/MALAYSIA_2022_PARLIAMENT_COMPOSITION.csv')[cols].set_index('PARLIAMENTARY CODE')
    df_c22['MALAY_2022'] = df_c22['MALAY (%)']/100
    df_c22['URBAN-RURAL CLASSIFICATION (2022)'] = df_c22['MALAY (%)']/100

    cols = ['PARLIAMENTARY CODE', 'STATE', 'TOTAL ELECTORATE', 'TOTAL VALID VOTES', 'BN', 'BN CANDIDATE VOTE', 'PH', 'PH CANDIDATE VOTE', 'GS', 'GS CANDIDATE VOTE']
    df_p18 = pd.read_csv('data/MALAYSIA_2018_PARLIAMENTARY_RESULTS.csv')[cols].set_index('PARLIAMENTARY CODE')

    cols = ['PARLIAMENTARY CODE',  'TOTAL ELECTORATE', 'TOTAL VALID VOTES', 'BN', 'BN VOTE', 'PH', 'PH VOTE', 'PN', 'PN VOTE']
    df_p22 = pd.read_csv('data/MALAYSIA_GE15_PARLIAMENT_ELECTIONS_v25122022.csv')[cols].set_index('PARLIAMENTARY CODE')


    try:
        cols = ['PARLIAMENTARY CODE', 'PARLIAMENTARY NAME', 'STATE CONSTITUENCY CODE', 'STATE CONSTITUENCY NAME', 'TOTAL ELECTORATE', 'TOTAL VALID VOTES', 'BN', 'BN CANDIDATE VOTE', 'PH', 'PH CANDIDATE VOTE', 'GS', 'GS CANDIDATE VOTE']
        df_s18 = pd.read_csv(f'data/{state}_2018_DUN_RESULTS.csv')[cols].set_index('PARLIAMENTARY CODE')
        df_s18.columns = ['PARLIAMENTARY NAME', 'STATE CONSTITUENCY CODE', 'STATE CONSTITUENCY NAME', 'TOTAL ELECTORATE', 'TOTAL VALID VOTES', 'BN', 'BN VOTE', 'PH', 'PH VOTE', 'GS', 'GS VOTE']
    except:
        cols = ['PARLIAMENTARY CODE', 'PARLIAMENTARY NAME', 'STATE CONSTITUENCY CODE', 'STATE CONSTITUENCY NAME', 'TOTAL ELECTORATE', 'TOTAL VALID VOTES', 'BN', 'BN VOTE', 'PH', 'PH VOTE', 'GS', 'GS VOTE']
        df_s18 = pd.read_csv(f'data/{state}_2018_DUN_RESULTS.csv')[cols].set_index('PARLIAMENTARY CODE')

    if state == 'NEGERI_SEMBILAN':
        print('changing')
        u = df_s18['STATE CONSTITUENCY CODE'] == 'N.27'
        df_s18.loc[u, 'BN VOTE'] = 10397
        df_s18.loc[u, 'PH VOTE'] = 5887
        df_s18.loc[u, 'GS VOTE'] = 0
        df_s18.loc[u, 'TOTAL VALID VOTES'] = 16446

    df_p18['P18_TOTAL_VOTES'] = df_p18['TOTAL VALID VOTES']
    df_p18['BN_2018'] = df_p18['BN CANDIDATE VOTE'].fillna(0)/df_p18['TOTAL VALID VOTES']
    df_p18['PH_2018'] = df_p18['PH CANDIDATE VOTE'].fillna(0)/df_p18['TOTAL VALID VOTES']
    df_p18['PN_2018'] = df_p18['GS CANDIDATE VOTE'].fillna(0)/df_p18['TOTAL VALID VOTES']

    df_p22['P22_TOTAL_VOTES'] = df_p22['TOTAL VALID VOTES']
    df_p22['BN_2022'] = df_p22['BN VOTE'].fillna(0)/df_p22['TOTAL VALID VOTES']
    df_p22['PH_2022'] = df_p22['PH VOTE'].fillna(0)/df_p22['TOTAL VALID VOTES']
    df_p22['PN_2022'] = df_p22['PN VOTE'].fillna(0)/df_p22['TOTAL VALID VOTES']

    df_s18['DUN_BN_2018'] = df_s18['BN VOTE'].fillna(0)/df_s18['TOTAL VALID VOTES']
    df_s18['DUN_PH_2018'] = df_s18['PH VOTE'].fillna(0)/df_s18['TOTAL VALID VOTES']
    df_s18['DUN_PN_2018'] = df_s18['GS VOTE'].fillna(0)/df_s18['TOTAL VALID VOTES']
    df_s18['S18_TOTAL_VOTES'] = df_s18['TOTAL VALID VOTES']

    df1 = df_c18[['MALAY_2018', ]]
    df2 = df_c22[['MALAY_2022', 'URBAN-RURAL CLASSIFICATION (2022)']]
    df3 = df_p18[['BN_2018', 'PH_2018', 'PN_2018', 'P18_TOTAL_VOTES']]
    df4 = df_p22[['BN_2022', 'PH_2022', 'PN_2022', 'P22_TOTAL_VOTES']]
    df_dun = df_s18[['PARLIAMENTARY NAME', 'STATE CONSTITUENCY CODE', 'STATE CONSTITUENCY NAME', 'DUN_BN_2018', 'DUN_PH_2018', 'DUN_PN_2018', 'S18_TOTAL_VOTES']]


    fdf = df_dun.join(df1).join(df2).join(df3).join(df4)
    fdf['CHG_P_TOTAL_VOTES'] = fdf['P22_TOTAL_VOTES'] - fdf['P18_TOTAL_VOTES']
    fdf['NEW_P_MALAY_VOTES'] = (fdf['CHG_P_TOTAL_VOTES']*(fdf['MALAY_2022'])).astype(int)
    fdf['NEW_P_MALAY_VOTES_PCT'] = (fdf['CHG_P_TOTAL_VOTES']*(fdf['MALAY_2022']))/fdf['P22_TOTAL_VOTES']
    fdf['NEW_P_NONMALAY_VOTES'] = (fdf['CHG_P_TOTAL_VOTES']*(1-fdf['MALAY_2022'])).astype(int)

    # assume number votes increase proportionately for each state seat based on increase in number of votes for federal seat
    fdf['S22_TOTAL_VOTES'] = (fdf['S18_TOTAL_VOTES'] * (fdf['P22_TOTAL_VOTES']/fdf['P18_TOTAL_VOTES'])).astype(int)

    fdf['CHG_S_TOTAL_VOTES'] = fdf['S22_TOTAL_VOTES'] - fdf['S18_TOTAL_VOTES']
    fdf['NEW_S_MALAY_VOTES'] = (fdf['CHG_S_TOTAL_VOTES']*(fdf['MALAY_2022'])).astype(int)
    fdf['NEW_S_MALAY_VOTES_PCT'] = (fdf['CHG_S_TOTAL_VOTES']*(fdf['MALAY_2022']))/fdf['S22_TOTAL_VOTES']
    fdf['NEW_S_NONMALAY_VOTES'] = (fdf['CHG_S_TOTAL_VOTES']*(1-fdf['MALAY_2022'])).astype(int)

    X = fdf[['URBAN-RURAL CLASSIFICATION (2022)']]
    drop_binary_enc = OneHotEncoder(drop='if_binary').fit(X)
    one_hot_urban = drop_binary_enc.transform(X).toarray()
    fdf['Urban'] = one_hot_urban[:, 2].astype(int)
    fdf['Rural'] = one_hot_urban[:, 0].astype(int)

    # Model 1: predict PN support 
     # features: previous support parliament results (2018), %Malay 2018, %Malay 2022, % new malay voters, urban/semi-urban classification (Urban, Rural one hot)
     # target: latest support parliament results (2022)    
    x_cols = ['PN_2018', 'MALAY_2018', 'MALAY_2022', 'NEW_P_MALAY_VOTES_PCT', 'Urban', 'Rural']
    y_col = 'PN_2022'
    X_train = fdf[x_cols]
    y_train = fdf[y_col]

    # model = HuberRegressor()
    model = RandomForestRegressor()
    model.fit(X_train.values, y_train)
    y_pred = model.predict(X_train.values)

    fdf['yp_PN_2022'] = y_pred

    x_dun_cols = ['DUN_PN_2018', 'MALAY_2018', 'MALAY_2022', 'NEW_P_MALAY_VOTES_PCT', 'Urban', 'Rural']
    X_test = fdf[x_dun_cols]
    y_pred_test = model.predict(X_test.values)

    fdf['yp_DUN_PN_2022'] = y_pred_test
    fdf['imp_DUN_BN_2022'] = fdf['DUN_BN_2018']*(1-fdf['yp_DUN_PN_2022'])/fdf[['DUN_PH_2018', 'DUN_BN_2018']].sum(axis=1)
    fdf['imp_DUN_PH_2022'] = fdf['DUN_PH_2018']*(1-fdf['yp_DUN_PN_2022'])/fdf[['DUN_PH_2018', 'DUN_BN_2018']].sum(axis=1)

    kdf = fdf[[ 'PARLIAMENTARY NAME', 'STATE CONSTITUENCY CODE', 'STATE CONSTITUENCY NAME']].copy()
    
    kdf['FED_PN_2018'] = fdf['PN_2018']
    kdf['FED_PH_2018'] = fdf['PH_2018']
    kdf['FED_BN_2018'] = fdf['BN_2018']
    kdf['Majority_FED_2018'] =  compute_majority(kdf[['FED_PN_2018', 'FED_PH_2018', 'FED_BN_2018']])
    kdf['Winner_FED_2018'] = kdf[['FED_PN_2018', 'FED_PH_2018', 'FED_BN_2018']].idxmax(axis=1).apply(lambda x:x.split("_")[1])
    
    kdf['FED_PN_2022'] = fdf['PN_2022']
    kdf['FED_PH_2022'] = fdf['PH_2022']
    kdf['FED_BN_2022'] = fdf['BN_2022']
    kdf['Majority_FED_2022'] =  compute_majority(kdf[['FED_PN_2022', 'FED_PH_2022', 'FED_BN_2022']])
    kdf['Winner_FED_2022'] = kdf[['FED_PN_2022', 'FED_PH_2022', 'FED_BN_2022']].idxmax(axis=1).apply(lambda x:x.split("_")[1])
    
    kdf['NEW_MALAY_VOTES%'] = fdf['NEW_P_MALAY_VOTES_PCT']
    kdf['Urban_Rural'] = fdf['URBAN-RURAL CLASSIFICATION (2022)']
    
    kdf['DUN_PN_2018'] = fdf['DUN_PN_2018']
    kdf['DUN_PH_2018'] = fdf['DUN_PH_2018']
    kdf['DUN_BN_2018'] = fdf['DUN_BN_2018']
    kdf['Majority_DUN_2018'] =  compute_majority(kdf[['DUN_PN_2018', 'DUN_PH_2018', 'DUN_BN_2018']])
    kdf['Winner_DUN_2018'] = kdf[['DUN_PN_2018', 'DUN_PH_2018', 'DUN_BN_2018']].idxmax(axis=1).apply(lambda x:x.split("_")[1])
    
    kdf['MODEL_DUN_PN_2022'] = fdf['yp_DUN_PN_2022']
    kdf['MODEL_DUN_PH_2022'] = fdf['imp_DUN_PH_2022']
    kdf['MODEL_DUN_BN_2022'] = fdf['imp_DUN_BN_2022']
    kdf['Majority_MODEL_DUN_2022'] =  compute_majority(kdf[['MODEL_DUN_PN_2022', 'MODEL_DUN_PH_2022', 'MODEL_DUN_BN_2022']])
    kdf['Winner_MODEL_DUN_2022'] = kdf[['MODEL_DUN_PN_2022', 'MODEL_DUN_PH_2022', 'MODEL_DUN_BN_2022']].idxmax(axis=1).apply(lambda x:x.split("_")[2])
    
    kdf = kdf.copy()
    kdf['PN_S1'] = kdf['MODEL_DUN_PN_2022'] + (lost_faith)*kdf[['MODEL_DUN_PH_2022', 'MODEL_DUN_BN_2022']].sum(axis=1)
    kdf['PH_S1'] = (1-lost_faith)*kdf['MODEL_DUN_PH_2022']
    kdf['BN_S1'] = (1-lost_faith)*kdf['MODEL_DUN_BN_2022']
    kdf['Majority_S1'] =  compute_majority(kdf[['PN_S1', 'PH_S1', 'BN_S1']])
    kdf['Winner_S1'] = kdf[['PN_S1', 'PH_S1', 'BN_S1']].idxmax(axis=1)
    
    kdf['PN_S2A'] = kdf['MODEL_DUN_PN_2022'] + (1-bn_to_ph)*kdf['MODEL_DUN_BN_2022'] 
    kdf['PH_S2A'] = kdf['MODEL_DUN_PH_2022'] + (bn_to_ph)*kdf['MODEL_DUN_BN_2022']
    kdf['PH_S2A'] = kdf['PH_S2A'] * (1-lost_faith)
    kdf['Majority_S2A'] =  compute_majority(kdf[['PN_S2A', 'PH_S2A']])
    kdf['Winner_S2A'] = kdf[['PN_S2A', 'PH_S2A']].idxmax(axis=1)
    
    kdf['PN_S2B'] = kdf['MODEL_DUN_PN_2022'] + (1-ph_to_bn)*kdf['MODEL_DUN_PH_2022'] 
    kdf['BN_S2B'] = kdf['MODEL_DUN_BN_2022'] + (ph_to_bn)*kdf['MODEL_DUN_PH_2022']
    kdf['BN_S2B'] = kdf['BN_S2B'] * (1-lost_faith)
    kdf['Majority_S2B'] =  compute_majority(kdf[['PN_S2B', 'BN_S2B']])
    kdf['Winner_S2B'] = kdf[['PN_S2B', 'BN_S2B']].idxmax(axis=1)

    kdf['Winner_S1'] = kdf['Winner_S1'].apply(lambda x:x.split("_")[0])
    kdf['Winner_S2A'] = kdf['Winner_S2A'].apply(lambda x:x.split("_")[0])
    kdf['Winner_S2B'] = kdf['Winner_S2B'].apply(lambda x:x.split("_")[0])
    kdf['Winner_S2'] = ((kdf['Winner_S2A'] == 'PN') & (kdf['Winner_S2B'] == 'PN')).map({True: "PN", False: "Unity"})
    winner_cols = kdf.columns[kdf.columns.str.startswith("Winner")]
    scenario_df = pd.concat([kdf[w].value_counts(normalize=False) for w in winner_cols], axis=1)
    scenario_df.columns = winner_cols
    return kdf, scenario_df


def get_pie_fig(sr, title):
    # Create a sample data with precalculated percentage values
    x = sr.sort_values(ascending=False).dropna()
    labels = x.index
    values = x.values

    # Define color mapping
    color_map = {
        'PN': '#85E579',    # Slightly darker green
        'PH': '#FF7171',    # Slightly darker red
        'Unity': '#FF84A3',  # Slightly darker gray
        'BN': '#82C9FF'     # Slightly darker blue
    }
    color_map = {
        'PN': '#65D25C',    # Darker green
        'PH': '#FF5E5E',    # Darker red
        'Unity': '#FF7399',  # Darker gray
        'BN': '#70BFFF'     # Darker blue
    }
    # color_map = {
    #     'PN': '#A7F59C',   # Light green
    #     'PH': '#FF8F8F',   # Light red
    #     'Unity': '#FF9DBB',  # Light gray
    #     'BN': '#9DD8FF'    # Light blue
    # }

    # Get colors from color mapping
    colors = [color_map[label] for label in labels]

    # Create the Donut Chart trace
    trace = go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo='value',
        hoverinfo='label+percent',
        hole=0.4,
        textfont_size=20,
    )

    # Create the layout for the Donut Chart
    layout = go.Layout(
        title=title,
        title_font=dict(size=24, color='#333'),
        width=600,
        height=600,
        margin=dict(t=100, b=0, l=0, r=0),
        hoverlabel=dict(font=dict(size=20))
    )

    # Create the Figure object
    fig = go.Figure(data=[trace], layout=layout)

    # Display the Donut Chart
    # fig.show()
    return fig