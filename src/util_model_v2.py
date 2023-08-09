import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
# from util_model_v1 import run_election_model
from src.util import run_election_model

# PH vs PN
# PH, PN, Unsure
# malay_1 = [0.39, 0.53, 0.08]
# chinese_1 = [0.62, 0.09, 0.29]
# indian_1 = [0.68, 0.21, 0.11]

malay_1 = [0.4, 0.46, 0.15]
chinese_1 = [0.62, 0.09, 0.29]
indian_1 = [0.68, 0.21, 0.11]

# BN vs PN
# BN, PN, Unsure
malay_2 = [0.35, 0.57, 0.07]
chinese_2 = [0.41, 0.28, 0.31]
indian_2 = [0.46, 0.37, 0.17]


def compute_outcome(state, unity_party, num_malay, num_chinese, num_indian, tilt=[0.5, 0.5, 0.5], turnout = [0.79, 0.69, 0.78]):
    if state == 'SELANGOR':
        # PH, PN, Unsure
        malay_1 = [0.4, 0.46, 0.15]
        chinese_1 = [0.62, 0.09, 0.29]
        indian_1 = [0.68, 0.21, 0.11]
        # BN, PN, Unsure
        malay_2 = [0.35, 0.57, 0.07]
        chinese_2 = [0.41, 0.28, 0.31]
        indian_2 = [0.46, 0.37, 0.17]
    elif state == 'KEDAH':
        # PH, PN, Unsure
        malay_1 = [0.28, 0.61, 0.12]
        chinese_1 = [0.72, 0.19, 0.09]
        indian_1 = [0.68, 0.32, 0.00]
        # BN, PN, Unsure
        malay_2 = [0.23, 0.75, 0.02]
        chinese_2 = [0.65, 0.26, 0.09]
        indian_2 = [0.68, 0.32, 0.00]


    tilt_malay, tilt_chinese, tilt_indian = tilt
    turnout_malay, turnout_chinese, turnout_indian = turnout
    # turnout_malay, turnout_chinese, turnout_indian = 0.8,0.9,0.9
    if unity_party == "PH":
        malay_k, chinese_k, indian_k = malay_1.copy(), chinese_1.copy(), indian_1.copy()
    elif unity_party == "BN":
        malay_k, chinese_k, indian_k = malay_2.copy(), chinese_2.copy(), indian_2.copy()
    else:
        raise

    malay_unity = malay_k[0]+malay_k[2]*(tilt_malay)
    chinese_unity = chinese_k[0]+chinese_k[2]*(tilt_chinese)
    indian_unity = indian_k[0]+indian_k[2]*(tilt_indian)

    malay_pn = malay_k[1]+malay_k[2]*(1-tilt_malay)
    chinese_pn = chinese_k[1]+chinese_k[2]*(1-tilt_chinese)
    indian_pn = indian_k[1]+indian_k[2]*(1-tilt_indian)
    
    unity_votes = num_malay*(malay_unity)*turnout_malay + num_chinese*(chinese_unity)*turnout_chinese + num_indian*(indian_unity)*turnout_indian
    pn_votes = num_malay*(malay_pn)*turnout_malay + num_chinese*(chinese_pn)*turnout_chinese + num_indian*(indian_pn)*turnout_indian
    if unity_votes > pn_votes:
        winner = 'Unity'
    else:
        winner = 'PN'
    d = {
        'UN_M': num_malay*(malay_unity)*turnout_malay,
        'UN_C':  num_chinese*(chinese_unity)*turnout_chinese ,
        'UN_I': num_indian*(indian_unity)*turnout_indian,
        'PN_M': num_malay*(malay_pn)*turnout_malay, 
        'PN_C': num_chinese*(chinese_pn)*turnout_chinese, 
        'PN_I': num_indian*(indian_pn)*turnout_indian, 
        'Turnout': unity_votes+pn_votes,
        'Majority': abs(unity_votes-pn_votes),
        'UN': unity_votes,
        'PN': pn_votes,
        'Winner': winner,
    }
    return d


def run_election_model_v2(state, tilt, turnout, skip_model_1=False):
    # tilt=[0.0, 0.5, 0.5]
    # tilt=[0.0, 0.5, 0.5]
    # turnout = [0.79, 0.69, 0.78]
    # turnout = [0.9, 0.6, 0.7]
    dun_comp_df = pd.read_csv(f'data/{state}_2023_DUN_COMPOSITION_custom_done.csv')
    dun_comp_df['STATE CONSTITUENCY CODE'] = dun_comp_df['STATE CONSTITUENCY CODE'].apply(lambda x: x.replace(" ", ""))
    dun_comp_df = dun_comp_df.set_index('STATE CONSTITUENCY CODE')
    df18 = pd.read_csv("data/MALAYSIA_2018_DUN_COMPOSITION_IN_PROGRESS.csv").query(f"STATE == '{state}'")
    df18['STATE CONSTITUENCY CODE'] = df18['STATE CONSTITUENCY CODE'].apply(lambda x: x.replace(" ", ""))
    df18 = df18.groupby('STATE CONSTITUENCY CODE').first()
    try:
        dun_comp_df['N'] = dun_comp_df['TOTAL ELECTORS (2023)']
    except:
        dun_comp_df['N'] = dun_comp_df['TOTAL ELECTORS']

    dun_comp_df['M_18'] = df18['MALAY (%)']
    dun_comp_df['C_18'] = df18['CHINESE (%)']
    dun_comp_df['I_18'] = df18['INDIANS (%)']
    dun_comp_df['O_18'] = df18['OTHERS (%)']
    dun_comp_df['Bumi'] = dun_comp_df[['ORANG ASLI (%)', 'BUMIPUTERA SABAH (%)', 'BUMIPUTERA SARAWAK (%)']].sum(axis=1)
    dun_comp_df['Malay'] = dun_comp_df['MALAY (%)']
    dun_comp_df['Chinese'] = dun_comp_df['CHINESE (%)']
    dun_comp_df['Indian'] = dun_comp_df['INDIANS (%)']
    dun_comp_df['Other'] = dun_comp_df['OTHERS (%)']
    dun_comp_df['Young1'] = dun_comp_df['18-20 (%)']
    dun_comp_df['Young2'] = dun_comp_df['18-20 (%)'] + dun_comp_df['21-29 (%)']
    # dun_comp_sub_df = dun_comp_df[['STATE CONSTITUENCY NAME', 'Invoke', 'N', 'Bumi', 'M_18', 'Malay', 'C_18', 'Chinese', 'I_18', 'Indian', 'O_18', 'Other', 'Young1', 'Young2', 'Party 2', 'Party 3']].copy()
    dun_comp_sub_df = dun_comp_df[['STATE CONSTITUENCY NAME', 'Invoke', 'N', 'Bumi', 'M_18', 'Malay', 'C_18', 'Chinese', 'I_18', 'Indian', 'O_18', 'Other', 'Young1', 'Young2', 'Party 2']].copy()
    for i in dun_comp_sub_df.index:
        d = dun_comp_sub_df.loc[i, :]
        x = compute_outcome(state=state, unity_party=d['Party 2'], num_malay=d['M_18'], num_chinese=d['C_18'], num_indian=d['I_18'], tilt=tilt, turnout=turnout)
        dun_comp_sub_df.loc[i, 'Model_2_2018'] = x['Winner']

    for i in dun_comp_sub_df.index:
        d = dun_comp_sub_df.loc[i, :]
        x = compute_outcome(state=state, unity_party=d['Party 2'], num_malay=d['Malay'], num_chinese=d['Chinese'], num_indian=d['Indian'], tilt=tilt, turnout=turnout)
        for key, val in x.items():
            dun_comp_sub_df.loc[i, key] = val
    dun_comp_sub_df['Turnout_N'] = dun_comp_sub_df['Turnout']/100*dun_comp_sub_df['N']

    # state = 'SELANGOR'
    lost_faith = 0
    bn_to_ph = 0.15
    bn_to_pn = 0.85
    ph_to_bn = 0.24
    ph_to_pn = 0.76

    if not skip_model_1:
        # kdf, scenario_df = run_election_model(state, lost_faith, bn_to_ph, ph_to_bn)
        kdf, scenario_df = run_election_model(state, bn_to_ph, bn_to_pn, ph_to_bn, ph_to_pn)
        dun_comp_sub_df['Model_1'] = kdf.set_index('STATE CONSTITUENCY CODE')['Winner_S2']
    dun_comp_sub_df['Model_2'] = dun_comp_sub_df['Winner']
    dun_comp_sub_df['Model_2 (2018)'] = dun_comp_sub_df['Model_2_2018']
    dun_comp_sub_df['Model_Invoke'] = dun_comp_sub_df['Invoke']
    return dun_comp_sub_df.drop(columns=['Winner', 'Model_2_2018', 'Invoke'])


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