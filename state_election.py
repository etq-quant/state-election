import os
import glob
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from util import run_election_model, get_pie_fig

# Streamlit main layout
title = "Malaysia State Elections 2023"
st.set_page_config(page_title=title, layout="wide", initial_sidebar_state="collapsed")
st.title(title)

# (tab_market, tab_sector) = st.tabs(
#     [
#         "Market",
#         "Sector",
#     ]
# )
# with tab_market:

cols_a = st.columns(6)

with cols_a[0]:
    state = st.selectbox("State", ['SELANGOR', 'PULAU_PINANG', 'NEGERI_SEMBILAN', 'KEDAH'])

with cols_a[1]:
    bn_to_ph = st.number_input('Transferability BN to PH %', min_value=0, max_value=100, value=15, step = 1)
    bn_to_pn = st.number_input('BN to PN %', min_value=0, max_value=100, value=80, step = 1, key='bn_pn')
    bn_to_ph /= 100
    bn_to_pn /= 100
    bn_to_none = 1-bn_to_ph-bn_to_pn
    st.text(f"BN to abstain: {bn_to_none:.2%}")

with cols_a[2]:
    ph_to_bn = st.number_input('Transferability PH to BN %', min_value=0, max_value=100, value=24, step = 1, key='bn_ph')
    ph_to_pn = st.number_input('PH to PN %', min_value=0, max_value=100, value=71, step = 1, key='ph_pn')
    ph_to_bn /= 100
    ph_to_pn /= 100
    ph_to_none = 1-ph_to_bn-ph_to_pn
    st.text(f"PH to abstain %: {ph_to_none:.2%}")

cols_b = st.columns(6)
with cols_b[0]:
    show_columns = st.checkbox("Show More Data", key="show_data")


df, scenario_df = run_election_model(state, bn_to_ph, bn_to_pn, ph_to_bn, ph_to_pn)
df = df.rename(columns = {'STATE CONSTITUENCY CODE': 'Code', 'STATE CONSTITUENCY NAME': 'Name'})
sdf = df.reset_index()
if not show_columns:
    cols_ = sdf.loc[:, 'MODEL_DUN_PN_2022':].columns.tolist()
    sdf = sdf.loc[:, ['Code', 'Name']  + cols_].copy()

# Create a function to format float columns as 1 decimal percentage
def format_float_as_percentage(value):
    return '{:.1%}'.format(value)

styled_df = sdf.set_index(['Code', 'Name']).style.format({col: format_float_as_percentage for col in sdf.select_dtypes(include='float').columns})

# Define a function to map categories to colors
def map_category_to_color(category):
    color_map = {
        'PN': '#A7F59C',   # Light green
        'PH': '#FF8F8F',   # Light red
        'Unity': '#FF7399',  # Light gray
        'BN': '#9DD8FF'    # Light blue
    }
    return 'background-color: {}'.format(color_map.get(category, ''))
  
# Apply the mapping to the Category column
winner_cols = sdf.columns[sdf.columns.str.startswith('Winner')]
styled_df = styled_df.applymap(map_category_to_color, subset=winner_cols)

majority_cols = sdf.columns[sdf.columns.str.startswith('Majority')]
styled_df = styled_df.background_gradient(subset=majority_cols,  cmap='Reds_r', vmin=0, vmax=0.1)

st.dataframe(styled_df)
# st.dataframe(scenario_df)

# Pie charts
fig_a0 = get_pie_fig(scenario_df['Winner_FED_2018'], title='Federal 2018')
fig_a1 = get_pie_fig(scenario_df['Winner_FED_2018'], title='Federal 2022')
fig_a2 = get_pie_fig(scenario_df['Winner_DUN_2018'], title='DUN 2018')

fig_0 = get_pie_fig(scenario_df['Winner_MODEL_DUN_2022'], title='Base Model - Three-way (DUN2022)')
# fig_1 = get_pie_fig(scenario_df['Winner_S1'], title='S1 - Three-Way (DUN2022)')

fig_2a = get_pie_fig(scenario_df['Winner_S2A'], title='S2A - PH Only (DUN2022)')
fig_2b = get_pie_fig(scenario_df['Winner_S2B'], title='S2B - BN Only (DUN2022)')
fig_2 = get_pie_fig(scenario_df['Winner_S2'], title='S2 - Optimal Unity (DUN2022)')

if show_columns:
    cols_f2 = st.columns(3)
    with cols_f2[0]:
        st.plotly_chart(fig_a0, use_container_width=True)
    with cols_f2[1]:
        st.plotly_chart(fig_a1, use_container_width=True)
    with cols_f2[2]:
        st.plotly_chart(fig_a2, use_container_width=True)

cols_f1 = st.columns([4, 1, 2, 2, 4])
with cols_f1[0]:
    st.plotly_chart(fig_0, use_container_width=True)
with cols_f1[1]:
    st.text('')
with cols_f1[2]:
    st.plotly_chart(fig_2a, use_container_width=True)
with cols_f1[3]:
    st.plotly_chart(fig_2b, use_container_width=True)
with cols_f1[4]:
        st.plotly_chart(fig_2, use_container_width=True)