from typing import List

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Output, Input, dash_table
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.similarity_package.configs import TARGET_PATIENT_ID, SHADOW_STYLE, DIAGNOSES_SHEET_NAME, \
    LAB_RESULTS_SHEET_NAME, DRUGS_SHEET_NAME, COMPONENT_STYLE, MARGIN_STYLE
from src.similarity_package.full_pipeline import get_df_by_sheet_name


def get_categories_encodings(label_encoder, patient_data: pd.DataFrame, col_name: str):
    return " ".join(map(str, label_encoder.transform(list(set(patient_data[col_name].values)))))


def get_rare_categories(data: pd.DataFrame, frequency_factor: float, col_name: str):
    label_encoder = LabelEncoder()
    label_encoder.fit(data[col_name].unique())

    categories_encoding = data.groupby('ID').apply(
        lambda x: get_categories_encodings(label_encoder, x, col_name)).values
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(categories_encoding)
    terms = vectorizer.get_feature_names()

    data = [(term, X.sum(axis=0)[0, col]) for col, term in enumerate(terms)]

    ranking = pd.DataFrame(data, columns=['term', 'rank'])

    ranking[col_name] = label_encoder.inverse_transform(list(map(int, ranking['term'].values)))
    category_rank = ranking.sort_values('rank', ascending=False)
    rank_percentile_threshold = np.percentile(category_rank['rank'].values, frequency_factor)

    return category_rank[category_rank['rank'] <= rank_percentile_threshold][col_name].values


def get_shared_categories(sheet_name: str, col_name: str, group_num: int, frequency_factor: int = 10):
    data = get_df_by_sheet_name(group_num, sheet_name)
    patient_ids = data['ID'].unique()
    # patient_ids = [id for id in data['ID'].unique() if id != TARGET_PATIENT_ID]

    # if frequency_factor < 100:
    rare_categories = get_rare_categories(data, int(frequency_factor), col_name)
    data = data[data[col_name].isin(rare_categories)]

    shared_categories = []
    target_patient_categories = data[data['ID'] == TARGET_PATIENT_ID][col_name].value_counts().keys()

    for patient_id in patient_ids:
        patient_categories = data[data['ID'] == patient_id]
        categories = patient_categories[col_name].value_counts().keys()
        shared_categories.append(list(set(categories) & set(target_patient_categories)))

    return pd.DataFrame(shared_categories, index=patient_ids).T


def create_agg_df_div(table_name: str, col_name: str, sheet_name: str, group_num: int) -> List[dbc.Col]:
    df = get_shared_categories(sheet_name=sheet_name, col_name=col_name, group_num=group_num,
                               frequency_factor=100)
    df.columns = list(map(str, df.columns))
    return [dbc.Col(html.Div([html.H2(id=f'H2_shared_{table_name}', children=table_name,
                                      style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 40}),
                              html.P("% Frequency:"),
                              html.Div(dcc.Input(id=f'frequency_{table_name}', type='range', min=0, max=100, step=10,
                                                 value=100), style={'margin-bottom': '0.2%'}),

                              html.Div(dash_table.DataTable(
                                  df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
                                  id=f'{table_name}_shared_table',
                                  style_cell={
                                      'overflow': 'hidden',
                                      'textOverflow': 'ellipsis',
                                      'maxWidth': 0
                                  },
                                  style_table={'overflowX': 'auto', 'maxWidth': '100%'},
                                  page_size=5), style={"padding": "2rem 1rem"}),

                              ], style={**SHADOW_STYLE, **MARGIN_STYLE, 'height': '90%', 'align': 'center'}), width=6),
            dbc.Col(html.Div([dcc.Graph(id=f'{table_name}_shared')], style=COMPONENT_STYLE),
                    width=6)
            ]


def draw_figure(table_name: str):
    return html.Div([dcc.Graph(id=f'{table_name}_shared')])


def draw_table(table_name: str, col_name: str, sheet_name: str, group_num: int):
    df = get_shared_categories(sheet_name=sheet_name, col_name=col_name, group_num=group_num,
                               frequency_factor=100)
    df.columns = list(map(str, df.columns))
    return html.Div([html.P("% Frequency:"),
                     html.Div(dcc.Input(id=f'frequency_{table_name}', type='range', min=0, max=100, step=10,
                                        value=100), style={'margin-bottom': '0.2%'}),

                     html.Div(dash_table.DataTable(
                         df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
                         id=f'{table_name}_shared_table',
                         style_cell={
                             'overflow': 'hidden',
                             'textOverflow': 'ellipsis',
                             'maxWidth': 0
                         },
                         style_table={'overflowX': 'auto', 'maxWidth': '100%'},
                         page_size=5), style={"padding": "2rem 1rem"}),

                     ])


def draw_category_text(caetgory: str):
    return html.Div([html.H2(caetgory)], style={'textAlign': 'center'})


def get_category_rows(table_name: str, col_name: str, sheet_name: str, group_num: int):
    return [dbc.Row([
        dbc.Col([
            draw_category_text(table_name)
        ], width=12),
    ], align='center'),
        html.Br(),
        dbc.Row([
            dbc.Col([
                draw_table(table_name, col_name, sheet_name, group_num)
            ], width=6),
            dbc.Col([
                draw_figure(table_name)
            ], width=6),
        ], align='center'),
        html.Br()]


def get_shared_categories_html_view():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                *get_category_rows('Diagnosis', 'Diagnosis', DIAGNOSES_SHEET_NAME, 1),
                *get_category_rows('Lab Results', 'TestName', LAB_RESULTS_SHEET_NAME, 1),
                *get_category_rows('Drugs', 'DrugName', DRUGS_SHEET_NAME, 1),
            ])
        )
    ])


def get_shared_category_figure(shared_categories: pd.DataFrame, table_name: str):
    print (shared_categories.columns)
    shared_percentage = [
        100 * len(shared_categories[col][shared_categories[col].notna()].values) / len(shared_categories[str(TARGET_PATIENT_ID)].values) for col
        in shared_categories.columns]
    fig = go.Figure([go.Bar(
        x=shared_categories.columns,
        y=shared_percentage,
        marker_color=shared_percentage
    )])
    fig.update_layout(xaxis_title="Similar Patient IDs",
                      yaxis_title=f"% Shared {table_name}")
    return fig


def register_shared_categories_callbacks(app):
    @app.callback(
        Output('Diagnosis_shared_table', 'data'),
        Input("frequency_Diagnosis", "value"),
        Input(component_id='dropdown', component_property='value'))
    def get_shared_diagnosis(frequency: int, group_num: int):
        return get_shared_categories(sheet_name=DIAGNOSES_SHEET_NAME, col_name='Diagnosis', group_num=group_num,
                                     frequency_factor=frequency).to_dict('records')

    @app.callback(
        Output('Lab Results_shared_table', 'data'),
        Input("frequency_Lab Results", "value"),
        Input(component_id='dropdown', component_property='value'))
    def get_shared_diagnosis(frequency: int, group_num: int):
        return get_shared_categories(sheet_name=LAB_RESULTS_SHEET_NAME, col_name='TestName', group_num=group_num,
                                     frequency_factor=frequency).to_dict('records')

    @app.callback(
        Output('Drugs_shared_table', 'data'),
        Input("frequency_Drugs", "value"),
        Input(component_id='dropdown', component_property='value'))
    def get_shared_diagnosis(frequency: int, group_num: int):
        return get_shared_categories(sheet_name=DRUGS_SHEET_NAME, col_name='DrugName', group_num=group_num,
                                     frequency_factor=frequency).to_dict('records')

    @app.callback(
        Output('Diagnosis_shared', 'figure'),
        Input(component_id='Diagnosis_shared_table', component_property='data'))
    def update_figure(shared_categories: list):
        return get_shared_category_figure(pd.DataFrame(shared_categories), 'Diagnosis')

    @app.callback(
        Output('Lab Results_shared', 'figure'),
        Input(component_id='Lab Results_shared_table', component_property='data'))
    def update_figure(shared_categories: list):
        return get_shared_category_figure(pd.DataFrame(shared_categories), 'Lab Results')

    @app.callback(
        Output('Drugs_shared', 'figure'),
        Input(component_id='Drugs_shared_table', component_property='data'))
    def update_figure(shared_categories: list):
        return get_shared_category_figure(pd.DataFrame(shared_categories), 'Drugs')
