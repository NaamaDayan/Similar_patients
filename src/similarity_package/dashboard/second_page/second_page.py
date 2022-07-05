import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import html, dcc, Output, Input, dash_table
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.similarity_package.configs import TARGET_PATIENT_ID, SHADOW_STYLE, DIAGNOSES_SHEET_NAME, \
    LAB_RESULTS_SHEET_NAME, DRUGS_SHEET_NAME
from src.similarity_package.dashboard.first_page.similarity_prediction_view import get_prediction_html_view
from src.similarity_package.dashboard.first_page.unique_patient_view import get_unique_patient_html_view
from src.similarity_package.full_pipeline import get_data_by_id, get_df_by_sheet_name


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

    # if frequency_factor < 100:
    rare_categories = get_rare_categories(data, frequency_factor, col_name)
    data = data[data[col_name].isin(rare_categories)]

    shared_categories = []
    target_patient_categories = data[data['ID'] == TARGET_PATIENT_ID][col_name].value_counts().keys()

    for patient_id in patient_ids:
        patient_categories = data[data['ID'] == patient_id]
        categories = patient_categories[col_name].value_counts().keys()
        shared_categories.append(list(set(categories) & set(target_patient_categories)))

    return pd.DataFrame(shared_categories, index=patient_ids).T.to_dict('records')


def create_agg_df_div(table_name: str, sheet_name, group_num) -> html.Div:
    df = get_data_by_id(group_num, 1, sheet_name=sheet_name)
    return html.Div([html.H2(id=f'H2_shared_{table_name}', children=table_name,
                             style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 40}),
                     html.P("% Frequency:"),
                     html.Div(dcc.Input(id=f'frequency_{table_name}', type='range', min=0, max=100, step=1,
                                        value=100), style={'margin-bottom': '0.2%'}),
                     html.Div([
                         dash_table.DataTable(
                             df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
                             id=f'{table_name}_shared_table',
                             page_size=5)
                     ])
                     ], style=SHADOW_STYLE)


def get_shared_categories_html_view():
    return html.Div(dbc.Row([dbc.Col(create_agg_df_div('Diagnosis', DIAGNOSES_SHEET_NAME, 1), width='100%'),
                             dbc.Col(create_agg_df_div('Lab Results', LAB_RESULTS_SHEET_NAME, 1), width='100%'),
                             dbc.Col(create_agg_df_div('Drugs', DRUGS_SHEET_NAME, 1), width='100%')]
                            ))


def register_shared_categories_callbacks(app):
    @app.callback(
        Output('Diagnosis_shared_table', 'data'),
        Input("frequency_", "value"),
        Input(component_id='dropdown', component_property='value'))
    def get_shared_diagnosis(frequency: int, group_num: int):
        return get_shared_categories(sheet_name=DIAGNOSES_SHEET_NAME, col_name='Diagnosis', group_num=group_num, frequency_factor=frequency)

    @app.callback(
            Output('Lab Results_shared_table', 'data'),
            Input("frequency_", "value"),
            Input(component_id='dropdown', component_property='value'))
    def get_shared_diagnosis(frequency: int, group_num: int):
        return get_shared_categories(sheet_name=LAB_RESULTS_SHEET_NAME, col_name='TestName', group_num=group_num, frequency_factor=frequency)

    @app.callback(
            Output('Drugs_shared_table', 'data'),
            Input("frequency_", "value"),
            Input(component_id='dropdown', component_property='value'))
    def get_shared_diagnosis(frequency: int, group_num: int):
        return get_shared_categories(sheet_name=DRUGS_SHEET_NAME, col_name='DrugName', group_num=group_num, frequency_factor=frequency)


def get_first_page_html_view():
    return html.Div(id='parent', children=[

        dbc.Row(dbc.Col(dcc.Dropdown(id='dropdown',
                                     options=[
                                         {'label': 'First Group', 'value': '1'},
                                         {'label': 'Second Group', 'value': '2'},
                                     ],
                                     value='1'))),

        get_prediction_html_view(),
        get_unique_patient_html_view()
    ])
