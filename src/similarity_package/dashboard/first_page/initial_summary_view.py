from datetime import datetime

import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objects as go
from dash import html, dcc, Output, Input

from src.similarity_package.configs import COMPONENT_STYLE, MARGIN_STYLE, DIAGNOSES_SHEET_NAME
from src.similarity_package.full_pipeline import get_distance_score_by_group, get_df_by_sheet_name


def drawText(text, subtext, id_):
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2(text),
                    html.H3(subtext),
                ], style={'textAlign': 'center', 'height': '150'})
            ])
        ),
    ], style=MARGIN_STYLE, id=id_)


def get_cohort_summary_html_view():
    return html.Div(
        dbc.Row([dbc.Col(drawText(3, 'Women', 'text_1_'), width=2),
                 dbc.Col(drawText(2, 'Men', 'text_2_'), width=2),
                 dbc.Col(drawText(5, 'Total', 'text_3_'), width={'size': 2, 'offset': 1}),
                 dbc.Col(drawText(23, 'Average Age', 'text_4_'), width={'size': 2})
                 ],
                ), id='cohort-summary'
    )


def register_cohort_summary_callbacks(app):
    @app.callback(
        [Output("text_1_", "value"),
         Output('text_2_', 'value'),
         Output('text_3_', 'value'),
         Output('text_4_', 'value')],
        Input('dropdown', 'value'))
    def get_texts(group_num: int):
        df = get_df_by_sheet_name(group_num, DIAGNOSES_SHEET_NAME)
        mean_age = datetime.now().year - df['birth_date'].mean()
        count = df.shape[0]
