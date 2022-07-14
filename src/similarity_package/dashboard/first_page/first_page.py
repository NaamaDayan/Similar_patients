from dash import html, dcc
import dash_bootstrap_components as dbc

from src.similarity_package.dashboard.first_page.initial_summary_view import get_cohort_summary_html_view
from src.similarity_package.dashboard.first_page.rules_view import get_rules_html_view
from src.similarity_package.dashboard.first_page.similarity_prediction_view import get_prediction_html_view
from src.similarity_package.dashboard.first_page.unique_patient_view import get_unique_patient_html_view


def get_first_page_html_view():
    return html.Div(id='parent', children=[

        dbc.Row(dbc.Col(dcc.Dropdown(id='dropdown',
                                     options=[
                                         {'label': 'First Group', 'value': '1'},
                                         {'label': 'Second Group', 'value': '2'},
                                     ],
                                     value='1'))),
        # get_cohort_summary_html_view(),
        get_prediction_html_view(),
        get_rules_html_view(),
        get_unique_patient_html_view()
    ])
