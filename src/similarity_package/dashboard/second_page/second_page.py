import dash_bootstrap_components as dbc
from dash import html, dcc

from src.similarity_package.dashboard.second_page.shared_categories_view import get_shared_categories_html_view


def get_second_page_html_view():
    return html.Div(id='parent', children=[

        dbc.Row(dbc.Col(dcc.Dropdown(id='dropdown',
                                     options=[
                                         {'label': 'First Group', 'value': '1'},
                                         {'label': 'Second Group', 'value': '2'},
                                     ],
                                     value='1'))),

        get_shared_categories_html_view()

    ])
