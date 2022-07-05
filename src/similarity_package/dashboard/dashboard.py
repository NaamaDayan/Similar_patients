import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from src.similarity_package.dashboard.first_page.first_page import get_first_page_html_view
from src.similarity_package.dashboard.first_page.rules_view import register_rules_callback
from src.similarity_package.dashboard.first_page.similarity_prediction_view import register_prediction_callbacks
from src.similarity_package.dashboard.first_page.unique_patient_view import register_callbacks
from src.similarity_package.dashboard.second_page.second_page import get_second_page_html_view
from src.similarity_package.dashboard.second_page.shared_categories_view import register_shared_categories_callbacks

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "background-color": "#f8f9fa"
}

page_mapping = {'/': get_first_page_html_view(), '/page-2': get_second_page_html_view()}


def get_sidebar_html_view():
    navbar = html.Div([dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/")),
            dbc.NavItem(dbc.NavLink("Page-2", href="/page-2")),
        ],
        brand="Similar Patient Dashboard",
        brand_href="#",
        color="#000000",
        dark=True,
    )], style={"background-color": "#f8f9fa", })

    content = html.Div(id="page-content", style=CONTENT_STYLE)

    return html.Div([dcc.Location(id="url"), navbar, content])


def register_sidebar_callback(app):
    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def render_page_content(pathname):
        try:
            print("here", pathname)
            return page_mapping[pathname]
        except KeyError:
            print("there")
            return dbc.Jumbotron(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognised..."),
                ]
            )


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True )
register_callbacks(app)
register_prediction_callbacks(app)
register_sidebar_callback(app)
register_shared_categories_callbacks(app)
register_rules_callback(app)
group_num = 1

app.layout = get_sidebar_html_view()
if __name__ == '__main__':
    app.run_server(debug=True)
