import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, Input, Output

from src.similarity_package.configs import COMPONENT_STYLE, MARGIN_STYLE
from src.similarity_package.full_pipeline import get_distance_score_by_group


def get_similarity_distance_fig(group_num: int, feature_weight_diagnosis: float, feature_weight_lab_result: float,
                                feature_weight_drugs: float):
    distances = get_distance_score_by_group(group_num, feature_weight_diagnosis, feature_weight_lab_result,
                                            feature_weight_drugs)['distance'].reset_index()
    print(distances)
    fig = go.Figure([go.Scatter(
        x=distances['distance'].values,
        y=[0] * len(distances),
        mode='markers',
        marker=dict(size=list(reversed([10 * i for i in range(len(distances))])),
                    color=distances['distance'].values),
        # size=100 * distances['distance'].values,
        text=list(distances['ID'].values))])

    print("here 1")
    fig.update_traces(customdata=list(distances['ID'].values))
    fig.update_layout(
        xaxis_title='Distance Score',
        # width=800,
        # height=500,
    ).update_xaxes(showgrid=False).update_yaxes(showgrid=False)
    print("here 2")
    return fig


def get_prediction_html_view():
    return html.Div(
        dbc.Row([dbc.Col(html.Div([dcc.Graph(id='scatter_plot', clickData={'points': [{'customdata': '1'}]})],
                                  style=COMPONENT_STYLE),
                         width='auto'),

                 dbc.Col(html.Div([dcc.Graph(id="pie_chart")], style=COMPONENT_STYLE),
                         width='auto'),

                 dbc.Col(html.Div([html.P("Diagnosis weight:"),
                                   dcc.Input(id='diagnosis_feature_weight', type='range', min=0, max=1, step=0.01,
                                             value=0.33),
                                   html.P("Lab Results weight:"),
                                   dcc.Input(id='lab_result_feature_weight', type='range', min=0, max=1, step=0.01,
                                             value=0.33),
                                   html.P("Drugs weight:"),
                                   dcc.Input(id='drug_feature_weight', type='range', min=0, max=1, step=0.01,
                                             value=0.33)], style=MARGIN_STYLE),
                         width='auto')
                 ]),
        id='loading-output'
    )


def get_loading_prediction_html_view():
    return html.Div(
        dcc.Loading(
            id="loading",
            type="circle",
            children=get_prediction_html_view()
        )
    )


def register_prediction_callbacks(app):
    @app.callback(
        [Output("pie_chart", "figure"),
         Output('scatter_plot', 'figure')],
        Input("diagnosis_feature_weight", "value"),
        Input("lab_result_feature_weight", "value"),
        Input("drug_feature_weight", "value"),
        Input('dropdown', 'value'))
    def generate_pie_chart(feature_weight_diagnosis: float, feature_weight_lab_result: float,
                           feature_weight_drugs: float, group_num: int):
        print("weights", feature_weight_lab_result, feature_weight_drugs, feature_weight_diagnosis)
        fig = px.pie(values=[feature_weight_diagnosis, feature_weight_lab_result, feature_weight_drugs],
                     names=['Diagnosis Features weight', 'Lab Results Features weight', 'Drug Features weight'],
                     hole=.3)

        return fig, get_similarity_distance_fig(group_num, feature_weight_diagnosis, feature_weight_lab_result,
                                                feature_weight_drugs)

    # @app.callback(Output(component_id='scatter_plot', component_property='figure'),
    #               [Input(component_id='dropdown', component_property='value')])
    # def graph_update(group_num):
    #
