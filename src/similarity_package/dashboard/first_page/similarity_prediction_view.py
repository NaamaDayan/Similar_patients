import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

from src.similarity_package.configs import COMPONENT_STYLE, MARGIN_STYLE
from src.similarity_package.full_pipeline import get_distance_score_by_group



def get_prediction_html_view():
    return html.Div(
        [dbc.Row([dbc.Col(html.Div([dcc.Graph(id='scatter_plot', clickData={'points': [{'customdata': '1'}]})],
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
                  ])
         ])


def register_prediction_callbacks(app):
    @app.callback(
        Output("pie_chart", "figure"),
        Input("diagnosis_feature_weight", "value"),
        Input("lab_result_feature_weight", "value"),
        Input("drug_feature_weight", "value"))
    def generage_pie_chart(val1, val2, val3):
        print(val1, val2, val3)
        fig = px.pie(values=[val1, val2, val3],
                     names=['Diagnosis Features weight', 'Lab Results Features weight', 'Drug Features weight'],
                     hole=.3)

        return fig

    @app.callback(Output(component_id='scatter_plot', component_property='figure'),
                  [Input(component_id='dropdown', component_property='value')])
    def graph_update(group_num):
        distances = get_distance_score_by_group(group_num)['distance'].reset_index()
        fig = go.Figure([go.Scatter(
            x=distances['distance'].values,
            y=[0] * len(distances),
            mode='markers',
            marker=dict(size=list(reversed([10 * i for i in range(len(distances))])),
                        color=distances['distance'].values),
            # size=100 * distances['distance'].values,
            text=list(distances['ID'].values))])
        fig.update_traces(customdata=list(distances['ID'].values))
        fig.update_layout(
            xaxis_title='Distance Score',
            # width=800,
            # height=500,
        ).update_xaxes(showgrid=False).update_yaxes(showgrid=False)
        return fig
