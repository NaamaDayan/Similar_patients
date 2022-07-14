import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, Input, Output
import dash_daq as daq

from src.similarity_package.configs import COMPONENT_STYLE, MARGIN_STYLE
from src.similarity_package.full_pipeline import get_distance_score_by_group


def get_similarity_distance_fig(group_num: int, feature_weight_diagnosis: float, feature_weight_lab_result: float,
                                feature_weight_drugs: float, max_similar_patients: int):
    distances = \
        get_distance_score_by_group(group_num, float(feature_weight_diagnosis), float(feature_weight_lab_result),
                                    float(feature_weight_drugs), max_similar_patients)['distance'].reset_index()
    print(distances)
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


def drawText(text, subtext, id_):
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2(text),
                    html.H3(subtext),
                ], style={'textAlign': 'center', 'height': '20%'})
            ])
        ),
    ], style=MARGIN_STYLE, id=id_)


def get_cohort_summary_html_view():
    return html.Div(
        ([dbc.Row(dbc.Col(drawText(3, 'Women', 'text_1_'))),
          dbc.Row(dbc.Col(drawText(2, 'Men', 'text_2_'))),
          dbc.Row(dbc.Col(drawText(5, 'Total', 'text_3_'))),
          ],
         ), id='cohort-summary'
    )


def get_prediction_html_view():
    return html.Div(
        dbc.Row([
            # dbc.Col(get_cohort_summary_html_view()),
                 dbc.Col(html.Div([dcc.Graph(id='scatter_plot', clickData={'points': [{'customdata': '1'}]})],
                                  style=COMPONENT_STYLE),
                         width=4),
                 dbc.Col(
                     html.Div(
                         dcc.Loading(
                             id="loading",
                             type="circle",
                             children=html.Div(id='loading-output')
                         ),
                         style={'margin-top': '200px'}
                     ),
                     width=1
                 ),
                 dbc.Col(html.Div([dcc.Graph(id="pie_chart")], style=COMPONENT_STYLE),
                         width=4),

                 dbc.Col(html.Div([html.P("Diagnosis weight:"),
                                   dcc.Input(id='diagnosis_feature_weight', type='range', min=0, max=1, step=0.01,
                                             value=0.33),
                                   html.P("Lab Results weight:"),
                                   dcc.Input(id='lab_result_feature_weight', type='range', min=0, max=1, step=0.01,
                                             value=0.33),
                                   html.P("Drugs weight:"),
                                   dcc.Input(id='drug_feature_weight', type='range', min=0, max=1, step=0.01,
                                             value=0.33),
                                   daq.NumericInput(
                                       label='Max #Similar Patients',
                                       labelPosition='bottom',
                                       max=1000,
                                       min=1,
                                       value='All',
                                       size=150,
                                       id='max_similar_patients'
                                   )
                                   ], style={**MARGIN_STYLE, 'margin-top': '30px'}),
                         width=2),

                 ],
                ),
        id='loading-output-1'
    )


def register_prediction_callbacks(app):
    @app.callback(
        [Output("pie_chart", "figure"),
         Output('scatter_plot', 'figure'),
         Output("loading-output", 'children')],
        Input("diagnosis_feature_weight", "value"),
        Input("lab_result_feature_weight", "value"),
        Input("drug_feature_weight", "value"),
        Input('max_similar_patients', 'value'),
        Input('dropdown', 'value'))
    def generate_pie_chart(feature_weight_diagnosis: float, feature_weight_lab_result: float,
                           feature_weight_drugs: float, max_similar_patients: int, group_num: int, ):
        print("weights", feature_weight_lab_result, feature_weight_drugs, feature_weight_diagnosis)
        fig = px.pie(values=[feature_weight_diagnosis, feature_weight_lab_result, feature_weight_drugs],
                     names=['Diagnosis Features weight', 'Lab Results Features weight', 'Drug Features weight'],
                     hole=.3)

        return fig, get_similarity_distance_fig(group_num, feature_weight_diagnosis, feature_weight_lab_result,
                                                feature_weight_drugs, max_similar_patients), None

    # @app.callback(Output(component_id='scatter_plot', component_property='figure'),
    #               [Input(component_id='dropdown', component_property='value')])
    # def graph_update(group_num):
    #
