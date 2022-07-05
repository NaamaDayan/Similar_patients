import dash_daq as daq
import plotly.graph_objects as go
from dash import html, Input, Output, dash_table
import dash_bootstrap_components as dbc
from src.similarity_package.configs import TARGET_PATIENT_ID, DRUGS_SHEET_NAME, LAB_RESULTS_SHEET_NAME, \
    DIAGNOSES_SHEET_NAME, COMPONENT_STYLE, SHADOW_STYLE
from src.similarity_package.full_pipeline import get_data_by_id


def get_unique_patient_html_view():
    return html.Div(dbc.Row([dbc.Col(create_df_div('Diagnosis', DIAGNOSES_SHEET_NAME, 1), width='100%'),
                             dbc.Col(create_df_div('Lab Results', LAB_RESULTS_SHEET_NAME, 1), width='100%'),
                             dbc.Col(create_df_div('Drugs', DRUGS_SHEET_NAME, 1), width='100%')]
                            ))


def create_df_div(table_name: str, sheet_name, group_num) -> html.Div:
    df = get_data_by_id(group_num, 1, sheet_name=sheet_name)
    return html.Div([html.H2(id=f'H2_{table_name}', children=table_name,
                             style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 40}),
                     html.Div([daq.BooleanSwitch(
                         on=True,
                         id=f'{table_name}_shared_switch',
                         label="Only Shared",
                         labelPosition="top"
                     )], style={'margin-bottom': '0.2%'}),
                     html.Div([
                         dash_table.DataTable(
                             df.to_dict('records'), [{"name": i, "id": i} for i in df.columns],
                             id=f'{table_name}_table',
                             page_size=5)
                     ])  # , style={'width': '49%', 'display': 'inline-block'}),
                     # html.Div([dcc.Graph(id=f'{table_name}_shared')], style={'width': '59%', 'display': 'inline-block'})
                     ], style=SHADOW_STYLE)


def get_table_by_group_and_patient(click_data, is_only_shared: bool, group_num: int, sheet_name: str, col_name: str):
    patient_id = click_data['points'][0]['customdata']
    df = get_data_by_id(group_num, patient_id, sheet_name=sheet_name)
    if is_only_shared:
        df_target_patient_values = get_data_by_id(group_num, TARGET_PATIENT_ID, sheet_name=sheet_name)[
            col_name].unique()
        df = df[df[col_name].isin(df_target_patient_values)]
    return df.to_dict('records')


def get_shared_categories_by_group_and_patient(click_data, group_num, sheet_name, col_name):
    patient_id = click_data['points'][0]['customdata']

    diagnoses_df_examined_patient = get_data_by_id(group_num, patient_id,
                                                   sheet_name=sheet_name)
    diagnoses_df_target_patient = get_data_by_id(group_num, patient_id,
                                                 sheet_name=sheet_name)
    fig = go.Figure([go.Bar(x=['target', 'examined'], y=[len(diagnoses_df_target_patient[col_name].unique()),
                                                         len(diagnoses_df_examined_patient[col_name].unique())])])

    return fig


def register_callbacks(app):
    @app.callback(
        Output('Diagnosis_table', 'data'),
        Input('scatter_plot', 'clickData'),
        Input('Diagnosis_shared_switch', 'on'),
        Input(component_id='dropdown', component_property='value'))
    def update_table(click_data, is_only_shared: bool, group_num: int):
        return get_table_by_group_and_patient(click_data, is_only_shared, group_num,
                                              DIAGNOSES_SHEET_NAME,
                                              'Diagnosis')

    @app.callback(
        Output('Drugs_table', 'data'),
        Input('scatter_plot', 'clickData'),
        Input('Drugs_shared_switch', 'on'),
        Input(component_id='dropdown', component_property='value'))
    def update_table(clickData, is_only_shared: bool, group_num: int):
        return get_table_by_group_and_patient(clickData, is_only_shared, group_num, DRUGS_SHEET_NAME,
                                              'DrugName')

    @app.callback(
        Output('Lab Results_table', 'data'),
        Input('scatter_plot', 'clickData'),
        Input('Lab Results_shared_switch', 'on'),
        Input(component_id='dropdown', component_property='value'))
    def update_table(clickData, is_only_shared: bool, group_num: int):
        return get_table_by_group_and_patient(clickData, is_only_shared, group_num, LAB_RESULTS_SHEET_NAME,
                                              'TestName')

    # @app.callback(
    #     Output('Diagnosis_shared', 'figure'),
    #     Input('scatter_plot', 'clickData'),
    #     Input(component_id='dropdown', component_property='value'))
    # def update_figure(clickData, group_num):
    #     return get_shared_categories_by_group_and_patient(clickData, group_num, f'קבוצה {group_num} - פרטים ואבחנות',
    #                                                       col_name='Diagnosis')
    #
    #
    # @app.callback(
    #     Output('Lab Results_shared', 'figure'),
    #     Input('scatter_plot', 'clickData'),
    #     Input(component_id='dropdown', component_property='value'))
    # def update_figure(clickData, group_num):
    #     return get_shared_categories_by_group_and_patient(clickData, group_num, f'קבוצה {group_num} - בדיקות מעבדה',
    #                                                       col_name='TestName')
    #
    #
    # @app.callback(
    #     Output('Drugs_shared', 'figure'),
    #     Input('scatter_plot', 'clickData'),
    #     Input(component_id='dropdown', component_property='value'))
    # def update_figure(clickData, group_num):
    #     return get_shared_categories_by_group_and_patient(clickData, group_num, f'קבוצה {group_num} - תרופות',
    #                                                       col_name='DrugName')
    #
