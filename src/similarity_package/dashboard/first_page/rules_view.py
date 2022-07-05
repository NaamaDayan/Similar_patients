from typing import List

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
from src.similarity_package.configs import DATA_SOURCE_MAPPING, MARGIN_STYLE
from src.similarity_package.full_pipeline import get_df_by_sheet_name


def get_rules_html_view():
    return html.Div([
        dbc.Row(dbc.Col(html.Div(html.H2(id=f'rule_headline', children='Improve Algorithm by Defining Rules',
                                         style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 40})),
                        width=12)),
        dbc.Row([dbc.Col(dcc.Dropdown(id="table-dropdown",
                                      options=[
                                          {'label': 'Diagnosis', 'value': 'Diagnosis'},
                                          {'label': 'Lab Results', 'value': 'Lab Results'},
                                          {'label': 'Drugs', 'value': 'Drugs'},
                                      ],
                                      value='Diagnosis'
                                      ), width=3),
                 dbc.Col(dcc.Dropdown(id="positive-table-options", multi=True), width=3),
                 dbc.Col(dcc.Dropdown(id="negative-table-options", multi=True), width=3),
                 dbc.Col(html.Div(dbc.Button("Add Rule", id="add"), style={'textAlign': 'center'}), width=1),
                 dbc.Col(html.Div(dbc.Button("Remove Unneeded Rules", id="clear-done"), style={'textAlign': 'center'}),
                         width=2),
                 ]),
        dbc.Row(html.Div(id="list-container")),
    ], style=MARGIN_STYLE)


style_todo = {"margin-left": "40px"}
style_done = {"textDecoration": "line-through", "color": "#888"}
style_done.update(style_todo)


def generate_text(table_name: str, positive_items: List[str], negative_items: List[str]) -> str:
    print("generate text", table_name, positive_items, negative_items)
    return f"In Data source {table_name}, apply more significance for {positive_items} and less significance for {negative_items}"


def register_rules_callback(app):
    @app.callback(
        [
            Output("list-container", "children"),
            Output("table-dropdown", "value"),
            Output("positive-table-options", "value"),
            Output("negative-table-options", "value")
        ],
        [
            Input("add", "n_clicks"),
            Input("table-dropdown", "n_submit"),
            Input("clear-done", "n_clicks")
        ],
        [
            State("table-dropdown", "value"),
            State("positive-table-options", "value"),
            State("negative-table-options", "value"),
            State({"index": ALL}, "children"),
            State({"index": ALL, "type": "done"}, "value")
        ]
    )
    def edit_list(add, add2, clear, table_dropdown, positive_items: List[str], negative_items: List[str],
                  rules,
                  rules_done):
        print(table_dropdown)
        positive_items = positive_items if positive_items else None
        negative_items = negative_items if negative_items else None
        triggered = [t["prop_id"] for t in dash.callback_context.triggered]
        adding = len([1 for i in triggered if i in ("add.n_clicks", "new-item.n_submit")])
        clearing = len([1 for i in triggered if i == "clear-done.n_clicks"])
        new_spec = [(rules_list, done) for rules_list, done in zip(rules, rules_done) if not (clearing and done)]
        if adding:
            new_spec.append((generate_text(table_dropdown, positive_items, negative_items), []))
        print("new spec", new_spec)
        new_list = [
            html.Div([
                dcc.Checklist(
                    id={"index": i, "type": "done"},
                    options=[{"label": "", "value": "done"}],
                    value=done,
                    style={"display": "inline", 'font-size': 20, 'margin-top': '10px'},
                    labelStyle={"display": "inline"}
                ),
                html.Div(text, id={"index": i}, style=style_done if done else {})
            ], style={"clear": "both"})
            for i, (text, done) in enumerate(new_spec)
        ]


        new_dropdowns_values = [None, None, None] if adding else table_dropdown, positive_items, negative_items
        return new_list, 'Diagnosis', None, None

    @app.callback(
        Output({"index": MATCH}, "style"),
        Input({"index": MATCH, "type": "done"}, "value")
    )
    def mark_done(done):
        return style_done if done else style_todo

    @app.callback(
        Output("positive-table-options", "options"),
        Output("negative-table-options", "options"),
        [Input("table-dropdown", "value"), Input(component_id='dropdown', component_property='value')]
    )
    def table_options(table_name: str, group_num: int):
        try:
            df = get_df_by_sheet_name(group_num, DATA_SOURCE_MAPPING[table_name].sheet_name)
            values = df[DATA_SOURCE_MAPPING[table_name].col_name].unique()
            options = [{"label": i, "value": i} for i in values]
            return options, options
        except KeyError:
            if table_name != None:
                print(get_df_by_sheet_name(group_num, DATA_SOURCE_MAPPING[table_name].sheet_name))
            print("key error", table_name)
            return None, None
