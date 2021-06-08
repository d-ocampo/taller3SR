import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import base64
import dash_table

# Data analytics library

import pandas as pd
import numpy as np
import plotly.express as px
import json

#import dfs
from layouts import peliculas, ratings,links, tags, users

spatial = html.Div([
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Usuario para recomendación",
                                            className="card-title"),
                                    html.P(id='recomend user')

                                ]
                            ),
                        ],
                    )
                ],
                className="mt-1 mb-2 pl-3 pr-3", lg="6", sm="12", md="auto"
            ),

            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Seleccione:",
                                            className="card-title"),
                                    html.P("Acá abajo puede seleccionar el usuario"),
                                    dcc.Dropdown(id='recomend drop',
                                                 options=[{'label': i, 'value': i} for i in users]
                                                 ),
                                ]
                            ),
                        ],
                    )
                ],
                className="mt-1 mb-2 pl-3 pr-3", lg="6", sm="12", md="auto"
            ),
        ],
    ),
    dbc.Col([
        dbc.Row([
            dbc.Card([
                dbc.CardBody([
                    dcc.Checklist(
                        options=[
                            {'label': 'Género', 'value': 'genero'},
                            {'label': 'Director', 'value': 'director'},
                            {'label': 'Actores', 'value': 'actores'}                        
                        ],
                        value='genero',
                        id='recomend relaciones'
                    )  
                ])
            ])
        ])
    ],className="mt-1 mb-2"),
 
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Cantidad de recomendaciones KNN",
                                            className="card-title"),
                                    dcc.Slider(id='recomend n_pred',
                                        min=1,
                                        max=20,
                                        step=1,
                                        value=5
                                   )

                                ]
                            ),
                        ],
                    )
                ],
                className="mt-1 mb-2 pl-3 pr-3", lg="6", sm="12", md="auto"
            ),

            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Recomendaciones basada en TAGS",
                                            className="card-title"),
                                    dcc.Slider(id='recomend n_sim',
                                        min=1,
                                        max=20,
                                        step=1,
                                        value=5
                                   )
                                    
                                ]
                            ),
                        ],
                    )
                ],
                className="mt-1 mb-2 pl-3 pr-3", lg="6", sm="12", md="auto"
            ),
        ],
    ),
    dbc.Row([
       dbc.Col([
           dcc.Graph(id='recomend grafo')
       ]) 
    ]),

    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    dash_table.DataTable(
                                        id='recomend table',
                                        columns=[{"name": "Negocios recomendados", "id": "Negocios recomendados"}]
                                    )
                                ]
                            ),
                        ],
                    )
                ],
                className="mt-1 mb-2 pl-3 pr-3", lg="6", sm="12", md="auto"
            ),

            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    dcc.Graph(id='recomend similar graph')
                                ]
                            ),
                        ],
                    )
                ],
                className="mt-1 mb-2 pl-3 pr-3", lg="6", sm="12", md="auto"
            ),
        ],
    ),




],
    className='container',
)
 