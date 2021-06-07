import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import base64
from app_ import app
from dash import callback_context as ctx


from dash.dependencies import Input, Output, State
# Data analytics library

import os
import pandas as pd
import numpy as np
import plotly.express as px
import json

# Surprise libraries



# cargar modelos

#export libraries

# from sklearn.externals import joblib
import joblib
import pickle

#graph libraries
import plotly.graph_objects as go
import networkx as nx
import plotly
import random


#Resources

#Cargar la ruta

ruta=os.getcwd()+'/Data/'

####Funciones 

####Datos

# valor de la cantidad de datos a cargar
n=100000

# Users

#Diccionario nombre de negocios


top_cards = dbc.Row([
        dbc.Col([dbc.Card(
            [
                dbc.CardBody(
                    [
                        # html.Span(html.I("add_alert", className="material-icons"),
                        #           className="float-right rounded w-40 danger text-center "),
                        html.H5(
                            "Cantidad total de usuarios", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(len(review_df['user_id'].unique())))),
                    ],

                    className="pt-2 pb-2 box "
                ),
            ],
            #color="warning",
            outline=True,
            #style={"width": "18rem"},
        ),
        ],
            className="col-xs-12 col-sm-6 col-xl-3 pl-3 pr-3 pb-3 pb-xl-0"
        ),
        dbc.Col([dbc.Card(
            [

                dbc.CardBody(
                    [
                        html.H5(
                            "Cantidad de negocios", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(len(review_df['business_id'].unique())))),

                     ],

                    className="pt-2 pb-2 box"
                ),
            ],
            # color="success",
            outline=True,
            #style={"width": "18rem"},
        ),
        ],

            className="col-xs-12 col-sm-6 col-xl-3 pl-3 pr-3 pb-3 pb-xl-0"
        ),
        dbc.Col([dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H5(
                            "Prom. palabras por reseña", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(users_df['review_count'].median()))),
                    ],

                    className="pt-2 pb-2 box"
                ),
            ],
            # color="info",
            outline=True,
            #style={"width": "18rem"},
        ),
        ],

            className="col-xs-12 col-sm-6 col-xl-3 pl-3 pr-3 pb-3 pb-xl-0"
        ),
        dbc.Col([dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H5(
                            "Cantidad de reseñas", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(review_df['review_id'].count()))),
                    ],

                    className="pt-2 pb-2 box"
                ),
            ],
            # color="warning",
            outline=True,
            #style={"width": "18rem"},
        ),
        ],

            className="col-xs-12 col-sm-6 col-xl-3 pl-3 pr-3 pb-3 pb-xl-0"
        ),


    ],
        className="mt-1 mb-2"

    )


home = html.Div([
    # dbc.Jumbotron(
    #     [
    #         html.Img(src="/assets/images/francebanner.webp",
    #                  className="img-fluid")
    #     ], className="text-center"),


    dbc.Row(

        dbc.Col([
#banner del home
            html.I(className="fa fa-bars",
                   id="tooltip-target-home",
                   style={"padding": "1rem", "transform" : "rotate(90deg)", "font-size": "2rem", "color": "#999999"}, ),
# Descripción del problema
            html.P('''
                    El conjunto de datos de Yelp es un subconjunto de negocios, reseñas y datos de usuario para su uso con fines personales, educativos y académicos. Disponible como archivos JSON, incluye
                    cuatro tablas principales con datos de negocios, reseñas, usuarios, fotos, checkins y reseñas cortas.
                   ''',
            style = { "font-color": "#666666", "font-size": "16px", "margin": "1rem auto 0", "padding": "0 12rem"}, className="text-muted"
            
            ),

            html.P('''Licencia: Los datos contenidos en esta herramienta son distribuidos y manipulados con permiso de Yelp. Los datos se encuentran disponibles para su uso no comercial. Para más información, se sugiere revisar los términos de servicio de Yelp (https://www.yelp.com/dataset/).''', style = { "font-color": "#666666", "font-size": "16px", "margin": "1rem auto 0", "padding": "0 12rem"}, className="text-muted"),

            html.Hr(style = {"width" : "100px", "border": "3px solid #999999", "background-color": "#999999", "margin": "3rem auto"}),

        ],
        style = {"text-align": "center"},
        ),
    ),

    dbc.Container(
        [

            dbc.CardGroup([
                dbc.Card(
                    [
                        dbc.CardImg(
                            src="/assets/images/dashboard.jpeg", top=True),
                        dbc.CardBody(
                            [
                                html.H3("Dashboard", style = {"color": "#66666"}),
                                html.P(
                                    '''Un espacio para obtener estadísticas básicas de usuarios, negocios, reseñas e interacciones, junto a algunos insights sobre sus calificaciones.
                                    
                                    ''',
                                    className="card-text", style = {"font-size": "15px"},
                                ),
                                dbc.Button(
                                    "Dashboard", color="primary", href="/page-5"),
                            ],
                            className="text-center"
                        ),
                    ],
                    style={"width": "18rem", "margin": "0 1rem 0 0"},
                ),
                dbc.Card(
                    [
                        dbc.CardImg(
                            src="/assets/images/spatial_model.jpeg", top=True),
                        dbc.CardBody(
                            [

                                html.H3("Recomendación", style = {"color": "#66666"}),

                                html.P(
                                    '''Acá puedes encontrar el sistema de recomendación híbrido basado en la combinación de modelos colaborativos, de contenido, con factorización,...''',
                                    className="card-text", style = {"font-size": "15px"},
                                ),
                                dbc.Button("Sistema de recomendación",
                                           color="primary", href="/page-2"),    
                            ],
                            className="text-center"
                        ),
                    ],
                    style={"width": "18rem"},
                ),

                # dbc.Card(
                #     [
                #         dbc.CardImg(
                #             src="/assets/images/map.png", top=True),
                #         dbc.CardBody(

                #             [  html.H3("Exploración por usuarios", style = {"color": "#66666"}),

                #                 html.P(
                #                     '''
                #                     Finalmente, un apartado con las predicciones y el sistema diseñado para obtener recomendaciones de cualquier usuario en el sistema.
                #                     ''',
                #                     className="card-text", style = {"font-size": "15px"},
                #                 ),

                #                 dbc.Button("Exploration", color="primary",
                #                            href="/page-3", style={"align": "center"}),
                #             ],
                #             className="text-center"
                #         ),
                #     ],
                #     style={"width": "18rem", "margin": "0 0 0 1rem"},                
                #     )

            ]),

            html.Hr(style = {"width" : "100px", "border": "3px solid #999999", "background-color": "#999999", "margin": "3rem auto"}),

            dbc.Row(


                dbc.Col(
                
               
                html.H1("PARTNERS"),
                style = {"align": "center", "color": "#66666", "margin" : "0 auto 2rem"},
                className="text-center",


                ),

            ),

            dbc.Row ([

                dbc.Col (

                    html.Img(src="/assets/images/uniandes.png", className="img-fluid"),
                    className = "d-flex justify-content-center align-items-center",


                ),          


            ], 
            style = {"padding" : "0 0 5rem"}),
        ]

    )

])

dashboard = html.Div([

    top_cards,
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [


                                    html.H5("Cantidad de reseñas por calificación",
                                            className="card-title"),
                                    
                                    
                                    dcc.Graph(
                                        id='dashboard_hist_user',

                                        
                                        
                                        figure=px.bar(rev_stars, y = "stars", labels = {'index': 'stars', 'stars' : 'cantidad'})),
                                    
                                ]
                            ),
                        ],
                    )
                ],
                className="mt-1 mb-2 pl-3 pr-3"
            ),
        ],
    ),

    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H5("Cantidad de reseñas por negocio",
                                            className="card-title"),

                                    dcc.Graph(figure = px.histogram(review_df.groupby('business_id').agg({'stars':'count'}), x="stars", labels = {'stars' : 'reviews'})),
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
                                    html.H5("Relación entre estrellas y sentimientos",
                                            className="card-title"),

                                    dcc.Graph(figure =px.bar(rev_feel, x="stars", y="prom", color = 'feeling')),
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




aboutus = html.Div([
    dbc.CardDeck([
        dbc.Card([
            html.Div([
                 dbc.CardImg(src="assets/images/profiles/ocampo.jpg",
                             top=True, className="img-circle", style = {"margin-top": "1.125rem"}),
                 dbc.CardBody([
                     html.H4("David Ocampo",
                             className="card-title m-a-0 m-b-xs"),
                     html.Div([
                         html.A([
                                html.I(className="fa fa-linkedin"),
                                html.I(className="fa fa-linkedin cyan-600"),
                                ], className="btn btn-icon btn-social rounded white btn-sm", 
                                href="https://www.linkedin.com/in/david-alejandro-o-710247163/"),

                         html.A([
                             html.I(className="fa fa-envelope"),
                             html.I(className="fa fa-envelope red-600"),
                         ], className="btn btn-icon btn-social rounded white btn-sm", 
                            href="mailto:daocampol@unal.edu.co"),

                     ], className="block clearfix m-b"),
                     html.P(
                         "Statistician at Allianz. Universidad Nacional. Universidad de Los Andes.",
                         className="text-muted",
                     ),
                 ]
                 ),
                 ],
                className="opacity_1"
            ),
        ],
            className="text-center",

        ),
        dbc.Card([
            html.Div([
                dbc.CardImg(src="/assets/images/profiles/quinonez.png",
                            top=True, className="img-circle", style = {"margin-top": "1.125rem"}),
                dbc.CardBody([
                    html.H4("Juan David Quiñonez",
                            className="card-title m-a-0 m-b-xs"),
                    html.Div([
                        html.A([
                            html.I(className="fa fa-linkedin"),
                            html.I(className="fa fa-linkedin cyan-600"),
                        ], className="btn btn-icon btn-social rounded white btn-sm", href="https://www.linkedin.com/in/juandavidq/"),

                        html.A([
                            html.I(className="fa fa-envelope"),
                            html.I(className="fa fa-envelope red-600"),
                        ], className="btn btn-icon btn-social rounded white btn-sm", href="mailto:jdquinoneze@unal.edu.co"),

                    ], className="block clearfix m-b"),
                    html.P(
                        "Statistician at BBVA. Universidad Nacional. Universidad de Los Andes.",
                        className="text-muted",
                    ),
                ]
                ),
            ],
                className="opacity_1"
            ),
        ],
            className="text-center",
        ),
    ]),
])
