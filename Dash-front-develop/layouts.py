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

#api de imdb
from imdb import IMDb
ia = IMDb()

#similaridad del coseno
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 


#librerías sistema de recomendación
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy


#graph libraries
import plotly.graph_objects as go
import networkx as nx
import plotly
import random

import pickle


#Resources

#Cargar la ruta

ruta=os.getcwd()+'/Data/'

# valor de la cantidad de datos a cargar
n=100000

## Importar archivos
peliculas=pd.read_csv(ruta+'movies.csv',nrows=n)
ratings=pd.read_csv(ruta+'ratings.csv',nrows=n)

links=pd.read_csv(ruta+'links.csv',nrows=n)
tags=pd.read_csv(ruta+'tags.csv',nrows=n)

#nube de palabras
words = peliculas['genres'].str.cat(sep="|").split("|")
words_count = len(set(words))

#contar los géneros
freqs = {}
for word in words:
    freqs[word] = freqs.get(word, 0) + 1 # fetch and increment OR initialize
genres_df = pd.DataFrame(list(freqs.items()), columns=["Género", "Cantidad"]).sort_values("Cantidad", ascending=False)
# agrupar los tags
tags_df = tags.groupby(['tag', 'movieId']).size().reset_index(name='counts')
tags_df = tags_df.groupby(['tag']).size().reset_index(name='películas')
tags_df = tags_df.sort_values(by="películas", ascending=False)
tags_df.to_csv("test.csv", index = pickle.FALSE)


#sistema de recomendación
reader = Reader( rating_scale = ( 0, 5 ) )
#Se crea el dataset a partir del dataframe
surprise_dataset = Dataset.load_from_df( ratings[ [ 'userId', 'movieId', 'rating' ] ], reader )

# #Se crea el dataset para modelo 
# rating_data=surprise_dataset.build_full_trainset()
# # Se crea dataset de "prueba" con las entradas faltantes para generar las predicciones
# test=rating_data.build_anti_testset()

#usar modelo pequeño para entrenar
rating_data, test=  train_test_split(surprise_dataset, test_size=.2)

#usuarios para la predicción
users=list(ratings['userId'].unique())

# se crea un modelo knnbasic item-item con similitud coseno 
sim_options = {'name': 'cosine',
               'user_based': False  # calcule similitud item-item
               }
algo = KNNBasic(k=20, min_k=2, sim_options=sim_options)

#ajustar algoritmo y crear matriz de predicciones
algo.fit(rating_data)
predictions=algo.test(test)


# función para revisar las n predicciones de los usuarios
def prediccion_usuario(user,n): 
    user_predictions=list(filter(lambda x: x[0]==user,predictions))
    user_predictions.sort(key=lambda x : x.est, reverse=True)
    pred=user_predictions[0:n]
    return [i[1] for i in pred]

##########################
#### TAGS 
##########################

## data tags para similitud de coseno por tags
tags_peliculas=tags.groupby(['movieId'])['tag'].apply(','.join).reset_index()

# Crear la matrix de términios de reviews    
tfidf = TfidfVectorizer(ngram_range=(1, 1), min_df=0.0001, stop_words='english')
tfidf_matrix = tfidf.fit_transform(tags_peliculas['tag'])


# Calcular similirdad del coseno de las demás películas
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
results = {}
for idx, row in tags_peliculas.iterrows():
   similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 
   similar_items = [(cosine_similarities[idx][i], tags_peliculas['movieId'][i]) for i in similar_indices] 
   results[row['movieId']] = similar_items[1:]

#recomendación del negocio    
def peliculas_similares(pelicula,n):
    if pelicula in results.keys():
        recomendacion=results[pelicula] 
        recomendacion.sort(key=lambda x: x[0],reverse=True) 
        return [i[1] for i in recomendacion[0:n]]
    else:
        return [] 


#########################
#### características de la película
#########################

# diccionario imbd
imbd_dict={}
for i in range(len(links)):
    imbd_dict[links.at[i,'movieId']]=links.at[i,'imdbId']
    
# función para devolver películas    
def ctas_pelicula(pelicula):
    movie=ia.get_movie(str(imbd_dict[pelicula]))
    pelicula_dict={}
    pelicula_dict['director']=[director['name'] for director in movie['directors']]
    pelicula_dict['genero']=[genre for genre in movie['genres']]
    pelicula_dict['actores']=[actores['name'] for actores in movie['cast']]
    return pelicula_dict


# crear la red de películas para el grafo con los SR
def crear_red(usuario,n_pred,n_sim,relaciones): 
    G = nx.Graph() # crear un grafo
    G.add_nodes_from(prediccion_usuario(usuario,n_pred))
    ejes=[]
    for i in prediccion_usuario(usuario,n_pred):
        if len(peliculas_similares(i,n_sim))>0:
            for j in peliculas_similares(i,n_sim):
                ejes.append((i,j))
    G.add_edges_from(ejes)
    ejes_ctas=[]
    for i in G.nodes:
        for key in relaciones:
            if len(ctas_pelicula(i)[key])>0:
                for j in ctas_pelicula(i)[key]:
                    ejes_ctas.append((i,j))
    G.add_edges_from(ejes_ctas)
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    for i in G.nodes:
        G.nodes[i]['pos']=pos[i]
    return G

# diccionario con el nombre de las películas, llave es el ID
pelicula_dict={}
for i in range(len(peliculas)):
    pelicula_dict[peliculas.at[i,'movieId']]=peliculas.at[i,'title']
    
# elegir un nombre del nodo
def nombre_nodo(identificador):
    if identificador in pelicula_dict.keys():
        return pelicula_dict[identificador]
    else:
        return identificador

# realizar el gráfico de red    
def grafico_red(G):
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Cantidad de conexiones',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
    #     node_text.append(str(nombre_nodo(node))+'# de conexiones: '+str(len(adjacencies[1])))
        node_text.append(str(nombre_nodo(node))+' # de conexiones: '+str(len(adjacencies)))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Grafo de SR de películas',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig


def predi_similares(user,n_pred,n_sim):
    similares=[]
    for i in prediccion_usuario(user,n_pred):
        if len(peliculas_similares(i,n_sim))>0:
            for j in peliculas_similares(i,n_sim):
                similares.append(j)
    similares=np.unique(similares)
    return similares

top_cards = dbc.Row([
        dbc.Col([dbc.Card(
            [
                dbc.CardBody(
                    [
                        # html.Span(html.I("add_alert", className="material-icons"),
                        #           className="float-right rounded w-40 danger text-center "),
                        html.H5(
                            "Cantidad total de usuarios", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(len(ratings['userId'].unique())))),
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
                            "Cantidad de películas", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(len(peliculas['movieId'].unique())))),

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
                            "Cantidad de géneros", className="card-title text-muted font-weight-normal mt-2 mb-3 mr-5"),
                        html.H4(children = str('{:,}'.format(words_count))),
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
                        html.H4(children = str('{:,}'.format(ratings.shape[0]))),
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
                    MovieLens, un proyecto de recomendaciones personalizadas de películas impulsado por el grupo de investigación GroupLens de la Universidad de Minnesota, ofrece un conjunto de datos (reseñas, películas, enlaces,...) con lo que han obtenido de sus usuarios hasta septiembre de 2018. 
                   ''',
            style = { "font-color": "#666666", "font-size": "16px", "margin": "1rem auto 0", "padding": "0 12rem"}, className="text-muted"
            
            ),

            html.P('''Licencia: Los datos contenidos en esta herramienta son distribuidos y manipulados con permiso de MovieLens. Los datos se encuentran disponibles para su uso no comercial. Para más información, se sugiere revisar los términos de servicio de GroupLens (https://grouplens.org/datasets/movielens/ y https://bit.ly/3w4tksS). 

                   ''', style = { "font-color": "#666666", "font-size": "16px", "margin": "1rem auto 0", "padding": "0 12rem"}, className="text-muted"),

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
                                    '''Un espacio para obtener estadísticas básicas de usuarios, películas, ratings, tags e interacciones, junto a algunos insights sobre sus calificaciones. 

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
                                    '''Acá puedes encontrar el sistema de recomendación híbrido el cual combina modelos colaborativos con enriquecimiento semántico y filtraje ontológico. 
                                    ''',
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


                                    html.H5("Cantidad de películas por género",
                                            className="card-title"),
                                    
                                    
                                    dcc.Graph(
                                        id='dashboard_hist_user',

                                        
                                        
                                        figure=px.histogram(ratings, x = "rating", labels = {'index': 'cantidad', 'rating' : 'rating'})),
                                    
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

                                    dcc.Graph(figure = px.histogram(genres_df, x = "Género", y = "Cantidad")),
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
                                    html.H5("Tags más populares según sus películas",
                                            className="card-title"),
                                    html.Br(),
                                    html.Br(),
                                    html.Br(),        
                                    html.Div(html.Img(src=app.get_asset_url('wordcloud.png'), style={'height':'200%', 'width':'100%'})),
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
