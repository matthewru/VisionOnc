import dash
from dash import dcc
from dash import html

import pandas as pd
import numpy as np
from dash import no_update

from sklearn import preprocessing
from dash.dependencies import Input, Output, State
from plotly import graph_objs as go
from plotly.graph_objs import *
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.tools as tls

from lifelines import KaplanMeierFitter
from lifelines import AalenJohansenFitter
import dash_daq as daq
from dash import dash_table
from datetime import datetime

import glob, os
import base64
import pgeocode
from PIL import Image, ImageDraw
import io

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

app = dash.Dash(
    __name__, 
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    suppress_callback_exceptions=True
)
server = app.server

app.title = 'Clinical Trial Dynamic Data Visualization'

# set up a min max scaler
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(50,500))

# set UCLA blue colors
uclaBlue = 'rgba(39, 116, 174, .9)'
uclaLightBlue = 'rgba(128, 197, 250, 0.55)'

# graph types
graphs = [
    'Scatter',
    'Line',
    'Bar',
    'Histogram',
]

# filter types
filters = [
    'Treatment End Date',
    'Line',
    'Bar',
    'Histogram',
]

# Layout of Dash App
app.layout = html.Div(
    children=[
        dcc.Store(id='uploaded-data'),
        dcc.Store(id='field-types'),
        dcc.Store(id='save-timestamp-store'),  # Add this new store
        html.Div(
            className="row",
            children=[
                # Column for left logo
                html.Div(
                    className="three columns div-user-controls bg-grey-copy",
                    children=[
                        html.Div(
                            className="row",
                            children=[
                                html.Img(
                                    className="logo", 
                                    src="assets/UCLA_H_2019_RGB.png"
                                ),                                
                            ],
                        ),
                    ]
                ),
                # Initial title
                html.Div(
                    className="six columns div-for-title bg-grey",
                    children=[
                        html.H1("Clinical Trial Dynamic Data Visualization"),
                    ]
                )
            ]
        ),

        html.Div(
            children=[
                html.Div(
                    className="row",
                    children=[    
                        dcc.Tabs(
                            id="tabs",
                            parent_className='custom-tabs', 
                            className='custom-tabs-container', 
                            content_className='custom-tab-content',
                            colors={
                                "border": "#d6d6d6",
                                "primary": "rgba(39, 116, 174, .9)",
                                "background": "#f9f9f9"
                            },
                            style={
                                'width': '100%',
                                'fontSize': '1.2rem',
                                'height': 'auto'
                            },
                            value="tab-1", # Set the default tab
                            children=[

                            # Data Upload Tab
                            dcc.Tab(
                                id="tab-1",
                                value="tab-1",
                                label='Data Upload', 
                                className='custom-tab', 
                                selected_className='custom-tab--selected',
                                style={'padding': '12px 18px'},
                                selected_style={'padding': '12px 18px'},
                                children=[
                                    html.Div(
                                        style={
                                            'maxWidth': '1200px',
                                            'margin': '0 auto',
                                            'padding': '32px 16px',
                                            'backgroundColor': '#f9f9f9',
                                            'borderRadius': '8px',
                                            'boxShadow': '0 2px 8px rgba(0,0,0,0.04)'
                                        },
                                        children=[
                                            html.H4("Upload Your Data:"),
                                            html.P("""
                                                Welcome to our dynamic data visualization interface. This tool allows you to upload your own CSV data 
                                                and explore it through various interactive plots and visualizations. Simply upload your data file 
                                                and the system will automatically detect numerical and categorical variables for plotting.
                                            """),
                                            dcc.Upload(
                                                id='upload-data',
                                                children=html.Div([
                                                    'Drag and Drop or ',
                                                    html.A('Select a CSV File')
                                                ]),
                                                style={
                                                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                                                    'textAlign': 'center', 'margin': '10px 0 24px 0'
                                                },
                                                multiple=False
                                            ),
                                            html.Div(id='upload-status'),
                                            html.Div(id='field-selection-ui'),
                                        ]
                                    )
                                ]),

                            # Instructions Tab
                            dcc.Tab(
                                id="tab-2",
                                value="tab-2",
                                label='Instructions', 
                                className='custom-tab', 
                                selected_className='custom-tab-instructions--selected',
                                style={'padding': '12px 18px'},
                                selected_style={'padding': '12px 18px'},
                                children=[
                                
                                # Middle Graph
                                html.Div(
                                    className="eight columns div-for-charts bg-grey",
                                    children=[        
                                        html.Br(),

                                        html.H4("Interactive Data Visualization Tool:"),

                                        html.P(
                                            """

                                            Welcome to our dynamic data visualization interface. This tool allows you to upload your own CSV data 
                                            and explore it through various interactive plots and visualizations. Simply upload your data file 
                                            and the system will automatically detect numerical and categorical variables for plotting.

                                            """
                                        ),

                                        html.Br(),

                                        html.P(
                                            """

                                            The interactive tools developed here allow you to explore relationships in your data by dynamically 
                                            selecting variables to plot, types of graphs, groupings, and markers/sizes to customize. 
                                            In the graph tabs, any drop-down box in red can be customized to adjust the plots to explore the data.

                                            """
                                        ),                                         

                                        html.Br(),

                                        html.P(
                                            """
                                            
                                            Additionally, select the filters tab and type in expressions such as:
                                            
                                            """, 

                                        ),  

                                        html.Br(),

                                        html.Ul(id='my-list', children=[
                                            html.Li('>'),
                                            html.Li('<'),
                                            html.Li('='),
                                            html.Li('!='),
                                            html.Li('>= 2020-01'),
                                            ]
                                        ),  

                                        html.Br(),

                                        html.P(
                                            """
                                            
                                            This will filter your data to meet each criteria across all columns. The right hand gauge will indicate 
                                            how many data points are remaining for the analyses.
                                            
                                            """, 

                                        ),                                                                                                                           

                                        html.Br(),                                        

                                        html.H4(
                                            """
                                            
                                            Click on the tabs above to explore your data. 
                                            
                                            """, 
                                        ),    

                                        html.Br(),                                        

                                        html.P(
                                            """
                                            
                                            Note: For best browsing experience, we recommend using Chrome, Firefox, or Safari web browsers.
                                            
                                            """, 
                                        ),                                              

                                        html.Br(),
                                    ]
                                ),

                                # Right controls
                                html.Div(
                                    className="four columns div-user-controls bg-grey-copy",
                                    children=[

                                        # Change to side-by-side for mobile layout
                                        html.Div(
                                            className="row",
                                            children=[

                                                html.Br(),html.Br(),

                                                html.H4("Data Status:"),                                    

                                                html.Br(),                                                          
                              
                                                html.Div(id='data-status-display'),

                                            ],
                                        ),                                                    
                                    ],
                                ),
                            ]),

                            # Patient Filters
                            dcc.Tab(
                                id="tab-3",
                                value="tab-3",
                                label='Patient Filters', 
                                className='custom-tab', 
                                selected_className='custom-tab-filters--selected',
                                style={'padding': '12px 18px'},
                                selected_style={'padding': '12px 18px'},
                                children=[
                                
                                # Full width table section
                                html.Div(
                                    className="twelve columns div-for-charts bg-grey",
                                    children=[        
                                        html.Br(),

                                        html.Div([
                                            html.H4("Apply Filters to the data table: ", style={'display': 'inline-block', 'margin-right': '10px'}),
                                            html.Span(id='patient-count-display', style={'font-size': '18px', 'font-weight': 'bold', 'color': uclaBlue})
                                        ], style={'margin-bottom': '10px'}),

                                        html.Div([
                                            html.Span("Total patients in dataset: ", style={'font-size': '16px', 'color': '#666'}),
                                            html.Span(id='total-patient-count-display', style={'font-size': '16px', 'font-weight': 'bold', 'color': '#666'})
                                        ], style={'margin-bottom': '20px'}),

                                        html.Div(
                                            children=[
                                            # data table                                        
                                                dash_table.DataTable(
                                                        id='table-filters',
                                                        columns=[],
                                                        data=[],
                                                        style_data={
                                                            'whiteSpace': 'normal',
                                                            'height': 'auto',
                                                            'overflow': 'hidden',
                                                            'textOverflow': 'ellipsis',
                                                            'fontSize': 12,
                                                            'padding': '6px',
                                                        },
                                                        style_cell={
                                                            'textAlign': 'left',
                                                            'maxWidth': '150px',
                                                            'minWidth': '80px',
                                                            'overflow': 'hidden',
                                                            'textOverflow': 'ellipsis',
                                                        },
                                                        fixed_columns={'headers': True, 'data': 1},
                                                        editable=True,
                                                        filter_action="native",
                                                        sort_action="native",
                                                        sort_mode="multi",
                                                        column_selectable="multi",
                                                        # row_selectable="multi",
                                                        row_deletable=True,
                                                        selected_columns=[],
                                                        selected_rows=[],
                                                        page_action="native",
                                                        page_current= 0,
                                                        page_size= 15,
                                                        style_table={
                                                            'overflowX': 'auto',
                                                            'width': '100%',
                                                            'maxWidth': '100%',
                                                        },
                                                        style_header={
                                                            'backgroundColor': uclaBlue,
                                                            'fontWeight': 'bold',
                                                            'color': 'white',
                                                            'font': 'Open Sans',
                                                            'fontSize': 14,
                                                            'padding': '6px',
                                                            'overflow': 'hidden',
                                                            'textOverflow': 'ellipsis',
                                                        },
                                                        style_filter={
                                                            'backgroundColor': uclaLightBlue,
                                                            'fontWeight': 'bold',
                                                            'font': 'Open Sans',
                                                            'textColor': 'red',
                                                            'fontSize': 12,
                                                            'padding': '4px',
                                                        },
                                                        tooltip_data=[],
                                                        tooltip_duration=None,
                                                    ),
                                            ],
                                        ),

                                        html.Br(),
                                    ]
                                ),
                            ]),

                            dcc.Tab(
                                id="tab-4",
                                value="tab-4",
                                label='Scatter Plots', 
                                className='custom-tab', 
                                selected_className='custom-tab--selected',
                                style={'padding': '12px 18px'},
                                selected_style={'padding': '12px 18px'},
                                children=[
                                
                                # Column for left controls
                                html.Div(
                                    className="three columns div-user-controls bg-grey-copy",
                                    children=[

                                        # Change to side-by-side for mobile layout
                                        html.Div(
                                            className="row",
                                            children=[                        

                                                html.Br(),html.Br(),

                                                html.H3("Y-axis:"),

                                                html.Div(
                                                    className="div-for-dropdown",
                                                    children=[
                                                        # Dropdown for clinical T-Stage
                                                        dcc.Dropdown(
                                                            id="y-dropdown",
                                                            options=[],
                                                            value=None,
                                                            style=  {'borderColor': 'red', 'borderWidth': '3px'},
                                                        )
                                                    ],
                                                ),
                                            ]
                                        )
                                    ]
                                ),

                                # Middle Graph
                                html.Div(
                                    style={'position': 'relative'},
                                    className="six columns div-for-charts bg-grey",
                                    children=[        
                                        html.Br(),
                                        dcc.Graph(id="scatter-plot", 
                                                  style={'position': 'relative'}),
                                        dcc.Tooltip(
                                            id="scatter-tooltip", 
                                            direction='right', 
                                            background_color='white', 
                                            border_color='red', 
                                            style={
                                                'position': 'absolute',    
                                                'zIndex': 10000,
                                                'pointerEvents': 'none'
                                            }),
                                        

                                        html.H3("X-axis:"),

                                        html.Div(
                                            className="div-for-dropdown",
                                            children=[
                                            # Dropdown for clinical T-Stage
                                            dcc.Dropdown(
                                                id="x-dropdown",
                                                options=[],
                                                value=None,
                                                style=  {'borderColor': 'red', 'borderWidth': '3px'},
                                            )],
                                        ),

                                        html.Br(),
                                    ]
                                ),

                                # Right controls
                                html.Div(
                                    className="three columns div-user-controls bg-grey-copy",
                                    children=[

                                        # Change to side-by-side for mobile layout
                                        html.Div(
                                            className="row",
                                            children=[

                                                html.Br(),html.Br(),

                                                html.H3("Color:"),

                                                html.Div(
                                                    className="div-for-dropdown",
                                                    children=[
                                                        # Dropdown for clinical T-Stage
                                                        dcc.Dropdown(
                                                            id="color-dropdown",
                                                            options=[],
                                                            value=None,
                                                            style=  {'borderColor': 'red', 'borderWidth': '3px'},
                                                        )
                                                    ],
                                                ),

                                                html.Br(),

                                                html.H3("Size:"),

                                                html.Div(
                                                    className="div-for-dropdown",
                                                    children=[
                                                        # Dropdown for clinical T-Stage
                                                        dcc.Dropdown(
                                                            id="size-dropdown",
                                                            options=[],
                                                            value=None,
                                                            style=  {'borderColor': 'red', 'borderWidth': '3px'},
                                                        )
                                                    ],
                                                ),
                

                                                html.Br(),                                                           
                              
                                            ],
                                        ),                                                    
                                    ],
                                ),
                            ]),

                            dcc.Tab(
                                id="tab-5",
                                value="tab-5",
                                label='Histograms',  
                                className='custom-tab', 
                                selected_className='custom-tab--selected',
                                style={'padding': '12px 18px'},
                                selected_style={'padding': '12px 18px'},
                                children=[
                                
                                # Column for left controls
                                html.Div(
                                    className="two columns div-user-controls bg-grey-copy",
                                    children=[

                                        # Change to side-by-side for mobile layout
                                        html.Div(
                                            className="row",
                                            children=[                        

                                                html.Br(),html.Br(),

                                                html.H3("Y-axis:"),

                                                html.H2("Count"),

                                                html.Br()

                                                # html.Div(
                                                #     className="div-for-dropdown",
                                                #     children=[
                                                #         # Dropdown for clinical T-Stage
                                                #         dcc.Dropdown(
                                                #             id="y-histo",
                                                #             options=[
                                                #                 {"label": col, "value": col}
                                                #                 for col in df.columns
                                                #             ],
                                                #             value="V12 Skin Total ",
                                                #             style=  {'borderColor': 'red', 'borderWidth': '3px'},
                                                #         )
                                                #     ],
                                                # ),
                                            ]
                                        )
                                    ]
                                ),

                                # Middle Graph
                                html.Div(
                                    className="eight columns div-for-charts bg-grey",
                                    children=[        
                                        html.Br(),

                                        dcc.Graph(id="histo-plot", style={"width": "100%"}),

                                        html.H3("X-axis:"),


                                        html.Div(
                                            className="div-for-dropdown",
                                            children=[
                                            # Dropdown for clinical T-Stage
                                            dcc.Dropdown(
                                                id="x-histo",
                                                options=[],
                                                value=None,
                                                style=  {'borderColor': 'red', 'borderWidth': '3px'},
                                            )],
                                        ),

                                        html.Br(),

                                    ]
                                ),

                                # Right controls
                                html.Div(
                                    className="two columns div-user-controls bg-grey-copy",
                                    style={'padding': '16px 20px', 'whiteSpace': 'normal', 'wordBreak': 'break-word', 'overflow': 'visible', 'maxWidth': '100%'},
                                    children=[

                                        # Change to side-by-side for mobile layout
                                        html.Div(
                                            className="row",
                                            children=[

                                                html.Br(),html.Br(),

                                                html.H3("Distribution:"),

                                                html.Div(
                                                    className="div-for-dropdown",
                                                    children=[

                                                        dcc.RadioItems(
                                                            id='dist-marginal',
                                                            options=[{'label': x, 'value': x.lower()} 
                                                                        for x in ['Box', 'Violin', 'Rug']],
                                                            value='box',
                                                            labelStyle={'display': 'inline-block', 'marginRight': 8},
                                                            style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px'}
                                                        ), 
                                                    ]
                                                ),                                                      

                                                html.Br(),

                                                html.H3("Grouping:"),                                                

                                                html.Div(
                                                    className="div-for-dropdown",
                                                    children=[
                                                    # Dropdown for clinical T-Stage
                                                    dcc.Dropdown(
                                                        id="group-histo",
                                                        options=[],
                                                        value=None,
                                                        style=  {'borderColor': 'red', 'borderWidth': '3px'},
                                                    )],
                                                ),                                                      

                                                html.Br(),                                                

                                            ],
                                        ),                                                    
                                    ],
                                ),                                

                            ]),

                            dcc.Tab(
                                id="tab-6",
                                value="tab-6",
                                label='Box Plots', 
                                className='custom-tab', 
                                selected_className='custom-tab--selected',
                                style={'padding': '12px 18px'},
                                selected_style={'padding': '12px 18px'},
                                children=[

                                # Column for left controls
                                html.Div(
                                    className="three columns div-user-controls bg-grey-copy",
                                    children=[

                                        # Change to side-by-side for mobile layout
                                        html.Div(
                                            className="row",
                                            children=[                        

                                                html.Br(),html.Br(),

                                                html.H3("Y-axis:"),

                                                html.Div(
                                                    className="div-for-dropdown",
                                                    children=[
                                                        # Dropdown for clinical T-Stage
                                                        dcc.Dropdown(
                                                            id="y-box",
                                                            options=[],
                                                            value=None,
                                                            style=  {'borderColor': 'red', 'borderWidth': '3px'},
                                                        )
                                                    ],
                                                ),
                                            ]
                                        )
                                    ]
                                ),

                                # Middle Graph
                                html.Div(
                                    className="six columns div-for-charts bg-grey",
                                    children=[        
                                        html.Br(),

                                        dcc.Graph(id="box-plot"),
                                        dcc.Tooltip(
                                            id="box-tooltip", 
                                            direction='right', 
                                            background_color='white', 
                                            border_color='red', 
                                            style={
                                                'position': 'absolute',    
                                                'zIndex': 10000,
                                                'pointerEvents': 'none'
                                            }),

                                        html.H3("X-axis:"),


                                        html.Div(
                                            className="div-for-dropdown",
                                            children=[
                                            # Dropdown for clinical T-Stage
                                            dcc.Dropdown(
                                                id="x-box",
                                                options=[],
                                                value=None,
                                                style=  {'borderColor': 'red', 'borderWidth': '3px'},
                                            )],
                                        ),

                                        html.Br(),

                                    ]
                                ),

                            ]),

                            dcc.Tab(
                                id="tab-7",
                                value="tab-7",
                                label='Survival Plots', 
                                className='custom-tab', 
                                selected_className='custom-tab--selected',
                                style={'padding': '12px 18px'},
                                selected_style={'padding': '12px 18px'},
                                children=[

                                # Column for left controls
                                html.Div(
                                    className="three columns div-user-controls bg-grey-copy",
                                    children=[

                                        # Change to side-by-side for mobile layout
                                        html.Div(
                                            className="row",
                                            children=[                        

                                                html.Br(),html.Br(),

                                                html.H3("Y-axis:"),

                                                html.Div(
                                                    className="div-for-dropdown",
                                                    children=[
                                                        # Dropdown for clinical T-Stage
                                                        dcc.Dropdown(
                                                            id="y-surv",
                                                            options=[],
                                                            value=None,
                                                            style=  {'borderColor': 'red', 'borderWidth': '3px'},
                                                        )
                                                    ],
                                                ),
                                            ]
                                        )
                                    ]
                                ),

                                # Middle Graph
                                html.Div(
                                    className="six columns div-for-charts bg-grey",
                                    children=[        
                                        html.Br(),

                                        dcc.Graph(id="surv-plot"),

                                        html.Br(),

                                        html.H4("Time [years]"),

                                        html.Br(),

                                    ]
                                ),

                                # Right controls
                                html.Div(
                                    className="three columns div-user-controls bg-grey-copy",
                                    children=[

                                        # Change to side-by-side for mobile layout
                                        html.Div(
                                            className="row",
                                            children=[

                                                html.Br(),html.Br(),                                                    


                                                html.H3("Grouping:"),                                                

                                                html.Div(
                                                    className="div-for-dropdown",
                                                    children=[
                                                    dcc.Dropdown(
                                                        id="group-surv",
                                                        options=[],
                                                        value=None,
                                                        style=  {'borderColor': 'red', 'borderWidth': '3px'},
                                                    )],
                                                ),

                                                html.H3("Median # of Days:"),

                                                html.Div(id='days-output'),   

                                                html.Br(),html.Br(),            

                                                html.H3("Total # of patients:"),

                                                html.Div(id='pts-output'),      

                                                html.Br(),html.Br(),            

                                                html.H3("# of Event Patients:"),

                                                html.Div(id='events-output'),                                                                       

                                                html.Br(),                                                

                                            ],
                                        ),                                                    
                                    ],
                                ), 
                            ]), 

                            dcc.Tab(
                                id="tab-8",
                                value="tab-8",
                                label='Map', 
                                className='custom-tab', 
                                selected_className='custom-tab--selected',
                                style={'padding': '12px 18px'},
                                selected_style={'padding': '12px 18px'},
                                children=[


                                # Middle Graph
                                html.Div(
                                    className="twelve columns div-for-charts bg-grey",
                                    children=[        
                                        html.Br(),

                                        html.H4("Sarcoma Distribution"),

                                        dcc.Graph(id="map-plot"),
                                        dcc.Tooltip(
                                            id="map-tooltip", 
                                            direction='right', 
                                            background_color='white', 
                                            border_color='red', 
                                            style={
                                                'position': 'absolute',    
                                                'zIndex': 10000,
                                                'pointerEvents': 'none'
                                            }),

                                    ]
                                ),

                            ]), 

                            # dcc.Tab(
                            #     id="tab-9",
                            #     value="tab-9",
                            #     label='Swimmers', 
                            #     className='custom-tab', 
                            #     selected_className='custom-tab--selected',
                            #     style={'padding': '12px 18px'},
                            #     selected_style={'padding': '12px 18px'},
                            #     children=[
                            #
                            #     # Column for left controls
                            #     html.Div(
                            #         className="three columns div-user-controls bg-grey-copy",
                            #         children=[
                            #
                            #             # Change to side-by-side for mobile layout
                            #             html.Div(
                            #                 className="row",
                            #                 children=[
                            #
                            #                     html.Br(),html.Br(), 
                            #
                            #                     # Change to side-by-side for mobile layout
                            #                     html.Div(
                            #                         className="row",
                            #                         children=[
                            #
                            #                             html.Img(
                            #                                 className="legend", 
                            #                                 src="assets/swimLegend.png",
                            #                                 style={'height':'100%', 'width':'100%'}
                            #                             ),                                
                            #                         ],
                            #                     ),                                                         
                            #
                            #                     html.Br(),                                                
                            #
                            #                 ],
                            #             ),                                                    
                            #         ],
                            #     ),
                            #
                            #     # Middle Graph
                            #     html.Div(
                            #         className="six columns div-for-charts bg-grey",
                            #         children=[        
                            #             html.Br(),
                            #
                            #             html.H4("Wound Complications"),                                        
                            #
                            #             dcc.Graph(id="swimmer-plot"),
                            #
                            #             html.H4("Time [years]"),
                            #
                            #             html.Br(),
                            #
                            #         ]
                            #     ),
                            #
                            #     # Right controls
                            #     html.Div(
                            #         className="three columns div-user-controls bg-grey-copy",
                            #         children=[
                            #
                            #             # Change to side-by-side for mobile layout
                            #             html.Div(
                            #                 className="row",
                            #                 children=[
                            #
                            #                     html.Br(),html.Br(),
                            #
                            #                     html.H3("# of Wound Complication Patients:"),
                            #
                            #                     html.Div(id='wound-output'),   
                            #
                            #                     html.Br(),html.Br(),  
                            #
                            #                     # html.H3("Grouping:"),                                                
                            #
                            #                     # html.Div(
                            #                     #     className="div-for-dropdown",
                            #                     #     children=[
                            #                     #     # Dropdown for clinical T-Stage
                            #                     #     dcc.Dropdown(
                            #                     #         id="group-box",
                            #                     #         options=[
                            #                     #             
                            #                     #                 {"label": col, "value": col}
                            #                     #                 for col in sorted(df.select_dtypes(include='object')) 
                            #                     #         ],
                            #                     #         value="Histology Category",
                            #                     #         style=  {'borderColor': 'red', 'borderWidth': '3px'},
                            #                     #     )],
                            #                     # ),                                                     
                            #
                            #                     html.Br(),                                                
                            #
                            #                 ],
                            #             ),                                                    
                            #         ],
                            #     ), 
                            # ]),  
                            dcc.Tab(
                                id="tab-10",
                                value="tab-10",
                                label='Imaging', 
                                className='custom-tab', 
                                selected_className='custom-tab--selected',
                                style={'padding': '12px 18px'},
                                selected_style={'padding': '12px 18px'},
                                children=[

                                # Plane selection at the top (horizontal)
                                html.Div(
                                    className="row",
                                    children=[
                                        html.Div(
                                            className="twelve columns div-user-controls bg-grey-copy",
                                            children=[
                                                html.Div(
                                                    className="row",
                                                    children=[
                                                        html.Br(),
                                                        html.H3("Plane:", style={'textAlign': 'center', 'marginBottom': '10px'}),
                                                        html.Div(
                                                            className="div-for-dropdown",
                                                            style={'padding': '10px 20px', 'textAlign': 'center'},
                                                            children=[
                                                                dcc.RadioItems(
                                                                    id='imaging-plane',
                                                                    options=[{'label': x, 'value': x} 
                                                                                for x in ['Axial', 'Coronal', 'Sagittal']],
                                                                    value='Coronal',
                                                                    labelStyle={'display': 'inline-block', 'margin': '0 20px'},
                                                                    style={'fontSize': '16px'}
                                                                ), 
                                                            ]
                                                        ),
                                                        html.Br(),
                                                    ],
                                                ),                                                    
                                            ],
                                        ),
                                    ],
                                ),    

                                # Full-width imaging plot
                                html.Div(
                                    className="twelve columns div-for-charts bg-grey",
                                    children=[        
                                        html.Br(),
                                        html.H4("RT Imaging"),                                       
                                        dcc.Graph(id="imaging-plot"),
                                        html.Br(),
                                    ]
                                ),

                            ]),
                            dcc.Tab(
                                id="tab-11",
                                value="tab-11",
                                label='3D Visualization', 
                                className='custom-tab', 
                                selected_className='custom-tab--selected',
                                style={'padding': '12px 18px'},
                                selected_style={'padding': '12px 18px'},
                                children=[
                                    # Left controls
                                    html.Div(
                                        className="three columns div-user-controls bg-grey-copy",
                                        children=[
                                            html.Div(
                                                className="row",
                                                children=[
                                                    html.Br(),
                                                    html.H3("X-axis:", style={'textAlign': 'center'}),
                                                    html.Div(
                                                        className="div-for-dropdown",
                                                        style={'padding': '10px 20px'},
                                                        children=[
                                                            dcc.Dropdown(
                                                                id='x-3d-dropdown',
                                                                options=[],
                                                                value=None,
                                                                style={'borderColor': 'red', 'borderWidth': '3px'},
                                                                multi=False
                                                            ),
                                                        ]
                                                    ),
                                                    html.Br(),
                                                    html.H3("Y-axis:", style={'textAlign': 'center'}),
                                                    html.Div(
                                                        className="div-for-dropdown",
                                                        style={'padding': '10px 20px'},
                                                        children=[
                                                            dcc.Dropdown(
                                                                id='y-3d-dropdown',
                                                                options=[],
                                                                value=None,
                                                                style={'borderColor': 'red', 'borderWidth': '3px'},
                                                                multi=False
                                                            ),
                                                        ]
                                                    ),
                                                    html.Br(),
                                                    html.H3("Z-axis:", style={'textAlign': 'center'}),
                                                    html.Div(
                                                        className="div-for-dropdown",
                                                        style={'padding': '10px 20px'},
                                                        children=[
                                                            dcc.Dropdown(
                                                                id='z-3d-dropdown',
                                                                options=[],
                                                                value=None,
                                                                style={'borderColor': 'red', 'borderWidth': '3px'},
                                                                multi=False
                                                            ),
                                                        ]
                                                    ),
                                                    html.Br(),
                                                    html.H3("Color:", style={'textAlign': 'center'}),
                                                    html.Div(
                                                        className="div-for-dropdown",
                                                        style={'padding': '10px 20px'},
                                                        children=[
                                                            dcc.Dropdown(
                                                                id='color-3d-dropdown',
                                                                options=[],
                                                                value=None,
                                                                style={'borderColor': 'red', 'borderWidth': '3px'},
                                                                multi=False
                                                            ),
                                                        ]
                                                    ),
                                                    html.Br(),
                                                    html.H3("Size:", style={'textAlign': 'center'}),
                                                    html.Div(
                                                        className="div-for-dropdown",
                                                        style={'padding': '10px 20px'},
                                                        children=[
                                                            dcc.Dropdown(
                                                                id='size-3d-dropdown',
                                                                options=[],
                                                                value=None,
                                                                style={'borderColor': 'red', 'borderWidth': '3px'},
                                                                multi=False
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    # Middle Graph
                                    html.Div(
                                        className="nine columns div-for-charts bg-grey",
                                        children=[        
                                            html.Br(),
                                            html.H4("3D Patient Data Visualization"),
                                            dcc.Graph(
                                                id="3d-plot",
                                                clear_on_unhover=True,
                                                style={'width': '100%', 'height': '90vh'},
                                                config={'displayModeBar': True, 'scrollZoom': True}
                                            ),
                                            dcc.Tooltip(id="3d-tooltip", direction='right', background_color='white', border_color='red'),
                                            html.Br(),
                                        ]
                                    ),
                                ]
                            ),
                        ]),
                    ]                                                                   
                )
            ]
        )
    ]
)  

# File upload callback
@app.callback(
    [Output('uploaded-data', 'data'),
     Output('upload-status', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def parse_contents(contents, filename):
    if contents is None:
        return None, html.Div([
            html.H4("Please upload a CSV file to begin"),
            html.P("Drag and drop a CSV file above or click to select one.")
        ])
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Basic data cleaning
        # Convert date columns to datetime format
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col]).dt.strftime("%Y/%m/%d")
                except:
                    pass  # Skip if conversion fails
        
        # Add an index column if none exists
        if df.index.name is None:
            df.insert(loc=0, column='Row_ID', value=range(1, len(df) + 1))
            df.set_index('Row_ID', inplace=True)
        
        return df.to_dict('records'), html.Div([
            html.H4(f"Successfully uploaded: {filename}"),
            html.P(f"Data contains {len(df)} rows and {len(df.columns)} columns"),
            html.P("✓ Automatic field type detection completed"),
            html.P("✓ Default visualization values have been set"),
            html.H5("Next: Review field types below and start exploring your data!")
        ])
    except Exception as e:
        return None, html.Div([
            html.H4("Error processing file"),
            html.P(f"Could not process {filename}. Error: {str(e)}")
        ])

# Field type selection callback
@app.callback(
    Output('field-types', 'data'),
    [Input('uploaded-data', 'data')]
)
def determine_field_types(data):
    if data is None:
        return None
    
    df = pd.DataFrame(data)
    
    # Automatically determine field types
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        # Try to convert to numeric with improved logic
        try:
            # Use the improved convert_to_numeric function
            numeric_data = convert_to_numeric(df, col)
            # Check if we have enough valid numeric values (at least 50% or minimum 3 values)
            valid_count = numeric_data.notna().sum()
            total_count = len(numeric_data)
            
            if valid_count >= max(3, total_count * 0.5):
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        except:
            categorical_cols.append(col)
    
    # Ensure we have at least some columns of each type for plotting
    if not numerical_cols and categorical_cols:
        # If no numerical columns found, try to make the first categorical column numerical
        if categorical_cols:
            first_cat = categorical_cols.pop(0)
            numerical_cols.append(first_cat)
    
    return {
        'numerical': numerical_cols,
        'categorical': categorical_cols,
        'all_columns': list(df.columns)
    }

# Field selection UI callback
@app.callback(
    Output('field-selection-ui', 'children'),
    [Input('field-types', 'data'), Input('uploaded-data', 'data')]
)
def show_field_selection(field_types, uploaded_data):
    if field_types is None or uploaded_data is None:
        return html.Div()
    
    df = pd.DataFrame(uploaded_data)
    columns = list(df.columns)
    sample_values = [', '.join(map(str, df[col].dropna().astype(str).head(3))) for col in columns]
    
    # Check if we have saved field types that override the automatic detection
    if 'numerical' in field_types and 'categorical' in field_types:
        # Use saved field types
        type_map = {}
        for col in columns:
            if col in field_types.get('numerical', []):
                type_map[col] = 'Numerical'
            elif col in field_types.get('categorical', []):
                type_map[col] = 'Categorical'
            elif col in field_types.get('date', []):
                type_map[col] = 'Date'
            elif col in field_types.get('ignored', []):
                type_map[col] = 'Ignore'
            else:
                # Fallback to automatic detection for any missing columns
                col_lower = col.lower()
                date_patterns = ['date', 'enroll', 'consult', 'follow up', 'closure', 'death', 'recurrence', 'amputation', 'surgery', 'dehiscence']
                non_date_patterns = ['age', 'days', 'time to', 'duration', 'size', 'grade', 'stage', 'score', 'total', 'count', 'number']
                
                is_date = any(pattern in col_lower for pattern in date_patterns)
                is_not_date = any(pattern in col_lower for pattern in non_date_patterns)
                
                if is_date and not is_not_date:
                    type_map[col] = 'Date'
                elif col in field_types.get('numerical', []):
                    type_map[col] = 'Numerical'
                else:
                    type_map[col] = 'Categorical'
    else:
        # Use automatic detection (initial load)
        type_map = {}
        for col in columns:
            col_lower = col.lower()
            date_patterns = ['date', 'enroll', 'consult', 'follow up', 'closure', 'death', 'recurrence', 'amputation', 'surgery', 'dehiscence']
            non_date_patterns = ['age', 'days', 'time to', 'duration', 'size', 'grade', 'stage', 'score', 'total', 'count', 'number']
            
            is_date = any(pattern in col_lower for pattern in date_patterns)
            is_not_date = any(pattern in col_lower for pattern in non_date_patterns)
            
            if is_date and not is_not_date:
                type_map[col] = 'Date'
            elif col in field_types.get('numerical', []):
                type_map[col] = 'Numerical'
            else:
                type_map[col] = 'Categorical'
    
    # Build DataTable rows with Type as second column
    data = [
        {'Field Name': col, 'Type': type_map[col], 'Sample Values': sample_values[i]}
        for i, col in enumerate(columns)
    ]
    
    # Dropdown options for type
    dropdown = {
        'Type': {
            'options': [
                {'label': 'Numerical', 'value': 'Numerical'},
                {'label': 'Categorical', 'value': 'Categorical'},
                {'label': 'Date', 'value': 'Date'},
                {'label': 'Ignore', 'value': 'Ignore'},
            ]
        }
    }
    
    return html.Div([
        html.H4("Classify Each Field:"),
        html.P("Select the type for each field below. You can change the pre-selected types."),
        dash_table.DataTable(
            id='field-type-table',
            columns=[
                {'name': 'Field Name', 'id': 'Field Name', 'editable': False, 'selectable': True},
                {'name': 'Type', 'id': 'Type', 'presentation': 'dropdown'},
                {'name': 'Sample Values', 'id': 'Sample Values', 'editable': False},
            ],
            data=data,
            editable=True,
            dropdown=dropdown,
            style_cell={
                'textAlign': 'left', 
                'fontSize': 14,
                'padding': '8px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'maxWidth': 0,
            },
            style_cell_conditional=[
                {
                    'if': {'column_id': 'Field Name'},
                    'width': '25%',
                    'minWidth': '150px',
                    'maxWidth': '200px',
                },
                {
                    'if': {'column_id': 'Type'},
                    'width': '15%',
                    'minWidth': '100px',
                    'maxWidth': '120px',
                },
                {
                    'if': {'column_id': 'Sample Values'},
                    'width': '60%',
                    'minWidth': '200px',
                },
            ],
            style_header={
                'fontWeight': 'bold', 
                'backgroundColor': uclaBlue, 
                'color': 'white',
                'fontSize': 14,
                'padding': '8px',
            },
            style_data={
                'backgroundColor': 'white', 
                'color': 'black',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_table={
                'width': '100%', 
                'maxWidth': '100%',
                'overflowX': 'auto',
            },
            tooltip_data=[
                {
                    'Field Name': {'value': col, 'type': 'markdown'},
                    'Sample Values': {'value': sample_values[i], 'type': 'markdown'},
                } for i, col in enumerate(columns)
            ],
            tooltip_duration=None,
        ),
        html.Br(),
        html.Div([
            html.Button('Save Field Types', id='save-field-types', n_clicks=0, 
                       style={
                           'fontSize': '16px', 
                           'padding': '12px 32px', 
                           'backgroundColor': uclaBlue, 
                           'color': 'white', 
                           'border': 'none', 
                           'borderRadius': '8px',
                           'cursor': 'pointer',
                           'fontWeight': 'bold',
                           'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                           'transition': 'all 0.3s ease',
                           'textAlign': 'center',
                           'display': 'flex',
                           'alignItems': 'center',
                           'justifyContent': 'center'
                       }),
        ], style={'textAlign': 'center', 'margin': '20px 0'}),
        html.Div(id='field-save-timestamp'),
    ])

# Data status display callback
@app.callback(
    Output('data-status-display', 'children'),
    [Input('uploaded-data', 'data'),
     Input('field-types', 'data')]
)
def update_data_status(data, field_types):
    if data is None:
        return html.Div([
            html.H5("No data uploaded"),
            html.P("Please upload a CSV file to begin")
        ])
    
    if field_types is None:
        return html.Div([
            html.H5("Data uploaded"),
            html.P("Processing field types...")
        ])
    
    df = pd.DataFrame(data)
    return html.Div([
        html.H5("Data Ready"),
        html.P(f"Rows: {len(df)}"),
        html.P(f"Columns: {len(df.columns)}"),
        html.P(f"Numerical: {len(field_types['numerical'])}"),
        html.P(f"Categorical: {len(field_types['categorical'])}"),
    ])

@app.callback(
    Output('data-status-display', 'style'),
    [Input('table-filters', 'derived_virtual_indices'),
     Input('uploaded-data', 'data')]
)
def update_data(dataInds, uploaded_data):    
    return {'display': 'block'}

### filter callbacks ###

# highlight a selected column
@app.callback(
    Output('table-filters', 'style_data_conditional'),
    Input('table-filters', 'selected_columns')
)
def update_styles(selected_columns):
    return [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]



# define general figure layout
layout = go.Layout(
    xaxis=dict(
    showgrid=True,
    tickfont=dict(size=15,color='black'),
    zerolinewidth=5,
    zerolinecolor='black', 
    zeroline=True,
    gridcolor= 'rgba(0.5,0.5,0.5,0.2)',
    linecolor='black',
    linewidth=2,
    mirror=True,
),
    yaxis=dict(
    showgrid=True,
    zeroline=True,
    zerolinecolor='black', 
    tickfont=dict(size=15,color='black'),
    zerolinewidth=5,
    showticklabels=True,
    gridcolor= 'rgba(0.5,0.5,0.5,0.2)',
    linecolor='black',
    linewidth=2,
    mirror=True,
),
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=600,
    margin=go.layout.Margin(l=50, r=50, t=50, b=50),
    # Add border around the entire plot area
    shapes=[
        dict(
            type="rect",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(
                color="rgba(39, 116, 174, 0.8)",
                width=3,
            ),
            fillcolor="rgba(0,0,0,0)",
        )
    ]
)

# max marker size
maxSize = 40

# Helper function to convert data to numeric, handling Wound Toxicity Acute specially
def convert_to_numeric(data, column):
    if column == 'Wound Toxicity Acute':
        out = data[column].str.extract(r'(\d+)$').astype(float)
        return out[0]
    
    # Try to convert to numeric, handling various edge cases
    try:
        # First try direct conversion
        result = pd.to_numeric(data[column], errors='coerce')
        
        # If all values are NaN but the column has data, try string extraction
        if result.isna().all() and not data[column].isna().all():
            # Try to extract numbers from strings
            string_data = data[column].astype(str)
            # Extract first number found in each cell
            extracted = string_data.str.extract(r'([-+]?\d*\.?\d+)').astype(float)
            if not extracted[0].isna().all():
                result = extracted[0]
        
        return result
    except:
        # If all else fails, return a series of NaN values with the same index
        return pd.Series([np.nan] * len(data), index=data.index)

# Scatter plot callback
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-dropdown', 'value'),
    Input('y-dropdown', 'value'),
    Input('color-dropdown', 'value'),
    Input('size-dropdown', 'value'),
    Input('table-filters', 'derived_virtual_indices'),
    Input('uploaded-data', 'data')]
    )
def compute(xAxis, yAxis, colorAxis, sizeAxis, dataInds, uploaded_data):                  
    if uploaded_data is None:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="Please upload a CSV file to view plots",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig
    
    df = pd.DataFrame(uploaded_data)
    
    if xAxis is None or yAxis is None or colorAxis is None or sizeAxis is None:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="Please select valid X, Y, Color, and Size axis values",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig

    if dataInds is None:
        data = df.copy()
    else:
        data = df.iloc[dataInds, :]

    if any(col not in data.columns for col in [xAxis, yAxis, colorAxis, sizeAxis]):
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="One or more selected columns do not exist in the data",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig

    # Convert X and Y to numeric (required)
    x_data = convert_to_numeric(data, xAxis)
    y_data = convert_to_numeric(data, yAxis)
    
    # Handle color axis (categorical and numeric)
    if colorAxis in data.columns:
        color_values = data[colorAxis]
        # Convert categorical to numeric for plotting
        if color_values.dtype == 'object' or pd.api.types.is_categorical_dtype(color_values):
            unique_colors = color_values.dropna().unique()
            color_map = {val: i for i, val in enumerate(unique_colors)}
            color_data = color_values.map(color_map).astype(float)
            # Scale colors to use full colorscale range for better distinction
            if len(unique_colors) > 1:
                color_data = color_data * 10  # Spread out color values more
        else:
            color_data = convert_to_numeric(data, colorAxis)
    else:
        color_data = pd.Series([0] * len(data), index=data.index)
    
    # Handle size axis (categorical and numeric)
    if sizeAxis in data.columns:
        size_values = data[sizeAxis]
        # Convert categorical to numeric for plotting
        if size_values.dtype == 'object' or pd.api.types.is_categorical_dtype(size_values):
            unique_sizes = size_values.dropna().unique()
            # Map to distinct size values instead of just indices
            size_map = {val: 6 + i * 3 for i, val in enumerate(unique_sizes)}  # Start at 6, increment by 3
            size_data = size_values.map(size_map).astype(float)
        else:
            size_data = convert_to_numeric(data, sizeAxis)
    else:
        size_data = pd.Series([6] * len(data), index=data.index)  # Default size

    # Create mask for valid numeric values - only require X and Y to be valid
    valid_mask = ~(x_data.isna() | y_data.isna())
    
    if valid_mask.empty or not valid_mask.any():
        # Try a more permissive approach - accept if we have any X or Y values
        x_valid = ~x_data.isna()
        y_valid = ~y_data.isna()
        
        if x_valid.any() and y_valid.any():
            # Create a mask that accepts rows where both X and Y are valid
            valid_mask = x_valid & y_valid
            if not valid_mask.any():
                # Fill missing values with mean of available data
                x_data = x_data.fillna(x_data.mean())
                y_data = y_data.fillna(y_data.mean())
                valid_mask = ~(x_data.isna() | y_data.isna())
        
        if not valid_mask.any():
            fig = go.Figure(layout=layout)
            fig.add_annotation(
                text="No valid numeric data points available for X and Y axes",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=20, color=uclaBlue)
            )
            return fig

    # Filter data using the mask
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    color_data = color_data[valid_mask]
    size_data = size_data[valid_mask]
    patient_ids = data.index[valid_mask]

    # Fill any remaining NaN values in color/size data
    color_data = color_data.fillna(0)
    size_data = size_data.fillna(1)

    # Normalize size data to reasonable values
    size_min, size_max = np.nanmin(size_data), np.nanmax(size_data)
    if size_min == size_max or np.all(np.isclose(size_data, size_min)):
        normalized_sizes = np.full_like(size_data, 4.0)  # If all values are the same, use minimum size
    else:
        normalized_sizes = 4 + 4 * (size_data - size_min) / (size_max - size_min)

    xLabels = np.repeat(xAxis, len(x_data))
    yLabels = np.repeat(yAxis, len(y_data))
    colorLabels = np.repeat(colorAxis, len(color_data))
    sizeLabels = np.repeat(sizeAxis, len(size_data))

    scatterFig = go.Figure(layout=layout)
    scatterFig.add_trace(
        go.Scatter(
            x=x_data, y=y_data, mode='markers', marker=dict(
                color=color_data,
                size=normalized_sizes,
                sizemode='area',
                sizeref=2 * max(normalized_sizes)/(maxSize**2),
                sizemin=1,
                colorbar=dict(thickness=40),
                colorscale='Viridis',  # Better color scale for distinction
                opacity=0.7,
                line=dict(
                    width=2,
                    color='rgba(39, 116, 174, 0.8)'
                )
            ),
            customdata=np.stack((xLabels, yLabels, colorLabels, color_data, sizeLabels, size_data, patient_ids), axis=-1),
            hoverinfo='none'
        )
    )

    return scatterFig

# hover callback
@app.callback(
    Output("scatter-tooltip", "show"),
    Output("scatter-tooltip", "bbox"),
    Output("scatter-tooltip", "children"),
    [Input("scatter-plot", "hoverData"),
     Input('x-dropdown', 'value'),
     Input('y-dropdown', 'value'),
     Input('color-dropdown', 'value'),
     Input('size-dropdown', 'value'),]   
)
def display_hover(hoverData, x_axis, y_axis, color_axis, size_axis):
    if hoverData is None:
        return False, no_update, no_update
 
    # Get the hovered point data
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    pt_id = hover_data["customdata"][6]  # Patient ID is stored in customdata[6]
    x_val = hover_data["x"]
    y_val = hover_data["y"]
    color_val = hover_data["customdata"][3]  # Color value is stored in customdata[3]
    size_val = hover_data["customdata"][5]  # Size value is stored in customdata[5]
    
    # Format the values
    if isinstance(color_val, (int, float)):
        color_val = f"{color_val:.2f}"
    if isinstance(size_val, (int, float)):
        size_val = f"{size_val:.2f}"

    try:
        # Create paths for each image
        basePath = 'assets/Captures_Nums_small/'
        axial_path = basePath + str(pt_id) + '/Axial.tiff'
        coronal_path = basePath + str(pt_id) + '/Coronal.tiff'
        sagittal_path = basePath + str(pt_id) + '/Sagittal.tiff'

        # Load and process images using PIL
        def load_and_resize_image(path):
            try:
                photo = Image.open(path).convert("RGBA")
                photo = photo.resize((50, 50))  # Using the same size as 3D tooltip
                # Convert to base64
                buffered = io.BytesIO()
                photo.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()
            except Exception as e:
                print(f"Error loading image {path}: {str(e)}")
                return None

        axial_img = load_and_resize_image(axial_path)
        coronal_img = load_and_resize_image(coronal_path)
        sagittal_img = load_and_resize_image(sagittal_path)

        # Create a div with the patient info and images
        children = [
            html.Div([
                html.Div([
                    html.H4(f"Patient {pt_id}", style={'textAlign': 'center', 'color': 'black', 'marginBottom': '5px', 'fontSize': '14px'}),
                    html.Div([
                        html.P([
                            html.Strong(f"{x_axis}: "),
                            f"{x_val:.2f}"
                        ], style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                        html.P([
                            html.Strong(f"{y_axis}: "),
                            f"{y_val:.2f}"
                        ], style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                        html.P([
                            html.Strong(f"{color_axis}: "),
                            f"{color_val}"
                        ], style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                        html.P([
                            html.Strong(f"{size_axis}: "),
                            f"{size_val}"
                        ], style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                    ], style={'marginBottom': '5px'}),
                ], style={'marginBottom': '5px'}),
                
                # Images in a row
                html.Div([
                    html.Div([
                        html.Img(
                            src=f'data:image/png;base64,{axial_img}' if axial_img else 'assets/imagePlaceholder.png',
                            style={
                                "width": "50px",
                                'display': 'block',
                                'margin': '2px auto',
                                'border': '1px solid black'
                            }
                        ),
                        html.P("Axial", style={'textAlign': 'center', 'color': 'black', 'margin': '1px', 'fontSize': '10px'}),
                    ], style={'display': 'inline-block', 'margin': '0 2px'}),
                    html.Div([
                        html.Img(
                            src=f'data:image/png;base64,{coronal_img}' if coronal_img else 'assets/imagePlaceholder.png',
                            style={
                                "width": "50px",
                                'display': 'block',
                                'margin': '2px auto',
                                'border': '1px solid black'
                            }
                        ),
                        html.P("Coronal", style={'textAlign': 'center', 'color': 'black', 'margin': '1px', 'fontSize': '10px'}),
                    ], style={'display': 'inline-block', 'margin': '0 2px'}),
                    html.Div([
                        html.Img(
                            src=f'data:image/png;base64,{sagittal_img}' if sagittal_img else 'assets/imagePlaceholder.png',
                            style={
                                "width": "50px",
                                'display': 'block',
                                'margin': '2px auto',
                                'border': '1px solid black'
                            }
                        ),
                        html.P("Sagittal", style={'textAlign': 'center', 'color': 'black', 'margin': '1px', 'fontSize': '10px'}),
                    ], style={'display': 'inline-block', 'margin': '0 2px'}),
                ], style={'textAlign': 'center'}),
            ], style={
                'backgroundColor': 'white',
                'padding': '5px',
                'border': '1px solid red',
                'borderRadius': '3px',
                'boxShadow': '0 0 5px rgba(0,0,0,0.2)',
                'zIndex': '10000',
                'position': 'absolute'
            })
        ]
        return True, bbox, children
    except Exception as e:
        print(f"Error in tooltip: {str(e)}")
        children = [
            html.Div([
                html.H4(f"Patient {pt_id}", style={'textAlign': 'center', 'color': 'black', 'fontSize': '14px'}),
                html.P(f"{x_axis}: {x_val:.2f}", style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                html.P(f"{y_axis}: {y_val:.2f}", style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                html.P(f"{color_axis}: {color_val}", style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                html.P(f"{size_axis}: {size_val}", style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                html.P(f"Error loading images: {str(e)}", style={'color': 'red', 'fontSize': '10px'})
            ], style={
                'backgroundColor': 'white',
                'padding': '5px',
                'border': '1px solid red',
                'borderRadius': '3px',
                'zIndex': '1000',
                'position': 'relative'
            })
        ]
        return True, bbox, children

# Histogram callback
@app.callback(
    Output("histo-plot", "figure"), 
    [Input("dist-marginal", "value"),
    Input('x-histo', 'value'),
    Input('group-histo', 'value'),
    Input('table-filters', 'derived_virtual_indices'),
    Input('uploaded-data', 'data')]
    )
def display_histo(marginal, xAxis, groupAxis, dataInds, uploaded_data):
    if uploaded_data is None:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="Please upload a CSV file to view plots",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig
    
    df = pd.DataFrame(uploaded_data)
    
    if xAxis is None or groupAxis is None:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="Please select a valid grouping and X axis values",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig

    if dataInds is None:
        data = df.copy()
    else:
        data = df.iloc[dataInds, :]

    if xAxis not in data.columns or groupAxis not in data.columns:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="One or more selected columns do not exist in the data",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig
    
    # Convert x-axis data to numeric
    x_data = convert_to_numeric(data, xAxis)
    group_data = data[groupAxis]  # Keep group data as is since it's categorical
    
    # Create mask for valid numeric values in x_data
    valid_mask = ~x_data.isna()
    
    if not valid_mask.any():
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="No valid numeric data points available for the selected X variable",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig

    # Filter data using the mask
    x_data = x_data[valid_mask]
    group_data = group_data[valid_mask]
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        xAxis: x_data,
        groupAxis: group_data
    })

    # construct dict of sorted orders
    cats = plot_data[groupAxis].unique()
    cats = [str(i) for i in cats]
    cats.sort()
    cats = {
        groupAxis: cats
    }

    fig = px.histogram(
        plot_data, x=xAxis, color=groupAxis,
        barmode='group',
        marginal=marginal,
        opacity=0.7,
        category_orders=cats,
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=800,
        margin=go.layout.Margin(l=50, r=50, t=50, b=50),
        yaxis=dict(
            showgrid=True,
            tickfont=dict(size=15,color='black'),
            showticklabels=True,
            gridcolor= 'rgba(0.5,0.5,0.5,0.2)',
            title='',
            linecolor='black',
            linewidth=2,
            mirror=True,
        ),
        xaxis=dict(
            showgrid=True,
            tickfont=dict(size=15,color='black'),
            gridcolor= 'rgba(0.5,0.5,0.5,0.2)',
            categoryorder="category ascending",
            linecolor='black',
            linewidth=2,
            mirror=True,
        ),
        legend_font=dict(
            size=15,
        ),
        legend=dict(
            borderwidth=3,
            bordercolor=uclaBlue,
            bgcolor='rgba(255,255,255,0.9)',
        ),
        font=dict(
            size=20,
            color=uclaBlue,
        ),
        # Add border around the entire plot area
        shapes=[
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(
                    color="rgba(39, 116, 174, 0.8)",
                    width=3,
                ),
                fillcolor="rgba(0,0,0,0)",
            )
        ]
    )

    return fig

# Box plot callback
@app.callback(
    Output("box-plot", "figure"), 
    [Input('x-box', 'value'),
    Input('y-box', 'value'),
    Input('table-filters', 'derived_virtual_indices'),
    Input('uploaded-data', 'data')]
    )
def display_box(xAxis, yAxis, dataInds, uploaded_data):
    if uploaded_data is None:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="Please upload a CSV file to view plots",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig
    
    df = pd.DataFrame(uploaded_data)
    
    if xAxis is None or yAxis is None:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="Please select valid X and Y axis values",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig
    
    if dataInds is None:
        data = df.copy()
    else:
        data = df.iloc[dataInds, :]

    if xAxis not in data.columns or yAxis not in data.columns:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="One or more selected columns do not exist in the data",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig

    # Treat X as categorical labels
    x_data = data[xAxis].astype(str)
    # Ensure Y is numeric
    y_data = convert_to_numeric(data, yAxis)
    
    valid_mask = ~y_data.isna()
    
    if not valid_mask.any():
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="No valid numeric data points available for the selected Y variable",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig

    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    patient_ids = data.index[valid_mask]

    xLabels = np.repeat(xAxis, len(x_data))
    yLabels = np.repeat(yAxis, len(y_data))

    fig = go.Figure(layout=layout)

    fig.add_trace(
        go.Box(
            x=x_data,
            y=y_data,
            boxpoints='all',
            jitter=0.5,
            pointpos=-1.8,
            marker=dict(
                outliercolor='rgba(250,0,0,0.7)',
                line=dict(
                    outliercolor='rgba(255,255,255,0)',
                    width=2
                ),
            ),
            boxmean=True,
            fillcolor='rgba(39, 116, 174, 0.3)',
            line=dict(
                color='rgba(39, 116, 174, 1)',
                width=2
            ),
            customdata=np.stack((xLabels, yLabels, patient_ids), axis=-1),
            hoverinfo='none'
        )
    )

    fig.update_layout(
        xaxis=dict(
            categoryorder="category ascending"
        ),
        legend_font=dict(
            size=15,
        ),
        legend=dict(
            borderwidth=3,
            bordercolor=uclaBlue,
            bgcolor='rgba(255,255,255,0.9)',
        ),
        font=dict(
            size=20,
            color=uclaBlue,
        ),
    )

    return fig

# survival KM plots
@app.callback(
    [Output("surv-plot", "figure"), 
    Output("days-output", component_property='children'),
    Output("pts-output", component_property='children'),
    Output("events-output", component_property='children'),], 
    [Input('y-surv', 'value'),
    Input('group-surv', 'value'),
    Input('table-filters', 'derived_virtual_indices'),
    Input('uploaded-data', 'data')]
    )
def display_surv(yAxis, groupAxis, dataInds, uploaded_data):
    if uploaded_data is None:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="Please upload a CSV file to view plots",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig, 'Days: Upload data first', 'Patients: N/A', 'Events: N/A'
    
    df = pd.DataFrame(uploaded_data)
    
    if yAxis is None:
        # Return an empty figure with just a message prompt
        fig = go.Figure(layout=layout)
        
        # Add annotation explaining selection is needed
        fig.add_annotation(
            text="Please select a survival outcome from the dropdown menu",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        
        # Return placeholder values for the information sections
        return fig, 'Days: Select an option', 'Patients: N/A', 'Events: N/A'

    if dataInds is None:
        data = df.copy()
    else:
        data = df.iloc[dataInds, :]

    if yAxis == "Overall Survival":
        dataTimes = data['Last follow up date']
        dataEventDates = data['Date of Death']
        dataEvents = data['Survival']

        # get follow up time in Days  
        times = pd.to_datetime(dataTimes) - pd.to_datetime(data['Surgery Date'])
        times = times.dt.days
        fuDays = times[times>0]

        # find event times in days 
        eventTimes = pd.to_datetime(dataEventDates) - pd.to_datetime(data['Surgery Date'])
        eventTimes = eventTimes.dt.days
        eventDays = eventTimes[eventTimes>0]

        # overwrite follow up time with event times for patients with events
        fuDays[eventTimes>0] = eventDays

        # get events 
        d = {
            'yes': 1,
            'not evaluable': 0,
            'no': 0,
            'dead': 1,
            'alive': 0,
        }
        events = dataEvents.str.lower()
        events = events.map(d)
        events = events.fillna(0)
        events = events[times>0]

        # Check if we have valid data for survival analysis
        if len(fuDays) == 0 or len(events) == 0 or len(fuDays) != len(events):
            fig = go.Figure(layout=layout)
            fig.add_annotation(
                text="Insufficient data for survival analysis. Please check your date columns and survival data.",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig, 'Days: No valid data', 'Patients: 0', 'Events: 0'
        
        # Ensure data is numeric and valid
        try:
            fuDays = pd.to_numeric(fuDays, errors='coerce').dropna()
            events = pd.to_numeric(events, errors='coerce').dropna()
            
            # Align indices
            common_indices = fuDays.index.intersection(events.index)
            fuDays = fuDays.loc[common_indices]
            events = events.loc[common_indices]
            
            if len(fuDays) == 0 or len(events) == 0:
                raise ValueError("No valid survival data after cleaning")
                
        except Exception as e:
            fig = go.Figure(layout=layout)
            fig.add_annotation(
                text=f"Error processing survival data: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig, 'Days: Data error', 'Patients: 0', 'Events: 0'

        # fit the KM and plot 
        survFig = go.Figure(layout=layout)

        # If a grouping column is selected, stratify by categories
        if groupAxis and groupAxis in data.columns:
            group_series = data[groupAxis].astype(str)
            common_indices = fuDays.index.intersection(events.index)
            group_series = group_series.loc[common_indices]
            fuDays_aligned = fuDays.loc[common_indices]
            events_aligned = events.loc[common_indices]

            for cat in sorted(group_series.dropna().unique()):
                mask = group_series == cat
                if mask.any():
                    km = KaplanMeierFitter()
                    km.fit(durations=fuDays_aligned[mask], event_observed=events_aligned[mask], label=str(cat))
                    survFig.add_trace(
                        go.Scatter(
                            x=km.survival_function_.index, y=km.survival_function_.values.flatten(), mode='lines+markers', 
                            marker=dict(opacity=0.9, symbol="line-ns-open", size=12),
                            line=dict(shape="vh", width=3),
                            name=str(cat),
                            showlegend=True,
                        ),
                    )
        else:
            km = KaplanMeierFitter()
            km.fit(durations=fuDays, event_observed=events, label="Group 1")
            survFig.add_trace(
                go.Scatter(
                    x=km.survival_function_.index, y=km.survival_function_.values.flatten(), mode='lines+markers', 
                    marker=dict(opacity=0.9, symbol="line-ns-open", size=15, color="red"),
                    line=dict(shape="vh", width=4, color=uclaBlue),
                    showlegend=False,
                ),
            )

            ci_times = np.concatenate((km.cumulative_density_.index,km.cumulative_density_.index[::-1]), axis=0)
            ci_vals = np.concatenate((km.confidence_interval_['Group 1_lower_0.95'],km.confidence_interval_['Group 1_upper_0.95'][::-1]), axis=0)

            survFig.add_trace(
                go.Scatter(
                    x=ci_times,
                    y=ci_vals,
                    fill='toself',
                    fillcolor='rgba(100,0,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                )
            )

        survFig.update_layout(
            yaxis=dict(
                showgrid=True,
                tickfont=dict(size=15,color='black'),
                showticklabels=True,
                gridcolor= 'rgba(0.5,0.5,0.5,0.2)',
                title='',
                range=[-0.1,1.1],
            ),
            xaxis=dict(
                tickvals = [m * 365 for m in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.0, 4.5, 5.0]],
                ticktext = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.0, 4.5, 5.0]
            ),
        )

        eventStrDisp = 'Deaths: {}'.format(len(events[events==1]))

    elif yAxis == "Local Failure":
        data = data.dropna(subset=['Last follow up Primary imaging Date'])

        # if no surgery date, set end of RT to surgery date
        nullSurgInds = data[data['Surgery Date'].isnull()]['Surgery Date'].index
        data.loc[nullSurgInds,'Surgery Date'] = data.loc[nullSurgInds, 'End of RT Date']

        dataTimes = data['Last follow up Primary imaging Date']
        dataEventDates = data['Local Recurrence Date']
        dataEvents = data['Local Recurrence']

        dataDeathDates = data['Date of Death']
        dataDeaths = data['Survival']

        # get follow up time in Days  
        times = pd.to_datetime(dataTimes) - pd.to_datetime(data['Surgery Date'])
        times = times.dt.days
        fuDays = times.copy()
        fuDays = fuDays[times>0]

        # find event times in days 
        eventTimes = pd.to_datetime(dataEventDates) - pd.to_datetime(data['Surgery Date'])
        eventTimes = eventTimes.dt.days
        eventDays = eventTimes[eventTimes>0]

        # find death time in days 
        deathTimes = pd.to_datetime(dataDeathDates) - pd.to_datetime(data['Surgery Date'])
        deathTimes = deathTimes.dt.days
        deathDays = deathTimes[deathTimes>0]

        # overwrite follow up time with event times for patients with events
        fuDays[eventTimes>0] = eventDays

        # get events 
        d = {
            'yes': 1,
            'not evaluable': 0,
            'no': 0,
            'dead': 2,
            'alive': 0,
        }
        events = dataEvents.str.lower()
        events = events.map(d)
        events = events.fillna(0)
        events = events[times>0]

        # code the deaths
        deaths = dataDeaths.str.lower()
        deaths = deaths.map(d)
        deaths = deaths.fillna(0)
        deaths = deaths[deathTimes>0]

        # code event as death if occurred prior to local recurrence
        for ptInd in deaths.index:
            if events.index.isin([ptInd]).any() and events.loc[ptInd] == 0:
                events[ptInd] = 2
                fuDays[ptInd] = deathDays[ptInd]

        # Check if we have valid data for survival analysis
        if len(fuDays) == 0 or len(events) == 0 or len(fuDays) != len(events):
            fig = go.Figure(layout=layout)
            fig.add_annotation(
                text="Insufficient data for local failure analysis. Please check your date columns and recurrence data.",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig, 'Days: No valid data', 'Patients: 0', 'Events: 0'
        
        # Ensure data is numeric and valid
        try:
            fuDays = pd.to_numeric(fuDays, errors='coerce').dropna()
            events = pd.to_numeric(events, errors='coerce').dropna()
            
            # Align indices
            common_indices = fuDays.index.intersection(events.index)
            fuDays = fuDays.loc[common_indices]
            events = events.loc[common_indices]
            
            if len(fuDays) == 0 or len(events) == 0:
                raise ValueError("No valid local failure data after cleaning")
                
        except Exception as e:
            fig = go.Figure(layout=layout)
            fig.add_annotation(
                text=f"Error processing local failure data: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig, 'Days: Data error', 'Patients: 0', 'Events: 0'

        # fit and plot AJF
        survFig = go.Figure(layout=layout)

        if groupAxis and groupAxis in data.columns:
            group_series = data[groupAxis].astype(str)
            common_indices = fuDays.index.intersection(events.index)
            group_series = group_series.loc[common_indices]
            fuDays_aligned = fuDays.loc[common_indices]
            events_aligned = events.loc[common_indices]

            for cat in sorted(group_series.dropna().unique()):
                mask = group_series == cat
                if mask.any():
                    ajf = AalenJohansenFitter(calculate_variance=True, jitter_level=0.01)
                    ajf.fit(durations=fuDays_aligned[mask], event_observed=events_aligned[mask], event_of_interest=1)
                    survFig.add_trace(
                        go.Scatter(
                            x=ajf.cumulative_density_.index, y=ajf.cumulative_density_.values.flatten(), mode='lines+markers', 
                            marker=dict(opacity=0.9, symbol="line-ns-open", size=12),
                            line=dict(shape="vh", width=3),
                            name=str(cat),
                            showlegend=True,
                        ),
                    )
        else:
            ajf = AalenJohansenFitter(calculate_variance=True, jitter_level=0.01)
            ajf.fit(durations=fuDays, event_observed=events, event_of_interest=1)
            survFig.add_trace(
                go.Scatter(
                    x=ajf.cumulative_density_.index, y=ajf.cumulative_density_.values.flatten(), mode='lines+markers', 
                    marker=dict(opacity=0.9, symbol="line-ns-open", size=15, color="red"),
                    line=dict(shape="vh", width=4, color=uclaBlue),
                    showlegend=False,
                ),
            )

            ci_times = np.concatenate((ajf.cumulative_density_.index,ajf.cumulative_density_.index[::-1]), axis=0)
            ci_vals = np.concatenate((ajf.confidence_interval_['AJ_estimate_upper_0.95'],ajf.confidence_interval_['AJ_estimate_lower_0.95'][::-1]), axis=0)

            survFig.add_trace(
                go.Scatter(
                    x=ci_times,
                    y=ci_vals,
                    fill='toself',
                    fillcolor='rgba(100,0,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                )
            )

        survFig.update_layout(
            yaxis=dict(
                showgrid=True,
                tickfont=dict(size=15,color='black'),
                showticklabels=True,
                gridcolor= 'rgba(0.5,0.5,0.5,0.2)',
                title='',
                range=[-0.1,1.1],
            ),
            xaxis=dict(
                tickvals = [m * 365 for m in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.0, 4.5, 5.0]],
                ticktext = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.0, 4.5, 5.0]
            ),
        )

        eventStrDisp = 'Local Failures: {}'.format(len(events[events==1])) + ', Deaths: {}'.format(len(events[events==2]))        

    elif yAxis == "Distant Failure":
        data = data.dropna(subset=['Last follow up Distant Imaging Date'])

        # if no surgery date, set end of RT to surgery date
        nullSurgInds = data[data['Surgery Date'].isnull()]['Surgery Date'].index
        data.loc[nullSurgInds,'Surgery Date'] = data.loc[nullSurgInds, 'End of RT Date']

        dataTimes = data['Last follow up date']
        dataEventDates = data['Distant Recurrence Date']
        dataEvents = data['Distant Recurrence?']

        dataDeathDates = data['Date of Death']
        dataDeaths = data['Survival']

        # get follow up time in Days  
        times = pd.to_datetime(dataTimes) - pd.to_datetime(data['Surgery Date'])
        times = times.dt.days
        fuDays = times.copy()
        fuDays = fuDays[times>0]

        # find event times in days 
        eventTimes = pd.to_datetime(dataEventDates) - pd.to_datetime(data['Surgery Date'])
        eventTimes = eventTimes.dt.days
        eventDays = eventTimes[eventTimes>0]

        # find death time in days 
        deathTimes = pd.to_datetime(dataDeathDates) - pd.to_datetime(data['Surgery Date'])
        deathTimes = deathTimes.dt.days
        deathDays = deathTimes[deathTimes>0]

        # overwrite follow up time with event times for patients with events
        fuDays[eventTimes>0] = eventDays

        # get events 
        d = {
            'yes': 1,
            'not evaluable': 0,
            'no': 0,
            'dead': 2,
            'alive': 0,
        }
        events = dataEvents.str.lower()
        events = events.map(d)
        events = events.fillna(0)
        events = events[times>0]

        # code the deaths
        deaths = dataDeaths.str.lower()
        deaths = deaths.map(d)
        deaths = deaths.fillna(0)
        deaths = deaths[deathTimes>0]

        # code event as death if occurred prior to local recurrence
        for ptInd in deaths.index:
            if events.index.isin([(ptInd)]).any():
                if events.loc[ptInd] == 0:
                    events[ptInd] = 2
                    fuDays[ptInd] = deathDays[ptInd]

        # Check if we have valid data for survival analysis
        if len(fuDays) == 0 or len(events) == 0 or len(fuDays) != len(events):
            fig = go.Figure(layout=layout)
            fig.add_annotation(
                text="Insufficient data for distant failure analysis. Please check your date columns and recurrence data.",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig, 'Days: No valid data', 'Patients: 0', 'Events: 0'
        
        # Ensure data is numeric and valid
        try:
            fuDays = pd.to_numeric(fuDays, errors='coerce').dropna()
            events = pd.to_numeric(events, errors='coerce').dropna()
            
            # Align indices
            common_indices = fuDays.index.intersection(events.index)
            fuDays = fuDays.loc[common_indices]
            events = events.loc[common_indices]
            
            if len(fuDays) == 0 or len(events) == 0:
                raise ValueError("No valid distant failure data after cleaning")
                
        except Exception as e:
            fig = go.Figure(layout=layout)
            fig.add_annotation(
                text=f"Error processing distant failure data: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig, 'Days: Data error', 'Patients: 0', 'Events: 0'

        # fit and plot AJF
        survFig = go.Figure(layout=layout)

        if groupAxis and groupAxis in data.columns:
            group_series = data[groupAxis].astype(str)
            common_indices = fuDays.index.intersection(events.index)
            group_series = group_series.loc[common_indices]
            fuDays_aligned = fuDays.loc[common_indices]
            events_aligned = events.loc[common_indices]

            for cat in sorted(group_series.dropna().unique()):
                mask = group_series == cat
                if mask.any():
                    ajf = AalenJohansenFitter(calculate_variance=True, jitter_level=0.01)
                    ajf.fit(durations=fuDays_aligned[mask], event_observed=events_aligned[mask], event_of_interest=1)
                    survFig.add_trace(
                        go.Scatter(
                            x=ajf.cumulative_density_.index, y=ajf.cumulative_density_.values.flatten(), mode='lines+markers', 
                            marker=dict(opacity=0.9, symbol="line-ns-open", size=12),
                            line=dict(shape="vh", width=3),
                            name=str(cat),
                            showlegend=True,
                        ),
                    )
        else:
            ajf = AalenJohansenFitter(calculate_variance=True, jitter_level=0.01)
            ajf.fit(durations=fuDays, event_observed=events, event_of_interest=1)
            survFig.add_trace(
                go.Scatter(
                    x=ajf.cumulative_density_.index, y=ajf.cumulative_density_.values.flatten(), mode='lines+markers', 
                    marker=dict(opacity=0.9, symbol="line-ns-open", size=15, color="red"),
                    line=dict(shape="vh", width=4, color=uclaBlue),
                    showlegend=False,
                ),
            )

            ci_times = np.concatenate((ajf.cumulative_density_.index,ajf.cumulative_density_.index[::-1]), axis=0)
            ci_vals = np.concatenate((ajf.confidence_interval_['AJ_estimate_upper_0.95'],ajf.confidence_interval_['AJ_estimate_lower_0.95'][::-1]), axis=0)

            survFig.add_trace(
                go.Scatter(
                    x=ci_times,
                    y=ci_vals,
                    fill='toself',
                    fillcolor='rgba(100,0,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                )
            )

        survFig.update_layout(
            yaxis=dict(
                showgrid=True,
                tickfont=dict(size=15,color='black'),
                showticklabels=True,
                gridcolor= 'rgba(0.5,0.5,0.5,0.2)',
                title='',
                range=[-0.1,1.1],
            ),
            xaxis=dict(
                tickvals = [m * 365 for m in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.0, 4.5, 5.0]],
                ticktext = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.0, 4.5, 5.0]
            ),
        )        

        eventStrDisp = 'Distant Failures: {}'.format(len(events[events==1])) + ', Deaths: {}'.format(len(events[events==2]))


    return survFig, 'Days: {}'.format(np.median(fuDays)), 'Patients: {}'.format(len(fuDays)), eventStrDisp    

# map plots
@app.callback(
    Output("map-plot", "figure"), 
    [
    Input('table-filters', 'derived_virtual_indices'),
    Input('uploaded-data', 'data')
    ])
def display_map(dataInds, uploaded_data):
    if uploaded_data is None:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="Please upload a CSV file to view plots",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig
    
    df = pd.DataFrame(uploaded_data)
    
    if dataInds is None:
        data = df.copy()
    else:
        data = df.iloc[dataInds, :]

    # Check if Zipcode column exists
    if 'Zipcode' not in data.columns:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="Map requires 'Zipcode' column in your data",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig

    # Filter out rows with missing zipcodes
    data_with_zip = data.dropna(subset=['Zipcode'])
    
    if len(data_with_zip) == 0:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="No valid zipcode data found",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig

    # Initialize the geocoder for US postal codes
    nomi = pgeocode.Nominatim('us')
    
    # Convert zipcodes to coordinates
    latitudes = []
    longitudes = []
    valid_indices = []
    
    for idx, zipcode in data_with_zip['Zipcode'].items():
        try:
            # Clean the zipcode (remove any non-numeric characters except hyphens)
            clean_zip = str(zipcode).split('-')[0]  # Take only the first 5 digits
            clean_zip = ''.join(filter(str.isdigit, clean_zip))
            
            if len(clean_zip) >= 5:
                # Get coordinates for the zipcode
                location = nomi.query_postal_code(clean_zip[:5])
                
                if pd.notna(location.latitude) and pd.notna(location.longitude):
                    latitudes.append(location.latitude)
                    longitudes.append(location.longitude)
                    valid_indices.append(idx)
        except:
            # Skip invalid zipcodes
            continue
    
    if len(latitudes) == 0:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="No valid coordinates could be generated from zipcode data",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig

    # Filter data to only valid coordinate entries
    data_valid = data_with_zip.loc[valid_indices].copy()
    data_valid['latitude'] = latitudes
    data_valid['longitude'] = longitudes

    fig = go.Figure(layout=layout)

    # Use available columns for text display
    text_cols = []
    if 'Enrolled Patient #' in data_valid.columns:
        text_cols.append(data_valid['Enrolled Patient #'].astype(str))
    if 'Histology on Surgery' in data_valid.columns:
        text_cols.append(data_valid['Histology on Surgery'].astype(str))
    if 'Tumor Surgery Size' in data_valid.columns:
        text_cols.append('Size: ' + data_valid['Tumor Surgery Size'].astype(str))
    
    if text_cols:
        # Properly concatenate the Series with separators
        data_valid['mapText'] = text_cols[0]
        for i in range(1, len(text_cols)):
            data_valid['mapText'] = data_valid['mapText'] + ': ' + text_cols[i]
    else:
        data_valid['mapText'] = 'Patient Data'

    fig.add_trace(
        go.Scattergeo(
        lon = data_valid['longitude'],
        lat = data_valid['latitude'],
        text = data_valid['mapText'],
        mode = 'markers',
        marker_color = 'orange',  
        marker_size = 8,  
        customdata = np.stack((data_valid.get('Tumor Surgery Size', ['N/A']*len(data_valid)), 
                              data_valid.get('Histology on Surgery', ['N/A']*len(data_valid)), 
                              data_valid.index), axis=-1),
        hoverinfo='none'
        ),
    )

    fig.update_layout(
        geo_scope='usa',
        height=600,
        title=dict(
            text=f"Patient Distribution by Location (n={len(data_valid)})",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top'
        ),
    )    

    return fig

# Add tooltip callback for map
@app.callback(
    Output("map-tooltip", "show"),
    Output("map-tooltip", "bbox"),
    Output("map-tooltip", "children"),
    [Input("map-plot", "hoverData")]
)
def display_map_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update
 
    # Get the hovered point data
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    pt_id = hover_data["customdata"][2]  # Patient ID is stored in customdata[2]
    tumor_size = hover_data["customdata"][0]
    histology = hover_data["customdata"][1]

    try:
        # Create paths for each image
        basePath = 'assets/Captures_Nums_small/'
        axial_path = basePath + str(pt_id) + '/Axial.tiff'
        coronal_path = basePath + str(pt_id) + '/Coronal.tiff'
        sagittal_path = basePath + str(pt_id) + '/Sagittal.tiff'

        # Load and process images using PIL
        def load_and_resize_image(path):
            try:
                photo = Image.open(path).convert("RGBA")
                photo = photo.resize((50, 50))  # Using the same size as 3D tooltip
                # Convert to base64
                buffered = io.BytesIO()
                photo.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()
            except Exception as e:
                print(f"Error loading image {path}: {str(e)}")
                return None

        axial_img = load_and_resize_image(axial_path)
        coronal_img = load_and_resize_image(coronal_path)
        sagittal_img = load_and_resize_image(sagittal_path)

        # Create a div with the patient info and images
        children = [
            html.Div([
                html.Div([
                    html.H4(f"Patient {pt_id}", style={'textAlign': 'center', 'color': 'black', 'marginBottom': '5px', 'fontSize': '14px'}),
                    html.Div([
                        html.P([
                            html.Strong("Tumor Size: "),
                            f"{tumor_size}"
                        ], style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                        html.P([
                            html.Strong("Histology: "),
                            f"{histology}"
                        ], style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                    ], style={'marginBottom': '5px'}),
                ], style={'marginBottom': '5px'}),
                
                # Images in a row
                html.Div([
                    html.Div([
                        html.Img(
                            src=f'data:image/png;base64,{axial_img}' if axial_img else 'assets/imagePlaceholder.png',
                            style={
                                "width": "50px",
                                'display': 'block',
                                'margin': '2px auto',
                                'border': '1px solid black'
                            }
                        ),
                        html.P("Axial", style={'textAlign': 'center', 'color': 'black', 'margin': '1px', 'fontSize': '10px'}),
                    ], style={'display': 'inline-block', 'margin': '0 2px'}),
                    html.Div([
                        html.Img(
                            src=f'data:image/png;base64,{coronal_img}' if coronal_img else 'assets/imagePlaceholder.png',
                            style={
                                "width": "50px",
                                'display': 'block',
                                'margin': '2px auto',
                                'border': '1px solid black'
                            }
                        ),
                        html.P("Coronal", style={'textAlign': 'center', 'color': 'black', 'margin': '1px', 'fontSize': '10px'}),
                    ], style={'display': 'inline-block', 'margin': '0 2px'}),
                    html.Div([
                        html.Img(
                            src=f'data:image/png;base64,{sagittal_img}' if sagittal_img else 'assets/imagePlaceholder.png',
                            style={
                                "width": "50px",
                                'display': 'block',
                                'margin': '2px auto',
                                'border': '1px solid black'
                            }
                        ),
                        html.P("Sagittal", style={'textAlign': 'center', 'color': 'black', 'margin': '1px', 'fontSize': '10px'}),
                    ], style={'display': 'inline-block', 'margin': '0 2px'}),
                ], style={'textAlign': 'center'}),
            ], style={
                'backgroundColor': 'white',
                'padding': '5px',
                'border': '1px solid red',
                'borderRadius': '3px',
                'boxShadow': '0 0 5px rgba(0,0,0,0.2)',
                'zIndex': '10000',
                'position': 'absolute'
            })
        ]
        return True, bbox, children
    except Exception as e:
        print(f"Error in tooltip: {str(e)}")
        children = [
            html.Div([
                html.H4(f"Patient {pt_id}", style={'textAlign': 'center', 'color': 'black', 'fontSize': '14px'}),
                html.P(f"Tumor Size: {tumor_size}", style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                html.P(f"Histology: {histology}", style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                html.P(f"Error loading images: {str(e)}", style={'color': 'red', 'fontSize': '10px'})
            ], style={
                'backgroundColor': 'white',
                'padding': '5px',
                'border': '1px solid red',
                'borderRadius': '3px',
                'zIndex': '1000',
                'position': 'relative'
            })
        ]
        return True, bbox, children

# swimmer plots
@app.callback([
    Output("swimmer-plot", "figure"), 
    Output("wound-output", component_property='children'),
    ],
    [
    Input('table-filters', 'derived_virtual_indices'),
    Input('uploaded-data', 'data')
    ])
def display_swimmer(dataInds, uploaded_data):
    if uploaded_data is None:
        fig = go.Figure(layout=layout)
        fig.add_annotation(
            text="Please upload a CSV file to view plots",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig, '0'
    
    df = pd.DataFrame(uploaded_data)
    
    if dataInds is None:
        data = df.copy()
    else:
        data = df.iloc[dataInds, :]

    fig = go.Figure(layout=layout)

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(t=50,l=10,b=10,r=10),
    )

    # Look for potential date columns that could be used for swimmer plot
    potential_surgery_cols = [col for col in data.columns if 'surgery' in col.lower() and 'date' in col.lower()]
    potential_closure_cols = [col for col in data.columns if any(word in col.lower() for word in ['closure', 'close', 'wound']) and 'date' in col.lower()]
    potential_baseline_cols = [col for col in data.columns if any(word in col.lower() for word in ['rt', 'treatment', 'baseline', 'start']) and 'date' in col.lower()]
    
    # Try to find surgery date column
    surgery_col = None
    if potential_surgery_cols:
        surgery_col = potential_surgery_cols[0]
    elif 'Surgery Date' in data.columns:
        surgery_col = 'Surgery Date'
    
    # Try to find wound closure date column
    closure_col = None
    if potential_closure_cols:
        closure_col = potential_closure_cols[0]
    elif 'Date of wound closure' in data.columns:
        closure_col = 'Date of wound closure'
    
    # Try to find baseline date column
    baseline_col = None
    if potential_baseline_cols:
        baseline_col = potential_baseline_cols[0]
    elif 'End of RT Date' in data.columns:
        baseline_col = 'End of RT Date'
    
    # Check if we have minimum required columns
    if not surgery_col or not baseline_col:
        fig.add_annotation(
            text="Swimmer plot requires Surgery Date and Baseline Date columns.\nPlease ensure your data has date columns with 'surgery' and 'baseline/RT/treatment' in their names.",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color=uclaBlue)
        )
        return fig, '0'

    # Filter to patients with valid surgery dates
    valid_surgery_mask = data[surgery_col].notna() & data[baseline_col].notna()
    data = data[valid_surgery_mask]

    # Filter to only patients with wound complications
    wound_complication_cols = [col for col in data.columns if any(word in col.lower() for word in ['wound', 'complication']) and 'major' in col.lower()]
    if wound_complication_cols:
        wound_col = wound_complication_cols[0]
        data = data[data[wound_col].str.lower() == 'yes']
    elif 'Major Wound Complications' in data.columns:
        data = data[data['Major Wound Complications'].str.lower() == 'yes']
    else:
        # If no wound complication column found, show a message
        fig.add_annotation(
            text="Swimmer plot shows patients with wound complications.\nNo wound complication column found in data.",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color=uclaBlue)
        )
        return fig, '0'

    if len(data) == 0:
        fig.add_annotation(
            text="No patients with wound complications found",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color=uclaBlue)
        )
        return fig, '0'

    # Convert date columns to datetime
    try:
        data[surgery_col] = pd.to_datetime(data[surgery_col])
        data[baseline_col] = pd.to_datetime(data[baseline_col])
        if closure_col and closure_col in data.columns:
            data[closure_col] = pd.to_datetime(data[closure_col])
    except Exception as e:
        fig.add_annotation(
            text=f"Error converting dates: {str(e)}\nPlease check your date format.",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color=uclaBlue)
        )
        return fig, '0'

    # Calculate days from baseline
    data['Days to Surgery'] = (data[surgery_col] - data[baseline_col]).dt.days
    if closure_col and closure_col in data.columns:
        data['Days to Wound Close'] = (data[closure_col] - data[baseline_col]).dt.days
        # Sort by wound closure time if available, otherwise by surgery time
        data = data.sort_values('Days to Wound Close', na_position='last')
    else:
        data = data.sort_values('Days to Surgery')

    yShift = 0
    annotations = []
    numPts = len(data)

    for pt in range(len(data)):
        # Get variables
        surg = data['Days to Surgery'].iloc[pt]
        ptID = data.get('Enrolled Patient #', data.index).iloc[pt]
        
        # Determine maximum x value for timeline
        if closure_col and closure_col in data.columns:
            wound = data['Days to Wound Close'].iloc[pt]
            xMax = wound if pd.notna(wound) else surg
        else:
            wound = None
            xMax = surg
            
        # Draw horizontal timeline
        fig.add_shape(type="line",
                x0=0, y0=yShift, x1=xMax, y1=yShift,
                line=dict(color="gray", width=4))
        
        # Add wound closure time annotation if available
        if wound is not None and pd.notna(wound):
            annotations.append(dict(x=xMax + 10, y=yShift,
                                    xanchor='left', yanchor='middle',
                                    text=f"{int(wound)}d",
                                    font=dict(family='Arial', size=12, color='rgb(150,150,150)'),
                                    showarrow=False))

        # Add patient ID annotation
        annotations.append(dict(x=-5, y=yShift,
                                xanchor='right', yanchor='middle',
                                text=str(ptID),
                                font=dict(family='Arial', size=12, color='rgb(0,200,0)'),
                                showarrow=False))

        # Draw surgery marker (red)
        fig.add_trace(go.Scatter(x=[surg], y=[yShift], mode='markers', 
                                marker=dict(size=10, color='red'), 
                                name='Surgery', showlegend=False))

        # Draw wound closure marker if available (blue)
        if wound is not None and pd.notna(wound):
            fig.add_trace(go.Scatter(x=[wound], y=[yShift], mode='markers', 
                                    marker=dict(size=10, color='blue'), 
                                    name='Wound Closure', showlegend=False))
            
        # Increment vertical position
        yShift += 0.2

    # Configure axes
    max_days = data['Days to Surgery'].max() if len(data) > 0 else 365
    if closure_col and closure_col in data.columns:
        max_days = max(max_days, data['Days to Wound Close'].max())
    
    fig.update_xaxes(visible=True, fixedrange=True, range=[-25, max_days + 50])
    fig.update_yaxes(visible=False, fixedrange=True, range=[-0.5, yShift + 1])
    
    # Set x-axis tick marks for years
    tick_positions = [0, 365/2, 365, 365*1.5, 365*2, 365*2.5, 365*3]
    tick_labels = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    fig.update_xaxes(ticktext=tick_labels, tickvals=tick_positions)

    # Add title
    title_text = f"Surgery and Wound Closure Timeline (n={numPts})"
    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor='center'),
        annotations=annotations
    )

    return fig, str(numPts)

# imaging plots
@app.callback(
    Output("imaging-plot", "figure"), 
    [
    Input('table-filters', 'derived_virtual_indices'),
    Input("imaging-plane", "value"),
    Input('uploaded-data', 'data')
    ])
def display_image(dataInds, imagePlane, uploaded_data):
    
    if uploaded_data is None:
        # Return empty figure if no data uploaded
        fig = go.Figure()
        fig.add_annotation(
            text="Please upload data to view imaging",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            margin=dict(t=10,l=10,b=10,r=10),
            hovermode=False,
            width=800,
            height=600,
        )
        return fig
    
    df = pd.DataFrame(uploaded_data)
    
    if dataInds is None:
        data = df.copy()
    else:
        data = df.iloc[dataInds, :]

    ## sort data
    # data = data.loc[data['Major Wound Complications '] == 'Yes']
    # data.sort_values(by='Days to Wound Close', inplace=True)    

    # read in all patient images
    basePath = 'assets/Captures_Nums_small/'    
    imageType = imagePlane + '.tiff'
    
    numPts = len(data)
    
    # Create a more square-like grid by calculating optimal columns
    # Aim for roughly square aspect ratio
    numCols = max(6, int(np.ceil(np.sqrt(numPts))))  # More columns for better aspect ratio
    numRows = int(np.ceil(numPts / numCols))
    
    # Larger images to fill more space
    imgSizePx = 100  # Increased size
    spacing = 8  # Tighter spacing to fit more images
    
    # Calculate total dimensions
    totalWidth = (numCols * imgSizePx) + ((numCols - 1) * spacing)
    totalHeight = (numRows * imgSizePx) + ((numRows - 1) * spacing)
    
    # make a single PIL image from all patient images
    collage = Image.new("RGBA", (totalWidth, totalHeight), color=(255,255,255,255))    

    c = 0
    for row in range(numRows):
        for col in range(numCols):
            if c >= numPts:
                break
                
            # Calculate position
            x = col * (imgSizePx + spacing)
            y = row * (imgSizePx + spacing)
            
            # 13, 14, 15, 16, 19, 24, 27, 35, 37, 86, 114, 147, 149, 151, 153 (MISSING FROM DATA)
            # 13, 14, 15 ,16, 19, 24, 27, 35, 37, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161 (MISSING IMAGES)
            # 86, 114 is missing from the data but has images (MISSING FROM DATA BUT HAS IMAGES)
            ptID = data['Enrolled Patient #'].iloc[c]
            imagePath = basePath + str(ptID) + '/' + imageType
            file = imagePath
            try:
                photo = Image.open(file).convert("RGBA")
                photo = photo.resize((imgSizePx, imgSizePx))        
            except:
                photo = Image.open('assets/imagePlaceholder.png').convert("RGBA")
                photo = photo.resize((imgSizePx, imgSizePx))
            
            collage.paste(photo, (x, y))
            c += 1
            
        if c >= numPts:
            break

    fig = px.imshow(collage)
            
    # hide and lock down axes
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    # Set aspect ratio to be more square-like and fill the space
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(t=10,l=10,b=10,r=10),
        hovermode=False,
        # Set aspect ratio to be more square
        width=totalWidth,
        height=totalHeight,
        # Ensure the plot fills the available space
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True, scaleanchor="x", scaleratio=1),
    )    

    return fig


@app.callback(
    Output("3d-plot", "figure"),
    [
        Input("x-3d-dropdown", "value"),
        Input("y-3d-dropdown", "value"),
        Input("z-3d-dropdown", "value"),
        Input("color-3d-dropdown", "value"),
        Input("size-3d-dropdown", "value"),
        Input('uploaded-data', 'data')
    ],
)
def update_3d_plot(x_col, y_col, z_col, color_col, size_col, uploaded_data):
    if uploaded_data is None:
        return go.Figure().add_annotation(
            text="Please upload a CSV file to view plots",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    
    df = pd.DataFrame(uploaded_data)
    
    if not all([x_col, y_col, z_col, color_col, size_col]):
        return go.Figure().add_annotation(
            text="Please select all variables to display the 3D plot",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    # Convert X, Y, Z to numeric (required for 3D positioning)
    x_data = convert_to_numeric(df, x_col)
    y_data = convert_to_numeric(df, y_col)
    z_data = convert_to_numeric(df, z_col)
    
    # Handle color and size differently for categorical vs numeric
    # For color - if categorical, create distinct color mapping
    if color_col in df.columns:
        color_values = df[color_col]
        if color_values.dtype == 'object' or pd.api.types.is_categorical_dtype(color_values):
            # Categorical color: map unique values to distinct numbers
            unique_colors = color_values.dropna().unique()
            color_map = {val: i for i, val in enumerate(unique_colors)}
            color_data = color_values.map(color_map).astype(float)
            # Scale colors to use full colorscale range
            if len(unique_colors) > 1:
                color_data = color_data * (len(unique_colors) - 1) / max(1, color_data.max())
        else:
            color_data = convert_to_numeric(df, color_col)
    else:
        color_data = pd.Series([0] * len(df), index=df.index)
    
    # For size - if categorical, create distinct size mapping
    if size_col in df.columns:
        size_values = df[size_col]
        if size_values.dtype == 'object' or pd.api.types.is_categorical_dtype(size_values):
            # Categorical size: map unique values to distinct sizes
            unique_sizes = size_values.dropna().unique()
            size_map = {val: 8 + i * 4 for i, val in enumerate(unique_sizes)}  # Start at 8, increment by 4
            size_data = size_values.map(size_map).astype(float)
        else:
            size_data = convert_to_numeric(df, size_col)
    else:
        size_data = pd.Series([8] * len(df), index=df.index)  # Default size
    
    patient_ids = df.get('Enrolled Patient #', df.index)

    # Create mask for valid numeric values - only require X, Y, Z to be valid
    xyz_mask = ~(x_data.isna() | y_data.isna() | z_data.isna())
    
    if not xyz_mask.any():
        # Try a more permissive approach
        x_valid = ~x_data.isna()
        y_valid = ~y_data.isna()
        z_valid = ~z_data.isna()
        
        if x_valid.any() and y_valid.any() and z_valid.any():
            # Use rows where at least X, Y, Z have some valid data
            xyz_mask = x_valid & y_valid & z_valid
            if not xyz_mask.any():
                # Fill missing values for X, Y, Z with their means
                x_data = x_data.fillna(x_data.mean())
                y_data = y_data.fillna(y_data.mean())
                z_data = z_data.fillna(z_data.mean())
                xyz_mask = ~(x_data.isna() | y_data.isna() | z_data.isna())
        
        if not xyz_mask.any():
            return go.Figure().add_annotation(
                text="No valid numeric data points available for X, Y, Z axes",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )

    # Filter data using the XYZ mask and handle color/size separately
    x_data = x_data[xyz_mask]
    y_data = y_data[xyz_mask]
    z_data = z_data[xyz_mask]
    color_data = color_data[xyz_mask].fillna(0)  # Fill missing color with default
    size_data = size_data[xyz_mask].fillna(8)    # Fill missing size with default size
    patient_ids = patient_ids[xyz_mask]

    # Store original size values for tooltip
    original_sizes = size_data.copy()

    # For categorical size data, don't normalize - use the mapped values directly
    # For numeric size data, normalize to reasonable range
    if size_col in df.columns:
        size_values = df[size_col]
        if size_values.dtype == 'object' or pd.api.types.is_categorical_dtype(size_values):
            # Use categorical sizes directly (already mapped to distinct values)
            normalized_sizes = size_data
        else:
            # Normalize numeric size data to reasonable values (between 4 and 12)
            size_min, size_max = np.nanmin(size_data), np.nanmax(size_data)
            if size_min == size_max or np.all(np.isclose(size_data, size_min)):
                normalized_sizes = np.full_like(size_data, 8.0)  # If all values are the same, use default size
            else:
                normalized_sizes = 4 + 8 * (size_data - size_min) / (size_max - size_min)
    else:
        normalized_sizes = np.full_like(size_data, 8.0)  # Default size

    xLabels = np.repeat(x_col, len(x_data))
    yLabels = np.repeat(y_col, len(y_data))
    zLabels = np.repeat(z_col, len(z_data))
    colorLabels = np.repeat(color_col, len(color_data))

    scatterFig = go.Figure(layout=layout)
    scatterFig.add_trace(
        go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers',
            marker=dict(
                size=normalized_sizes,
                color=color_data,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(thickness=20),
                line=dict(
                    width=1,
                    color='black'
                )
            ),
            text=patient_ids,
            customdata=np.stack((original_sizes, color_data), axis=-1),
            hoverinfo='none'
        )
    )

    # Update the layout for 3D
    scatterFig.update_layout(
        scene=dict(
            xaxis=dict(
                title=x_col,
                gridcolor="white",
                showbackground=True,
                backgroundcolor="rgb(230, 230,230)",
            ),
            yaxis=dict(
                title=y_col,
                gridcolor="white",
                showbackground=True,
                backgroundcolor="rgb(230, 230,230)",
            ),
            zaxis=dict(
                title=z_col,
                gridcolor="white",
                showbackground=True,
                backgroundcolor="rgb(230, 230,230)",
            ),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
        ),
        margin=dict(l=50, r=50, b=50, t=80),
        showlegend=False,
        title=dict(
            text=f"3D Patient Data Visualization (n={len(x_data)})",
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top'
        ),
        hovermode='closest',
    )

    return scatterFig

@app.callback(
    Output("3d-tooltip", "show"),
    Output("3d-tooltip", "bbox"),
    Output("3d-tooltip", "children"),
    [Input("3d-plot", "hoverData"),
     Input("x-3d-dropdown", "value"),
     Input("y-3d-dropdown", "value"),
     Input("z-3d-dropdown", "value"),
     Input("color-3d-dropdown", "value"),
     Input("size-3d-dropdown", "value"),
     Input('uploaded-data', 'data')]
)
def display_3d_hover(hoverData, x_axis, y_axis, z_axis, color_axis, size_axis, uploaded_data):
    if hoverData is None or uploaded_data is None:
        return False, no_update, no_update
    
    df = pd.DataFrame(uploaded_data)
 
    # Get the hovered point data
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    pt_id = hover_data["text"]
    x_val = hover_data["x"]
    y_val = hover_data["y"]
    z_val = hover_data["z"]
    
    # Get color and size values from customdata if available, otherwise use fallback values
    try:
        if "customdata" in hover_data and hover_data["customdata"] is not None and len(hover_data["customdata"]) >= 2:
            size_val = hover_data["customdata"][0]
            color_val = hover_data["customdata"][1]
        else:
            # Fallback: try to look up in the dataframe by patient ID
            try:
                # First try to find the row with this patient ID
                if 'Enrolled Patient #' in df.columns:
                    patient_row = df[df['Enrolled Patient #'] == pt_id]
                    if not patient_row.empty:
                        color_val = patient_row[color_axis].iloc[0]
                        size_val = patient_row[size_axis].iloc[0]
                    else:
                        color_val = "N/A"
                        size_val = "N/A"
                else:
                    # Use index lookup
                    color_val = df.loc[pt_id, color_axis]
                    size_val = df.loc[pt_id, size_axis]
            except (KeyError, IndexError):
                color_val = "N/A"
                size_val = "N/A"
    except Exception:
        color_val = "N/A"
        size_val = "N/A"
    
    # Format the values
    if isinstance(color_val, (int, float)) and color_val != "N/A":
        color_val = f"{color_val:.2f}"
    if isinstance(size_val, (int, float)) and size_val != "N/A":
        size_val = f"{size_val:.2f}"

    try:
        # Create paths for each image
        basePath = 'assets/Captures_Nums_small/'
        axial_path = basePath + str(pt_id) + '/Axial.tiff'
        coronal_path = basePath + str(pt_id) + '/Coronal.tiff'
        sagittal_path = basePath + str(pt_id) + '/Sagittal.tiff'

        # Load and process images using PIL
        def load_and_resize_image(path):
            try:
                photo = Image.open(path).convert("RGBA")
                photo = photo.resize((50, 50))  # Using the same size as 3D tooltip
                # Convert to base64
                buffered = io.BytesIO()
                photo.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()
            except Exception as e:
                print(f"Error loading image {path}: {str(e)}")
                return None

        axial_img = load_and_resize_image(axial_path)
        coronal_img = load_and_resize_image(coronal_path)
        sagittal_img = load_and_resize_image(sagittal_path)

        # Create a div with the patient info and images
        children = [
            html.Div([
                html.Div([
                    html.H4(f"Patient {pt_id}", style={'textAlign': 'center', 'color': 'black', 'marginBottom': '5px', 'fontSize': '14px'}),
                    html.Div([
                        html.P([
                            html.Strong(f"{x_axis}: "),
                            f"{x_val:.2f}"
                        ], style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                        html.P([
                            html.Strong(f"{y_axis}: "),
                            f"{y_val:.2f}"
                        ], style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                        html.P([
                            html.Strong(f"{z_axis}: "),
                            f"{z_val:.2f}"
                        ], style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                        html.P([
                            html.Strong(f"{color_axis}: "),
                            f"{color_val}"
                        ], style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                        html.P([
                            html.Strong(f"{size_axis}: "),
                            f"{size_val}"
                        ], style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                    ], style={'marginBottom': '5px'}),
                ], style={'marginBottom': '5px'}),
                
                # Images in a row
                html.Div([
                    html.Div([
                        html.Img(
                            src=f'data:image/png;base64,{axial_img}' if axial_img else 'assets/imagePlaceholder.png',
                            style={
                                "width": "50px",
                                'display': 'block',
                                'margin': '2px auto',
                                'border': '1px solid black'
                            }
                        ),
                        html.P("Axial", style={'textAlign': 'center', 'color': 'black', 'margin': '1px', 'fontSize': '10px'}),
                    ], style={'display': 'inline-block', 'margin': '0 2px'}),
                    html.Div([
                        html.Img(
                            src=f'data:image/png;base64,{coronal_img}' if coronal_img else 'assets/imagePlaceholder.png',
                            style={
                                "width": "50px",
                                'display': 'block',
                                'margin': '2px auto',
                                'border': '1px solid black'
                            }
                        ),
                        html.P("Coronal", style={'textAlign': 'center', 'color': 'black', 'margin': '1px', 'fontSize': '10px'}),
                    ], style={'display': 'inline-block', 'margin': '0 2px'}),
                    html.Div([
                        html.Img(
                            src=f'data:image/png;base64,{sagittal_img}' if sagittal_img else 'assets/imagePlaceholder.png',
                            style={
                                "width": "50px",
                                'display': 'block',
                                'margin': '2px auto',
                                'border': '1px solid black'
                            }
                        ),
                        html.P("Sagittal", style={'textAlign': 'center', 'color': 'black', 'margin': '1px', 'fontSize': '10px'}),
                    ], style={'display': 'inline-block', 'margin': '0 2px'}),
                ], style={'textAlign': 'center'}),
            ], style={
                'backgroundColor': 'white',
                'padding': '5px',
                'border': '1px solid red',
                'borderRadius': '3px',
                'boxShadow': '0 0 5px rgba(0,0,0,0.2)',
                'zIndex': '10000',
                'position': 'absolute'
            })
        ]
        return True, bbox, children
    except Exception as e:
        print(f"Error in tooltip: {str(e)}")
        children = [
            html.Div([
                html.H4(f"Patient {pt_id}", style={'textAlign': 'center', 'color': 'black', 'fontSize': '14px'}),
                html.P(f"{x_axis}: {x_val:.2f}", style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                html.P(f"{y_axis}: {y_val:.2f}", style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                html.P(f"{z_axis}: {z_val:.2f}", style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                html.P(f"{color_axis}: {color_val}", style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                html.P(f"{size_axis}: {size_val}", style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                html.P(f"Error loading images: {str(e)}", style={'color': 'red', 'fontSize': '10px'})
            ], style={
                'backgroundColor': 'white',
                'padding': '5px',
                'border': '1px solid red',
                'borderRadius': '3px',
                'zIndex': '1000',
                'position': 'relative'
            })
        ]
        return True, bbox, children

# Add tooltip callback for box plot
@app.callback(
    Output("box-tooltip", "show"),
    Output("box-tooltip", "bbox"),
    Output("box-tooltip", "children"),
    [Input("box-plot", "hoverData"),
     Input('x-box', 'value'),
     Input('y-box', 'value')]
)
def display_box_hover(hoverData, x_axis, y_axis):
    if hoverData is None:
        return False, no_update, no_update
 
    # Get the hovered point data
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    pt_id = hover_data["customdata"][2]  # Patient ID is stored in customdata[2]
    x_val = hover_data["x"]
    y_val = hover_data["y"]

    try:
        # Create paths for each image
        basePath = 'assets/Captures_Nums_small/'
        axial_path = basePath + str(pt_id) + '/Axial.tiff'
        coronal_path = basePath + str(pt_id) + '/Coronal.tiff'
        sagittal_path = basePath + str(pt_id) + '/Sagittal.tiff'

        # Load and process images using PIL
        def load_and_resize_image(path):
            try:
                photo = Image.open(path).convert("RGBA")
                photo = photo.resize((50, 50))  # Using the same size as 3D tooltip
                # Convert to base64
                buffered = io.BytesIO()
                photo.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()
            except Exception as e:
                print(f"Error loading image {path}: {str(e)}")
                return None

        axial_img = load_and_resize_image(axial_path)
        coronal_img = load_and_resize_image(coronal_path)
        sagittal_img = load_and_resize_image(sagittal_path)

        # Create a div with the patient info and images
        children = [
            html.Div([
                html.Div([
                    html.H4(f"Patient {pt_id}", style={'textAlign': 'center', 'color': 'black', 'marginBottom': '5px', 'fontSize': '14px'}),
                    html.Div([
                        html.P([
                            html.Strong(f"{x_axis}: "),
                            f"{x_val:.2f}"
                        ], style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                        html.P([
                            html.Strong(f"{y_axis}: "),
                            f"{y_val:.2f}"
                        ], style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                    ], style={'marginBottom': '5px'}),
                ], style={'marginBottom': '5px'}),
                
                # Images in a row
                html.Div([
                    html.Div([
                        html.Img(
                            src=f'data:image/png;base64,{axial_img}' if axial_img else 'assets/imagePlaceholder.png',
                            style={
                                "width": "50px",
                                'display': 'block',
                                'margin': '2px auto',
                                'border': '1px solid black'
                            }
                        ),
                        html.P("Axial", style={'textAlign': 'center', 'color': 'black', 'margin': '1px', 'fontSize': '10px'}),
                    ], style={'display': 'inline-block', 'margin': '0 2px'}),
                    html.Div([
                        html.Img(
                            src=f'data:image/png;base64,{coronal_img}' if coronal_img else 'assets/imagePlaceholder.png',
                            style={
                                "width": "50px",
                                'display': 'block',
                                'margin': '2px auto',
                                'border': '1px solid black'
                            }
                        ),
                        html.P("Coronal", style={'textAlign': 'center', 'color': 'black', 'margin': '1px', 'fontSize': '10px'}),
                    ], style={'display': 'inline-block', 'margin': '0 2px'}),
                    html.Div([
                        html.Img(
                            src=f'data:image/png;base64,{sagittal_img}' if sagittal_img else 'assets/imagePlaceholder.png',
                            style={
                                "width": "50px",
                                'display': 'block',
                                'margin': '2px auto',
                                'border': '1px solid black'
                            }
                        ),
                        html.P("Sagittal", style={'textAlign': 'center', 'color': 'black', 'margin': '1px', 'fontSize': '10px'}),
                    ], style={'display': 'inline-block', 'margin': '0 2px'}),
                ], style={'textAlign': 'center'}),
            ], style={
                'backgroundColor': 'white',
                'padding': '5px',
                'border': '1px solid red',
                'borderRadius': '3px',
                'boxShadow': '0 0 5px rgba(0,0,0,0.2)',
                'zIndex': '10000',
                'position': 'absolute'
            })
        ]
        return True, bbox, children
    except Exception as e:
        print(f"Error in tooltip: {str(e)}")
        children = [
            html.Div([
                html.H4(f"Patient {pt_id}", style={'textAlign': 'center', 'color': 'black', 'fontSize': '14px'}),
                html.P(f"{x_axis}: {x_val:.2f}", style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                html.P(f"{y_axis}: {y_val:.2f}", style={'color': 'black', 'margin': '1px', 'fontSize': '12px'}),
                html.P(f"Error loading images: {str(e)}", style={'color': 'red', 'fontSize': '10px'})
            ], style={
                'backgroundColor': 'white',
                'padding': '5px',
                'border': '1px solid red',
                'borderRadius': '3px',
                'zIndex': '1000',
                'position': 'relative'
            })
        ]
        return True, bbox, children

# DataTable update callback
@app.callback(
    [Output('table-filters', 'columns'), Output('table-filters', 'data'), Output('table-filters', 'tooltip_data')],
    [Input('uploaded-data', 'data'),
     Input('field-types', 'data')]
)
def update_table_filters(data, field_types):
    if data is None or len(data) == 0:
        return [], [], []
    df = pd.DataFrame(data)
    
    # Filter out ignored columns if field_types contains ignored fields
    excluded_cols = ["latitude", "longitude"]  # Always exclude generated coordinate columns
    if field_types and 'ignored' in field_types:
        excluded_cols.extend(field_types['ignored'])
    
    columns = [
        {"name": i, "id": i, "deletable": False, "selectable": True}
        for i in df.columns
        if i not in excluded_cols
    ]
    
    # Create tooltip data for all cells to show full content when truncated
    tooltip_data = []
    for row_idx, row in df.iterrows():
        tooltip_row = {}
        for col in df.columns:
            if col not in excluded_cols:
                cell_value = str(row[col]) if pd.notna(row[col]) else ""
                tooltip_row[col] = {'value': cell_value, 'type': 'markdown'}
        tooltip_data.append(tooltip_row)
    
    return columns, df.to_dict('records'), tooltip_data

# --- Dynamic Dropdown Callbacks ---

# Scatter plot dropdowns
@app.callback(
    [Output('x-dropdown', 'options'),
     Output('y-dropdown', 'options'),
     Output('color-dropdown', 'options'),
     Output('size-dropdown', 'options'),
     Output('x-dropdown', 'value'),
     Output('y-dropdown', 'value'),
     Output('color-dropdown', 'value'),
     Output('size-dropdown', 'value')],
    [Input('field-types', 'data')]
)
def update_scatter_dropdowns(field_types):
    if not field_types:
        return [], [], [], [], None, None, None, None

    def pick(preferred_list, target_name):
        if not preferred_list or not target_name:
            return None
        target_norm = str(target_name).strip().lower()
        for col in preferred_list:
            if str(col).strip().lower() == target_norm:
                return col
        return None

    num_cols = field_types.get('numerical', [])
    cat_cols = field_types.get('categorical', [])

    num_options = [{'label': col, 'value': col} for col in num_cols]
    cat_options = [{'label': col, 'value': col} for col in cat_cols]
    color_options = [{'label': col, 'value': col} for col in (num_cols + cat_cols)]

    # Requested defaults
    x_default = pick(num_cols, 'Tumor Surgery Size') or (num_cols[0] if num_cols else None)
    y_default = pick(num_cols, 'V12 Skin Total') or (num_cols[1] if len(num_cols) > 1 else (num_cols[0] if num_cols else None))
    color_default = pick(num_cols + cat_cols, 'Tumor Surgery Size') or (cat_cols[0] if cat_cols else (num_cols[0] if num_cols else None))
    size_default = pick(num_cols, 'V12 Skin Total') or (num_cols[0] if num_cols else None)

    return num_options, num_options, color_options, num_options, x_default, y_default, color_default, size_default

# Histogram plot dropdowns
@app.callback(
    [Output('x-histo', 'options'),
     Output('group-histo', 'options'),
     Output('x-histo', 'value'),
     Output('group-histo', 'value')],
    [Input('field-types', 'data')]
)
def update_histogram_dropdowns(field_types):
    if not field_types:
        return [], [], None, None

    def pick(preferred_list, target_name):
        if not preferred_list or not target_name:
            return None
        target_norm = str(target_name).strip().lower()
        for col in preferred_list:
            if str(col).strip().lower() == target_norm:
                return col
        return None

    num_cols = field_types.get('numerical', [])
    cat_cols = field_types.get('categorical', [])

    num_options = [{'label': col, 'value': col} for col in num_cols]
    cat_options = [{'label': col, 'value': col} for col in cat_cols]

    # Requested defaults
    x_default = pick(num_cols, 'Tumor Surgery Size') or (num_cols[0] if num_cols else None)
    group_default = pick(cat_cols, 'Histology Category') or (cat_cols[0] if cat_cols else None)

    return num_options, cat_options, x_default, group_default

# Box plot dropdowns
@app.callback(
    [Output('x-box', 'options'),
     Output('y-box', 'options'),
     Output('x-box', 'value'),
     Output('y-box', 'value')],
    [Input('field-types', 'data')]
)
def update_box_dropdowns(field_types):
    if not field_types:
        return [], [], None, None

    def pick(preferred_list, target_name):
        if not preferred_list or not target_name:
            return None
        target_norm = str(target_name).strip().lower()
        for col in preferred_list:
            if str(col).strip().lower() == target_norm:
                return col
        return None

    num_cols = field_types.get('numerical', [])
    cat_cols = field_types.get('categorical', [])

    x_options = [{'label': col, 'value': col} for col in cat_cols]
    y_options = [{'label': col, 'value': col} for col in num_cols]

    # Requested defaults
    x_default = pick(cat_cols, 'AJCC Stage') or (cat_cols[0] if cat_cols else None)
    y_default = pick(num_cols, 'V12 Skin Total') or (num_cols[0] if num_cols else None)

    return x_options, y_options, x_default, y_default

# Survival plot dropdown
@app.callback(
    [Output('y-surv', 'options'),
     Output('y-surv', 'value')],
    [Input('field-types', 'data')]
)
def update_survival_dropdown(field_types):
    if not field_types:
        return [], None
    
    # Return predefined survival options with default
    survival_options = [
        {'label': 'Overall Survival', 'value': 'Overall Survival'},
        {'label': 'Local Failure', 'value': 'Local Failure'},
        {'label': 'Distant Failure', 'value': 'Distant Failure'}
    ]
    
    default_value = 'Overall Survival'  # Set default to Overall Survival
    
    return survival_options, default_value

# Survival grouping dropdown (categorical focus, e.g., Diabetes)
@app.callback(
    [Output('group-surv', 'options'),
     Output('group-surv', 'value')],
    [Input('field-types', 'data')]
)
def update_survival_group_dropdown(field_types):
    if not field_types:
        return [], None

    def pick(preferred_list, target_name):
        if not preferred_list or not target_name:
            return None
        target_norm = str(target_name).strip().lower()
        for col in preferred_list:
            if str(col).strip().lower() == target_norm:
                return col
        return None

    # Prefer categorical columns for grouping
    cat_cols = field_types.get('categorical', [])
    # Include numeric columns only if user wants; keep it simple: categorical only for now
    group_candidates = cat_cols
    group_options = [{'label': col, 'value': col} for col in group_candidates]

    # Prefer common clinical categories
    group_default = (
        pick(group_candidates, 'Diabetes') or
        pick(group_candidates, 'Histology Category') or
        (group_candidates[0] if group_candidates else None)
    )

    return group_options, group_default

# 3D plot dropdowns
@app.callback(
    [Output('x-3d-dropdown', 'options'),
     Output('y-3d-dropdown', 'options'),
     Output('z-3d-dropdown', 'options'),
     Output('color-3d-dropdown', 'options'),
     Output('size-3d-dropdown', 'options'),
     Output('x-3d-dropdown', 'value'),
     Output('y-3d-dropdown', 'value'),
     Output('z-3d-dropdown', 'value'),
     Output('color-3d-dropdown', 'value'),
     Output('size-3d-dropdown', 'value')],
    [Input('field-types', 'data')]
)
def update_3d_dropdowns(field_types):
    if not field_types:
        return [], [], [], [], [], None, None, None, None, None

    def pick(preferred_list, target_name):
        if not preferred_list or not target_name:
            return None
        target_norm = str(target_name).strip().lower()
        for col in preferred_list:
            if str(col).strip().lower() == target_norm:
                return col
        return None

    num_cols = field_types.get('numerical', [])
    cat_cols = field_types.get('categorical', [])

    num_options = [{'label': col, 'value': col} for col in num_cols]
    cat_options = [{'label': col, 'value': col} for col in cat_cols]
    color_options = [{'label': col, 'value': col} for col in (num_cols + cat_cols)]

    # Requested defaults
    x_default = pick(num_cols, 'Tumor Surgery Size') or (num_cols[0] if num_cols else None)
    y_default = pick(num_cols, 'V12 Skin Total') or (num_cols[1] if len(num_cols) > 1 else (num_cols[0] if num_cols else None))
    z_default = pick(num_cols, 'Age at Consult') or (num_cols[2] if len(num_cols) > 2 else (num_cols[0] if num_cols else None))
    color_default = pick(num_cols + cat_cols, 'Tumor Surgery Size') or (cat_cols[0] if cat_cols else (num_cols[0] if num_cols else None))
    size_default = pick(num_cols, 'V12 Skin Total') or (num_cols[0] if num_cols else None)

    return num_options, num_options, num_options, color_options, num_options, x_default, y_default, z_default, color_default, size_default

# set the pandas dataframe to be datatable and update patient count display 
@app.callback(
    Output('patient-count-display', 'children'),
    [Input('table-filters', 'derived_virtual_indices'),
     Input('uploaded-data', 'data')]
)
def update_patient_count_display(dataInds, uploaded_data):    
    if uploaded_data is None:
        return "No data uploaded"
    
    df = pd.DataFrame(uploaded_data)
    
    # set the data count
    if dataInds is None:
        return f"({len(df)} patients)"
    else:
        return f"({len(dataInds)} patients)"

# Total patient count callback  
@app.callback(
    Output('total-patient-count-display', 'children'),
    [Input('uploaded-data', 'data')]
)
def update_total_patient_count_display(uploaded_data):    
    if uploaded_data is None:
        return "No data uploaded"
    
    df = pd.DataFrame(uploaded_data)
    return f"({len(df)} patients)"

# Save field types callback - update to use the store
@app.callback(
    [Output('field-types', 'data', allow_duplicate=True),
     Output('save-timestamp-store', 'data')],
    [Input('save-field-types', 'n_clicks')],
    [State('field-type-table', 'data'),
     State('uploaded-data', 'data')],
    prevent_initial_call=True
)
def save_field_types(n_clicks, table_data, uploaded_data):
    if n_clicks == 0 or not table_data or not uploaded_data:
        return no_update, no_update
    
    # Process the field type selections from the table
    numerical_cols = []
    categorical_cols = []
    date_cols = []
    ignored_cols = []
    all_columns = []
    
    for row in table_data:
        field_name = row['Field Name']
        field_type = row['Type']
        
        if field_type == 'Numerical':
            numerical_cols.append(field_name)
            all_columns.append(field_name)
        elif field_type == 'Categorical':
            categorical_cols.append(field_name)
            all_columns.append(field_name)
        elif field_type == 'Date':
            date_cols.append(field_name)
            all_columns.append(field_name)
        elif field_type == 'Ignore':
            ignored_cols.append(field_name)
            # Don't add ignored fields to all_columns
    
    updated_field_types = {
        'numerical': numerical_cols,
        'categorical': categorical_cols,
        'date': date_cols,
        'ignored': ignored_cols,
        'all_columns': all_columns  # Only non-ignored columns
    }
    
    # Store timestamp in the store
    timestamp = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")
    
    return updated_field_types, timestamp

# Add a new callback to display the timestamp from the store
@app.callback(
    Output('field-save-timestamp', 'children'),
    [Input('save-timestamp-store', 'data')]
)
def display_save_timestamp(timestamp):
    if timestamp:
        return html.P(f"Last saved: {timestamp}", 
                     style={'color': '#2c5aa0', 'fontWeight': 'bold', 'textAlign': 'center', 'margin': '10px 0'})
    return ""

# Clear success message when new data is uploaded - REMOVED to prevent flickering

# New: Dynamic border color for dropdowns based on selection state
@app.callback(
    [
        Output('x-dropdown', 'style'),
        Output('y-dropdown', 'style'),
        Output('color-dropdown', 'style'),
        Output('size-dropdown', 'style'),
        Output('x-histo', 'style'),
        Output('group-histo', 'style'),
        Output('x-box', 'style'),
        Output('y-box', 'style'),
        Output('y-surv', 'style'),
        Output('group-surv', 'style'),
        Output('x-3d-dropdown', 'style'),
        Output('y-3d-dropdown', 'style'),
        Output('z-3d-dropdown', 'style'),
        Output('color-3d-dropdown', 'style'),
        Output('size-3d-dropdown', 'style'),
    ],
    [
        Input('x-dropdown', 'value'),
        Input('y-dropdown', 'value'),
        Input('color-dropdown', 'value'),
        Input('size-dropdown', 'value'),
        Input('x-histo', 'value'),
        Input('group-histo', 'value'),
        Input('x-box', 'value'),
        Input('y-box', 'value'),
        Input('y-surv', 'value'),
        Input('group-surv', 'value'),
        Input('x-3d-dropdown', 'value'),
        Input('y-3d-dropdown', 'value'),
        Input('z-3d-dropdown', 'value'),
        Input('color-3d-dropdown', 'value'),
        Input('size-3d-dropdown', 'value'),
    ]
)
def update_dropdown_border_styles(
    x_dropdown_value,
    y_dropdown_value,
    color_dropdown_value,
    size_dropdown_value,
    x_histo_value,
    group_histo_value,
    x_box_value,
    y_box_value,
    y_surv_value,
    group_surv_value,
    x3d_value,
    y3d_value,
    z3d_value,
    color3d_value,
    size3d_value,
):
    def style_for(value):
        is_selected = (
            value is not None and
            (not isinstance(value, str) or value.strip() != '') and
            (not isinstance(value, list) or len(value) > 0)
        )
        return {'borderColor': 'green' if is_selected else 'red', 'borderWidth': '3px'}

    return (
        style_for(x_dropdown_value),
        style_for(y_dropdown_value),
        style_for(color_dropdown_value),
        style_for(size_dropdown_value),
        style_for(x_histo_value),
        style_for(group_histo_value),
        style_for(x_box_value),
        style_for(y_box_value),
        style_for(y_surv_value),
        style_for(group_surv_value),
        style_for(x3d_value),
        style_for(y3d_value),
        style_for(z3d_value),
        style_for(color3d_value),
        style_for(size3d_value),
    )

if __name__ == "__main__":
    # Respect PORT env var for local runs (GAE sets this in production)
    port = int(os.environ.get("PORT", "8080"))
    app.run(host='0.0.0.0', port=port, debug=False)