import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import plotly as ptly


""" Loading the visualization Data """
visual_data = np.load('./visual_data.npy')
data = visual_data[:, 0:3]
labels = visual_data[:, 3:21].astype(int)
performance = np.load('./performance.npy')

""" Initializing the app """
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.H1('PosterLens Clustering', style={'text-align': 'center'}),
    html.Div([
        html.Div([
            dcc.Dropdown(id='cluster_method',
                         options=[
                             {'label': 'K-Means', 'value': 'kmeans'},
                             {'label': 'Guassian Mixture', 'value': 'gm'},
                             {'label': 'Hierarchical', 'value': 'h'}],
                         multi=False,
                         value='kmeans',
                         style={'width': '60%', 'margin-left': '10%'}
                         )
        ],

            style={'width': '100%', 'height': 'max-content', 'display': 'flex', 'flex-direction': 'row',
                   'justify-content': 'center', 'align-items': 'center'},
        ),
        html.Div([
            dcc.Dropdown(id="clusters",
                         options=[
                             {"label": "3", "value": 0},
                             {"label": "5", "value": 1},
                             {"label": "7", "value": 2},
                             {"label": "10", "value": 3},
                             {"label": "15", "value": 4},
                             {"label": "20", "value": 5}],
                         multi=False,
                         value=0,
                         style={'width': "40%", 'margin-left': '12%'}
                         )
        ], style={'width': '100%', 'height': 'max-content', 'display': 'flex', 'flex-direction': 'row',
                  'justify-content': 'center', 'align-items': 'center'}, )
    ],
        style={'width': '100%', 'height': 'max-content', 'display': 'flex', 'flex-direction': 'column',
               'justify-content': 'center', 'align-items': 'center'},
    ),
    html.Br(),
    dcc.Graph(id='3dplot', figure={}),
    dcc.Graph(id='performance', figure={})
])


@app.callback(
    [Output(component_id='performance', component_property='figure'),
     Output(component_id='3dplot', component_property='figure')],
    [Input(component_id='cluster_method', component_property='value'),
     Input(component_id='clusters', component_property='value')])
def upgrade_graph(method_slctd, n_clusters):
    """ The Scatter Plot """
    plot = []
    if method_slctd == 'kmeans':
        pad = 0
    elif method_slctd == 'gm':
        pad = 6
    else:
        pad = 12
    for i in range(np.unique(labels[:, n_clusters]).size):
        plot.append(go.Scatter3d(
            x=data[labels[:, n_clusters + pad] == i, 0],
            y=data[labels[:, n_clusters + pad] == i, 1],
            z=data[labels[:, n_clusters + pad] == i, 2],
            mode='markers',
            marker=dict(size=3,
                        color=i,
                        opacity=1
                        ),
            showlegend=True,
            name='Cluster %d' % i
        )
        )
    """ The Elbow method Plot """
    fig1 = go.Figure(data=plot)
    plot2 = []
    clusters = ['K-Means', 'Gaussian Mixture', 'Hierarchical Clustering']
    for i in range(3):
        plot2.append(go.Scatter(
            x=np.array([3, 5, 7, 10, 15, 20]),
            y=performance[6 * i:((6 * i) + 6)],
            name=clusters[i]
        )
        )
    fig2 = go.Figure(data=plot2)
    fig1.update_layout(title=' Scatter Plot ', title_x=0.5, scene=dict(
                    xaxis_title='PCA: 1',
                    yaxis_title='PCA: 2',
                    zaxis_title='PCA: 3')
                    )
    fig2.update_layout(title=' Elbow method with Distortion ', title_x=0.5)
    fig2.update_xaxes(title='No of clusters')
    fig2.update_yaxes(title='Distortion')
    return fig2, fig1


if __name__ == '__main__':
    app.run_server(debug=False)
