# %%

from shapely.geometry import shape, GeometryCollection
import json
import itertools
import fiona
from shapely.geometry import shape
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import scipy.io
from IPython import get_ipython
from sklearn.cluster import KMeans
import osmnx as ox
from osmnx.utils_graph import add_edge_lengths
import random
import zonas_urbanas
import geopandas as gpd
import momepy
from contextily import add_basemap
from libpysal import weights
import peartree as pt

ox.config(log_console=True, use_cache=True)

# Matplotlib inline
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    # provavelmente não está executando IPython
    pass

# %%


def plot_eigen_graph(graph, eigenvectors, plot=True, save=False, nome='grafo'):
    x = eigenvectors[:, 1]
    y = eigenvectors[:, 2]
    ns = list(graph.nodes())
    spectral_coordinates = {ns[i]: (x[i], y[i]) for i in range(len(x))}

    fig, ax = plt.subplots(figsize=(10, 10))
    options = {
        'node_color': 'blue',
        'node_size': 1.5,
        'linewidths': 0,
        'width': 0.1,
        'arrows': False,
        'with_labels': False
    }
    nx.draw(graph, pos=spectral_coordinates, ax=ax, **options)
    if save:
        plt.savefig(f'{nome}.png', bbox_inches='tight')
    if plot:
        plt.show()
    else:
        plt.close(fig)


def plot_graph(graph):
    options = {
        'node_color': 'blue',
        'node_size': 1.5,
        'linewidths': 0,
        'width': 0.1,
        'arrows': False,
        'with_labels': False
    }
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw(graph, ax=ax, **options)
    plt.show()


# %%
# G = ox.graph_from_place(
#     'Porto Alegre, Rio Grande do Sul, Brazil',
#     network_type='drive'
# )
REGIAO_CENTRAL = './ruas_central/regiao_central.json'
regiao_central = gpd.read_file(REGIAO_CENTRAL)
G = ox.graph_from_polygon(
    regiao_central["geometry"].unary_union,
    network_type='walk'
)
G = add_edge_lengths(G)

# %%
# impute speed on all edges missing data
G = ox.add_edge_speeds(G)
# calculate travel time (seconds) for all edges
G = ox.add_edge_travel_times(G)

# %%
fig, ax = ox.plot_graph(G)

# %%
# convert the OSMnx directed graph into an undirected one
G.to_undirected()

# adjacency matrix
A = nx.adjacency_matrix(G)

# for each diagonal on the node matrix,
# calculate the total number of edges that are attached to that node
a_shape = A.shape
a_diagonals = A.sum(axis=1)
# Return a sparse matrix from diagonals
D = scipy.sparse.spdiags(
    a_diagonals.flatten(),
    [0],
    a_shape[0],
    a_shape[1],
    format='csr'
)

# calculating the Laplacian
L = D - A

L_dense = L.todense()

# %%
eiT = zonas_urbanas.Eigen(L_dense)
eiT.calcular()
e_vectors = eiT.auto_vetores()
e_values = eiT.auto_valores()[0]

# %%
NOME_IMAGEM = 'grafo_porto_alegre'
plot_eigen_graph(G, e_vectors, plot=True, save=True, nome=NOME_IMAGEM)

# %%
x = e_vectors[:, 1]
y = e_vectors[:, 2]
ns = list(G.nodes())

spectral_coordinates = [[x[i], y[i]] for i in range(len(x))]

# %%
N_CLUSTERS = 12
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(
    spectral_coordinates
)
groupings = kmeans.labels_
nodes_groupings = dict(zip(G.nodes(), groupings))

# %%
cores_grupos = [
    f'#{random.randrange(0x1000000):06x}' for _ in range(N_CLUSTERS+1)
]
cores_nodes = [
    cores_grupos[nodes_groupings.get(node)] for node in G.nodes()
]

ox.plot_graph(
    G,
    node_color=cores_nodes,
    save=True, filepath=f'{NOME_IMAGEM}_vizinhancas.png'
)
plt.show()

# %%
