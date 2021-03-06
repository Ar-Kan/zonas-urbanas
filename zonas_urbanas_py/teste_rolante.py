# %%

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

ox.config(log_console=True, use_cache=True)

# %%
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


# %%
G = ox.graph_from_place(
    'Rolante, Rio Grande do Sul, Brazil',
    network_type='walk'
)
G = add_edge_lengths(G)

# %%
# fig, ax = ox.plot_graph(G)

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

# L_dense = L.todense()

# %%
NOME_ARQUIVO = './matriz_sparca_rolante.mtx'
scipy.io.mmwrite(NOME_ARQUIVO, L, symmetry='general')


# %%
# eiT = zonas_urbanas.Eigen(L_dense)
IGNORAR_LINHAS_ATE = 3
eiT = zonas_urbanas.Eigen(NOME_ARQUIVO, L.shape[0], IGNORAR_LINHAS_ATE)
eiT.calcular()
e_vectors = eiT.auto_vetores()
e_values = eiT.auto_valores()[0]

# e_values, e_vectors = scipy.linalg.eigh(
#     L_dense
# )
# %%
# plot_eigen_graph(G, e_vectors)
NOME_IMAGEM = 'grafo_rolante'
plot_eigen_graph(G, e_vectors, plot=False, save=True, nome=NOME_IMAGEM)

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

# %%
ox.plot_graph(
    G,
    node_color=cores_nodes,
    save=True, filepath=f'{NOME_IMAGEM}_vizinhancas.png'
)
plt.show()

# %%
