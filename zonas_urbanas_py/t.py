# %
# performed a k-means clustering algorithm on the graph of the Laplacian
# %
# %%

import scipy.sparse as sps
import scipy.io
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
from IPython import get_ipython
from sklearn.cluster import KMeans
import osmnx as ox
import peartree as pt
from osmnx.utils_graph import add_edge_lengths
import random
import h5py
import zonas_urbanas

ox.config(log_console=True, use_cache=True)

# %%
# Matplotlib inline
get_ipython().run_line_magic('matplotlib', 'inline')

# %%
G = ox.graph_from_place('Berkeley, California, USA', network_type='walk')
G = add_edge_lengths(G)
G.to_undirected()

# %%
A = nx.adjacency_matrix(G)

# for each diagonal on the node matrix,
# calculate the total number of edges that are attached to that node
a_shape = A.shape
a_diagonals = A.sum(axis=1)
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
eigen = zonas_urbanas.Eigen(L_dense)
eigen.calcular()

# %%
# teste de eigen
MM = L_dense.copy()
QQ = eigen.auto_vetores().copy()
DD = np.diag(eigen.auto_valores()[0].copy())
MM2 = np.dot(np.dot(QQ, DD), QQ.T)
print(MM-MM2)  # should be zero

# %%
autovec = eigen.auto_vetores().copy()

# %%
autovec = np.multiply(autovec, -1)


# %%
x = autovec[:, 1]
y = autovec[:, 2]

ns = list(G.nodes())
spectral_coordinates = {ns[i]: (x[i], y[i]) for i in range(len(x))}

# %%
Gt = G.copy()
node_ref = list(Gt.nodes(data=True))
for i, node in node_ref:
    sc = spectral_coordinates[i]
    Gt.nodes[i]['x'] = sc[0]
    Gt.nodes[i]['y'] = sc[1]

# %%

fig, ax = plt.subplots(figsize=(10, 10))
options = {
    "node_color": "#8aedfc",
    "node_size": 5,
    "edge_color": '#e2dede',
    "linewidths": 0,
    "width": 0,
}
nx.draw(Gt, ax=ax, **options)

# %%
A_w = nx.adjacency_matrix(G, weight='weight')

# Next generate degrees matrix
a_shape_w = A_w.shape
a_diagonals_w = A_w.sum(axis=1)
D_w = scipy.sparse.spdiags(
    a_diagonals_w.flatten(),
    [0],
    a_shape_w[0],
    a_shape_w[1],
    format='csr'
)

# %%
matriz_teste = np.array(
    [[1, 2, 8],
     [2, 1, 1],
     [8, 1, 1]]
)

w_w, v_w = scipy.linalg.eigh(matriz_teste)  # w = eigenvalues, v = eigenvectors

eiT = zonas_urbanas.Eigen(matriz_teste)
eiT.calcular()
eiTve = eiT.auto_vetores()
eiTva = eiT.auto_valores()[0]

# %%
eiTve[:, 1]

# %%
v_w[:, 1]

# %%

w, v = scipy.linalg.eigh(L_dense)  # w = eigenvalues, v = eigenvectors

# %%
ei_vect = eigen.auto_vetores()

# %%
np.allclose(ei_vect[:, 1], v[:, 1])

# %%
ei_vect[:, 1]

# %%
v[:, 1]

# %%

mat = np.array(
    [[2, 1, 0, 0, 1, 0],
     [1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 0],
     [0, 0, 1, 0, 1, 1],
     [1, 1, 0, 1, 0, 0],
     [0, 0, 0, 1, 0, 0]]
)

w_w, v_w = scipy.linalg.eigh(mat)  # w = eigenvalues, v = eigenvectors

eiT = zonas_urbanas.Eigen(mat)
eiT.calcular()
eiTve = np.multiply(eiT.auto_vetores(), -1)
eiTva = eiT.auto_valores()[0]

# %%
eiTva
# %%

w_w

# %%
eiTve

# %%
v_w

#%%
MM = mat.copy()
QQ = eiT.auto_vetores().copy()
DD = np.diag(eiT.auto_valores()[0].copy())
MM2 = np.dot(np.dot(QQ, DD), QQ.T)
print(MM-MM2)  # should be zero

# %%
scipyw, scipyv = scipy.linalg.eigh(L_dense)
x = scipyv[:, 1]
y = scipyv[:, 2]

# %%

ns = list(G.nodes())
spectral_coordinates = {ns[i]: (x[i], y[i]) for i in range(len(x))}

Gt = G.copy()
node_ref = list(Gt.nodes(data=True))
for i, node in node_ref:
    sc = spectral_coordinates[i]
    Gt.nodes[i]['x'] = sc[0]
    Gt.nodes[i]['y'] = sc[1]


fig, ax = plt.subplots(figsize=(10, 10))
options = {
    "node_color": "#8aedfc",
    "node_size": 5,
    "edge_color": '#e2dede',
    "linewidths": 0,
    "width": 0.1,
}
nx.draw(Gt, ax=ax, **options)

# %%
