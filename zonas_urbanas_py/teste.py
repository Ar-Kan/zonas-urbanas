import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import scipy.io
from scipy.sparse import csr_matrix
from IPython import get_ipython
from sklearn.cluster import KMeans
import osmnx as ox
from osmnx.utils_graph import add_edge_lengths
import random
import zonas_urbanas

mat = csr_matrix(
    [[2, 1, 0, 0, 1, 0],
     [1, 0, 1, 0, 1, 0],
     [0, 1, 0, 1, 0, 0],
     [0, 0, 1, 0, 1, 1],
     [1, 1, 0, 1, 0, 0],
     [0, 0, 0, 1, 0, 0]]
)


NOME_ARQUIVO = './matriz_sparca_teste.mtx'
scipy.io.mmwrite(NOME_ARQUIVO, mat, field='integer', symmetry='general')

IGNORAR_LINHAS_ATE = 3
eiT = zonas_urbanas.Eigen(NOME_ARQUIVO, mat.shape[0], IGNORAR_LINHAS_ATE)
eiT.calcular()
e_vectors = eiT.auto_vetores()
e_values = eiT.auto_valores()[0]
print(e_vectors)
print(e_values)

eiT = zonas_urbanas.Eigen(mat.toarray())
eiT.calcular()
e_vectors = eiT.auto_vetores()
e_values = eiT.auto_valores()[0]
print(e_vectors)
