# %%

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
from IPython import get_ipython
import random
import tensorflow as tf

# Matplotlib inline
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    # provavelmente não está executando IPython
    pass


# %%
NOME_ARQUIVO = './matriz_sparca_rolante.mtx'
# scipy.io.mmwrite(NOME_ARQUIVO, L, symmetry='general')

L = scipy.io.mmread(NOME_ARQUIVO)

# %%
tensor = tf.convert_to_tensor(L.todense(), dtype=tf.float32)

# %%
values, vectors = tf.linalg.eigh(tensor)

# %%
NOME_ARQUIVO_VETORES = './autovetores_rolante.npy'
np.save(NOME_ARQUIVO_VETORES, vectors, allow_pickle=False)

# %%
np.load(NOME_ARQUIVO_VETORES)

# %%
