import networkx as nx;
import numpy as np;
from joblib import Parallel, delayed;
import multiprocessing;

def core_func(repeat_index, G, numpy_arrary_2D):
  for u in G.nodes():
    numpy_arrary_2D[repeat_index][u] = 2
  return

if __name__ == "__main__":
  G = nx.erdos_renyi_graph(100,0.99)
  nRepeat = 50
  numpy_array = np.zeros([nRepeat,G.number_of_nodes()])
  Parallel(n_jobs=4, prefer="threads")(delayed(core_func)(repeat_index, G, numpy_array) for repeat_index in range(nRepeat))
  print(np.mean(numpy_array))