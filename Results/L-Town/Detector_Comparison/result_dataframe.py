import numpy as np
import pandas as pd

trained_on_real = np.load('trained_on_real.npy')
trained_on_toy = np.load('trained_on_toy.npy')
picked_junctions = np.load('picked_junctions.npy')

df = pd.DataFrame(
	{'real': trained_on_real, 'toy': trained_on_toy},
	index=picked_junctions
)
df.index.name = 'junction'
