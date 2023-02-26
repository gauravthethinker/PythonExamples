import numpy as np
import pandas as pd
import plotly.express as px


corr = np.array([[1, 0.7, 0.7], [0.7, 1, 0.7], [0.7, 0.7, 1]])
print(corr); 

chol = np.linalg.cholesky(corr)
print(chol)