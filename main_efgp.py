import pandas as pd
from fuzzy_granular import EFGPredictor as FBeM
from graph_utils import *
from utils import *
from river import stream
import matplotlib.pyplot as plt

n = 12

xls = pd.ExcelFile("datasets/DeathValleyAvg.xls")
sheetx = xls.parse(0)

# Preparing x
x = []
for index, row in sheetx.iterrows():
    x = x + row[1:].tolist()
xavg = np.zeros((n, len(x) - n - 1))

# Preparing y
data_structure = {i: [] for i in range(0, n)}
data_structure['y'] = []

yavg = []
for i in range(0, len(x) - n - 1):
    for j in range(0, n):
        xavg[j][i] = x[i + j]
        data_structure[j].append(x[i+j])
    yavg.insert(i, x[i + j + 1])
    data_structure['y'].append(x[i+j+1])

dataset = pd.DataFrame(data_structure)

fbi = FBeM()
fbi.debug = True
to_normalize = 0
fbi.n = n

# Normalize data
if to_normalize:
    fbi.__rho = 0.3
    min_v = min(yavg)
    max_v = max(yavg)
    xavg = normalize(array=xavg, min=min_v, max=max_v)
    yavg = normalize(array=yavg, min=min_v, max=max_v)
else:
    min_v = min(yavg)
    max_v = max(yavg)
    fbi.__rho = 0.2 * (max_v - min_v)

axis_1 = len(yavg)
axis_2 = len(xavg)

x = []
y = []

from river import metrics

ys = []
yhats = []
metric = metrics.RMSE()

"""
for i in range(0, axis_1):
    x = []
    y = []
    for j in range(0, axis_2):
        x.append(xavg[j][i])

    y.append(yavg[i])

    y_hat, _ = fbi.predict_one(x)
    # onde está _, se recebe os limites superior e inferior em forma de vetor[2].

    yhats.append(y_hat)
    ys.append(yavg[i])

    metric.update(yavg[i], y_hat)
    fbi.learn_one(x=x, y=y)
    print(metric)

print(metric)
"""
for x, y in stream.iter_pandas(dataset.drop(columns=['y']), dataset["y"]):
    if fbi.h == 0:
        fbi.n = len(x.values())
    y_hat, _ = fbi.predict_one(x)
    # onde está _, se recebe os limites superior e inferior em forma de vetor[2].

    yhats.append(y_hat)
    ys.append(y)

    metric.update(y, y_hat)
    fbi.learn_one(x=x, y=[y])
    print(metric)

import seaborn as sns

sns.lineplot(x=range(0, len(ys)), y=ys)
sns.lineplot(x=range(0, len(yhats)), y=yhats)
plt.show();