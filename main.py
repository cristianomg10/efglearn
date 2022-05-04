import pandas as pd
from graph_utils import *
from sklearn.model_selection import train_test_split
import numpy as np
from timeit import default_timer as timer
from river.metrics import Accuracy
from efg import EFGClassifier

n = 4

rules = []
acc = []
tempo_gasto = []

for i in range(0, n):

    fbi = EFGClassifier()
    acc_r = Accuracy()
    fbi.debug = True
    to_normalize = 0

    data = pd.read_csv('https://bit.ly/iris_ds', header=None)
    data = data.sample(frac=1)

    # data = data.iloc[:-140,:]

    X = data.drop(columns=[4], axis=1)
    y = data[4].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

    # x_treino, x_teste, y_treino, y_teste = train_test_split(X, y)
    x_treino = X
    y_treino = y

    # Normalize data
    if to_normalize:
        fbi.rho = 1.
        fbi.hr = 20
        fbi.eta = 0.5


    else:
        fbi.rho = 1
        fbi.hr = 20
        fbi.eta = 0.5

    fbi.n = len(x_treino.columns)

    axis_1 = len(y_treino)
    axis_2 = len(x_treino)

    from timeit import default_timer as timer

    agora = timer()
    for i in range(0, axis_1):
        fbi.learn(x=x_treino.iloc[i], y=[y_treino.iloc[i]])
        print(i, y_treino.iloc[i], fbi.P[-1])
        acc_r.update(y_treino.iloc[i], fbi.P[-1])
        print(i, fbi.P)
    tempo_gasto.append(timer() - agora)
    print(f"Tempo gasto {timer() - agora}")

    # sleep(.75)
    # clear_output(wait=True)
    # plt.figure()
    # plot_granules(fbi)
    # plt.show()

    # axis_1 = len(y_teste)
    # axis_2 = len(x_teste)

    # for i in range(0, axis_1):
    #    fbi.learn(x=x_teste.iloc[i], y=[y_teste.iloc[i]])

    fbi.file.close()

    print("Final accuracy: ", fbi.acc[len(fbi.acc) - 1])
    acc.append(fbi.acc[len(fbi.acc) - 1])
    rules.append(np.mean(fbi.store_num_rules))

print(fbi.wrong)
print(fbi.right)
print("Rules: " + str(rules))
print("Acc: " + str(acc))
print(f"Acc {acc_r.get()}")
print(f"Acc: {np.mean(acc)} +/- {np.std(acc)}")
print(f"Tempo gasto {np.mean(tempo_gasto)} +/- {np.std(tempo_gasto)}")
print(fbi.granules[0].output_granules[0].coef)

plot_granules(fbi)
len(fbi.granules)