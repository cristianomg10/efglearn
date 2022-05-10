import pandas as pd
from graph_utils import *
from sklearn.model_selection import train_test_split
import numpy as np
from timeit import default_timer as timer
from river.metrics import Accuracy
from river import stream
from efg import EFGClassifier

n = 5

rules = []
acc = []
tempo_gasto = []

for i in range(0, n):

    fbi = EFGClassifier()
    acc_r = Accuracy()
    fbi.debug = True
    to_normalize = 0

    # Normalize data
    if to_normalize:
        fbi.rho = 1.
        fbi.hr = 20
        fbi.eta = 0.5
    else:
        fbi.rho = 1
        fbi.hr = 20
        fbi.eta = 0.5

    from timeit import default_timer as timer

    agora = timer()
    df = pd.read_csv('https://bit.ly/iris_ds', names=['petal_w', 'petal_l', 'sepal_w', 'sepal_l', 'class'])
    df = df.sample(frac=1)

    for x, y in stream.iter_pandas(df.drop(columns=['class']), df["class"]):
        if fbi.h == 0:
            fbi.n = len(x.values())

        fbi.learn_one(x=x, y=y)
        acc_r.update(fbi.numeric_class[y], fbi.P[-1])

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

    print("Final accuracy: ", acc_r.get() * 100)
    acc.append(acc_r.get() * 100)
    rules.append(np.mean(fbi.store_num_rules))

print(fbi.wrong)
print(fbi.right)
print("Rules: " + str(rules))
print("Acc: " + str(acc))
print(f"Acc {acc_r.get() * 100}")
print(f"Acc: {np.mean(acc)} +/- {np.std(acc)}")
print(f"Tempo gasto {np.mean(tempo_gasto)} +/- {np.std(tempo_gasto)}")
#print(fbi.granules[0].output_granules[0].coef)

plot_granules(fbi)
# len(fbi.granules)
