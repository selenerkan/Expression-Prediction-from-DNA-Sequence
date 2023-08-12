import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from Bio.GenBank import FeatureParser
with open("../data/yeast pTpA GPRA vector with N80.gb") as file_handle:
    parser = FeatureParser()
    seq_record = parser.parse(file_handle)

y_tr = pd.read_csv('../data/train_cluster_float_sequences.txt', sep="\t", header=None)[1].values
x_tr = np.load("../data/train_cluster_float_features.npy")

x_tst = np.load("../data/test_features.npy")

# selected_indices = np.random.choice(np.arange(len(y_tr)),20000)
# x_tr = x_tr[selected_indices].astype(np.float64)
# y_tr = y_tr[selected_indices]


# x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, train_size=0.8)

# regressor = MLPRegressor(verbose=True, learning_rate_init=0.001)
# regressor.fit(x_tr, y_tr)
# tr_pred = regressor.predict(x_tr)
# tr_score = r2_score(y_tr,tr_pred)

# val_pred = regressor.predict(x_val)
# val_score = r2_score(y_val, val_pred)

# print(f"TR Score: {tr_score}, VAL Score: {val_score}")
# plt.hist(val_pred, color = 'blue', edgecolor = 'black',bins=int(180/5))
# plt.show()

import pandas as pd 
x_tst = x_tr
print(x_tst.shape)
print(x_tst[np.where(x_tst.max(axis=1)!=0)])
x_tst = x_tst[np.where(x_tst.max(axis=1)!=0)]

print(x_tst.shape)
data = pd.DataFrame(x_tst).idxmax(1)
#labels = pd.Series(y_tr)
groups = pd.DataFrame(data,columns=["gene type"])#.groupby(by="gene type")
groups["gene type"] = data
#groups["expression"] = labels
groups = groups.groupby(by="gene type")

#print(groups)

for data in groups:
    if len(data[1]["gene type"]) > 1000:
        print(data[0], len(data[1]["gene type"]), end=" ")
        print(seq_record.features[data[0]].qualifiers["label"])