import six
import sys
sys.modules['sklearn.externals.six'] = six
import pydot
import pandas as pd
from id3 import Id3Estimator
from id3 import export_graphviz
from subprocess import check_call
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

np.float = float    
np.int = int   #module 'numpy' has no attribute 'int'
np.object = object    #module 'numpy' has no attribute 'object'
np.bool = bool    #module 'numpy' has no attribute 'bool'
#read in the raw data
raw_df = pd.read_csv("raw_data.csv")

#select only relevant attributes
df = raw_df[["finalRank", "circuitName", "year", "raceRound", "ageAtRace", "seasonAvgPlace", "qualPos", "pointsGoingIn", \
            "winsGoingIn", "recentPlacement", "driverCircuitAvgPlace", "teamCircuitAvgPlace", "positionChange", "seasonOvertake", \
            "careerOvertake"]]

#encode circuitNames to ints
circuit_names = df["circuitName"].unique()
circuit_encode = {}

for idx, circuit in enumerate(circuit_names):
    circuit_encode[circuit] = idx

df['circuitName'] = df['circuitName'].apply(lambda x: circuit_encode[x])

#split into test and train sets
train, test = train_test_split(df, test_size=0.2)

#get the target labels
y = train['finalRank'].values

#get the data
no_target = train.drop(columns = ['finalRank'])
X = no_target.to_numpy()

#explicitly convert strings to floats
for row in X:
    row[0] = float(row[0])

#use the id3 algorithm to generate a decision tree given the data
#id3_estimator = Id3Estimator(prune=True)
#id3_estimator.fit(train_data, df_target)

#use the randomforestclassifier to generate decision trees given the data
#print(X)
clf = RandomForestClassifier(n_estimators = 100, ccp_alpha= 0)
clf.fit(X, y)

#generate graph as .dot
#export_graphviz(id3_estimator.tree_, 'f1_tree.dot', list(no_target))
#convert .dot to .png
#check_call(['dot','-Tpng','f1_tree.dot','-o','f1_tree.png'])

def plot_matrix(test_labels, predictions):
    conf_matrix = confusion_matrix(test_labels, predictions)

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_loss(loss, title):
    plt.plot(range(len(loss)), loss)
    plt.title(title)
    plt.show()

test_labels = test['finalRank'].values

no_target_test = test.drop(columns = ['finalRank'])
test_data = no_target_test.to_numpy()

for row in test_data:
    row[0] = float(row[0])

predict_labels = clf.predict(test_data)

#Generate metrics

acc = accuracy_score(test_labels, predict_labels)
f1 = f1_score(test_labels, predict_labels, average="weighted")

print("Accuracy: ", acc)
print("F-score: ", f1)
plot_matrix(test_labels, predict_labels)