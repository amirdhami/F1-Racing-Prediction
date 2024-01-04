import math
import numpy as np
import pandas

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class F1FeedForward(nn.Module):
    def __init__(self):
        super(F1FeedForward, self).__init__()
        self.linear1 = nn.Linear(13, 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, 5)
        self.linear4 = nn.Linear(5, 1)
        self.MSELoss = nn.MSELoss()
        self.relu = nn.LeakyReLU()

    # simple basic feed forward form inputs to outputs
    def forward(self, x):
        # attribs: 12 -> 20
        x = self.linear1(x)
        x = self.relu(x)
        # attribs: 20 -> 10
        x = self.linear2(x)
        x = self.relu(x)
        # attribs: 10 -> 5
        x = self.linear3(x)
        x = self.relu(x)
        # attribs: 5 -> 1
        x = self.linear4(x)
        return x
    
    def trainOneBatch(self, attribs, labels, optimizer):
        pred = self(attribs)
        loss = self.MSELoss(pred,labels.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def trainAll(self, dataloader, optimizer, respondEpoch = True) -> int:
        self.train()
        train_loss = []
        for batch, (attribs, labels) in enumerate(dataloader):
            train_loss.append(self.trainOneBatch(attribs, labels, optimizer=optimizer))
            if batch % 50 == 0:
                print("working on batch number: ", batch, "/", len(dataloader))
                print("train loss rn:           ", train_loss[-1])
        if respondEpoch:
            return sum(train_loss) / len(train_loss)
        else:
            return train_loss
    
    def testAll(self, dataloader, respondEpoch = True):
        self.eval()
        test_loss = []
        for batch, (attribs, labels) in enumerate(dataloader):
            pred = self(attribs)
            loss = (self.MSELoss(pred,labels.unsqueeze(1))).item()
            test_loss.append(loss)
        if respondEpoch:
            return sum(test_loss)/len(test_loss)
        else:
            return test_loss

    def makeConfusionMatrix(self, dataloader):
        self.eval()
        pred_vals_individual = []
        pred_vals_freq = [0 for i in range(28)]
        actual_vals_individual = []
        actual_vals_freq = [0 for i in range(28)]
        for batch, (attribs, labels) in enumerate(dataloader):
            pred = self(attribs)
            pred_int = int(round(pred.item()))

            pred_vals_individual.append(pred_int)
            actual_vals_individual.append(int(labels.item()))

            pred_vals_freq[pred_int - 1 ] += 1
            actual_vals_freq[int(round(labels.item())-1)] += 1
        cm = confusion_matrix(actual_vals_individual, pred_vals_individual)
        print("cm: ", cm)
        ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=[i for i in range(1,cm.shape[0] + 1)]).plot(cmap='Blues')
        pyplot.show()


#data loader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputDf : pandas.DataFrame):
        # drop the label
        self.df = inputDf.drop(columns = 'finalRank')
        self.labels = inputDf[['finalRank']]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        raw = self.df.iloc[idx].values
        if type(idx) == int:
            raw = raw.reshape(1, -1)
        data = torch.tensor(raw[:].copy(), dtype=torch.float32)
        label = torch.tensor(self.labels.iloc[idx].values, dtype=torch.float32)
        return data, label

# become the best classifier for racers
def predict_races():
    #class
    # final rank - where you score on the podium


    #attribs
    # circuitName - circuitID
    # year
    # raceRound
    # ageAtRace
    # seasonAvgPlace
    # qualPos
    # pointsGoingIn
    # winsGoingIn
    # recentPlacement (placement from the last 5 rounds)
    # driverCircuitAvgPlace
    # teamCircuitAvgPlace
    # seasonOvertake
    # careerOvertake

    # incompatible
    # constructorName - constructorId
    # driverName - driverId
    f1data = pandas.read_csv('raw_data.csv')
    print('pre dropping cols: ',list(f1data.columns))
    f1data = f1data[['year', 'circuitId', 'raceRound', 'ageAtRace',
                    'seasonAvgPlace', 'qualPos', 'pointsGoingIn',
                    'winsGoingIn', 'recentPlacement', 'driverCircuitAvgPlace',
                    'teamCircuitAvgPlace', 'seasonOvertake', 'careerOvertake', 'finalRank']]
    print('post dropping cols',list(f1data.columns))
    print('max finalrank: ', f1data[['finalRank']].max('index'))

    f1_train , f1_test_valid = train_test_split(f1data, test_size = 0.2)
    f1_test, f1_valid = train_test_split(f1_test_valid, test_size = 0.5)
    print("train size: " ,len(f1_train.index))
    print("test size: " ,len(f1_test.index))
    print("valid size: " ,len(f1_valid.index))


    f1_train = DataLoader(CustomDataset(f1_train), batch_size=64, shuffle=True)
    f1_test = DataLoader(CustomDataset(f1_test), batch_size=1)
    f1_valid = DataLoader(CustomDataset(f1_valid), batch_size= 1)
    print("cleared data loading")


    epochs = 100
    LR = 1E-2
    epsilon = 5
    valid_loss = []
    test_loss = []
    train_loss =[]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f1_ff = F1FeedForward().to(device)
    optimizer = torch.optim.Adam(f1_ff.parameters(), lr=LR, weight_decay=1E-6)
    for t in range(epochs):
        print(f"Epoch {t}")
        # put it into training mode
        f1_ff.train()
        train_loss.append(f1_ff.trainAll(f1_train, optimizer))
        f1_ff.eval()
        valid_loss.append(f1_ff.testAll(f1_valid))
        print("valid loss: {:.3f}".format(valid_loss[-1]))
        test_loss.append(f1_ff.testAll(f1_test))
        print("test loss: {:.3f}".format(test_loss[-1]))
        if t < 5:
            continue
        if abs(valid_loss[t] - valid_loss[t-1]) < epsilon:
            print("broken at epoch: ", t)
            break
    f1_ff.makeConfusionMatrix(f1_test)
    print("train loss: ", train_loss)
    print("test loss: ", test_loss)
    print("valid loss: ", valid_loss)
    pyplot.plot([i for i, val in enumerate(test_loss)],test_loss)
    pyplot.xlabel("Epoch Number")
    pyplot.title(f"lr = {1E-6}")
    pyplot.ylabel("Test Loss")
    pyplot.show()
    pyplot.plot([i for i, val in enumerate(valid_loss)],valid_loss)
    pyplot.xlabel("Epoch Number")
    pyplot.title(f"lr = {1E-6}")
    pyplot.ylabel("Valid Loss")
    pyplot.show()
    print('Done!')

if __name__ == "__main__":
    print("F1 training")
    predict_races()
