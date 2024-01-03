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
        self.linear1 = nn.Linear(13, 100)
        self.linear2 = nn.Linear(100, 75)
        self.linear3 = nn.Linear(75, 50)
        self.linear4 = nn.Linear(50, 28)
        self.crossEntropyLoss = nn.CrossEntropyLoss()
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
        x= F.softmax(x, dim=-1)
        return x
    
    def trainOneBatch(self, attribs, labels, optimizer):
        pred = self(attribs)
        # print("before pred view: ", pred)
        # pred = pred.view(-1, pred.shape[-1])
        pred= pred.squeeze(1)
        # print("inside batch\npred: ", pred)
        # labels = labels.view(-1).long()
        # labels = labels.view(-1)
        # labels = labels.squeeze(-1)
        # labels = torch.nn.functional.one_hot(labels.squeeze(), num_classes=28, dtype=long)
        # print("inside batch\nlables: ", labels)
        loss = self.crossEntropyLoss(pred,labels)
        # print("completed loss\n\n\n\n\n\n")
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
        #batch size of 1 but still makes mounting the tensors really easy
        for batch, (attribs, labels) in enumerate(dataloader):
            pred = self(attribs)
            loss = (self.crossEntropyLoss(pred.squeeze(1),labels)).item()
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
        #batch size of 1 but still makes mounting the tensors really easy
        for batch, (attribs, labels) in enumerate(dataloader):
            pred = self(attribs)
            # pred_int = int(round(pred.item()))
            pred_int = torch.argmax(pred,dim=-1).item()
            # print("pred")
            # print("pred val: ", torch.argmax(pred,dim=-1).item())
            print(pred_int)
            pred_vals_individual.append(pred_int)
            actual_vals_individual.append(torch.argmax(labels,dim=-1).item())

            # pred_vals_freq[pred_int - 1 ] += 1
            # actual_vals_freq[int(round(labels.item())-1)] += 1

        # print("pred vals freq:   ", pred_vals_freq)
        # print("actual vals freq: ", actual_vals_freq)
        cm = confusion_matrix(actual_vals_individual, pred_vals_individual)
        print("cm: ", cm)
        ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=[i for i in range(1,cm.shape[0] + 1)]).plot(cmap='Blues')
        pyplot.show()


#data loader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputDf : pandas.DataFrame):
        # drop the label
        self.df = inputDf.drop(columns = 'finalRank')
        # print("self df head: ", self.df.head(5))
        # only use the label
        self.labels = inputDf[['finalRank']]
        # print("self df labels head: ", self.labels.head(5))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        raw = self.df.iloc[idx].values
        # print("raw: ", raw)
        # print("label: ", self.labels.iloc[idx].values)
        if type(idx) == int:
            raw = raw.reshape(1, -1)
        data = torch.tensor(raw[:].copy(), dtype=torch.float32)
        label = torch.tensor([1 if self.labels.iloc[idx].values == i else 0 for i in range(0,28)], dtype=torch.float32)
        # label = torch.tensor(self.labels.iloc[idx].values, dtype=torch.long)
        # print("returned")
        # print("data: ", data)
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


    epochs = 1
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
            # https://www.youtube.com/watch?v=S9LAxmDGQGw&ab_channel=TrendingSoundEffect
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

##########################################
#              Part 2 & 3                #
##########################################

#data loader
class CustomMnistDataset(torch.utils.data.Dataset):
    def __init__(self, attribs, labels):
        self.df = pandas.DataFrame(attribs)
        self.labels = labels

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        raw = self.df.iloc[idx].values
        if type(idx) == int:
            raw = raw.reshape(1, -1)
        data = torch.tensor(raw[:].copy(), dtype=torch.float32)
        return data, self.labels[idx]

##############################
#      ANTHONY EDIT THIS     #
##############################
def mountToTensorMnist(filename):
    # demeterParsed = read_mnist(filename)

    pca = PCA(0.9)
    train_data = read_mnist('mnist_train.csv')

    ###################################
    #  code from anthony, project 02  #
    ###################################
    train_data_ints = [[0, [0 for b in range(len(train_data))]] for a in range(len(train_data))]
    i = 0
    for example in train_data:
        label = int(example[0])
        data = [int(attribute) for attribute in example[1]]
        train_data_ints[i] = [label, data]
        i += 1
    train_data = train_data_ints
    for example in train_data:
        label = int(example[0])
        data = [int(attribute) for attribute in example[1]]
    train_scaler = StandardScaler()
    train_attributes = [row[1] for row in train_data]
    train_attributes_scaled = train_scaler.fit_transform(train_attributes)
    train_attributes_scaled_reduced = pca.fit_transform(train_attributes_scaled)
    for i in range(len(train_data)):
        train_data[i][1] = [train_attributes_scaled_reduced[i][a] for a in range(len(train_attributes_scaled_reduced[i]))]
    ###################################

    attributes = [sample[1] for sample in train_data]

    # one hot encoding
    labels= []
    for sample in train_data:
        labels.append(torch.tensor([1.0 if a == sample[0] else 0 for a in range(10)], dtype=torch.float32))
    # # value encoding
    # labels = [torch.tensor(sample[1], dtype=torch.float32) for sample in train_data]
    return attributes, labels

class MNIST(nn.Module):
    def __init__(self, inputDims):
        super(MNIST, self).__init__()
        self.linear1 = nn.Linear(inputDims, 50)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(50, 20)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(20, 10)
        self.crossEntropyLoss = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = F.softmax(x, dim=-1)
        return x

    def trainOneBatch(self, attribs, labels, optimizer):
        pred = self(attribs)
        loss = self.MSELoss(pred,labels.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def trainAll(self, dataloader, optimizer, respondEpoch = True) -> int:
        # TODO make this not bad
        self.train()
        train_loss = []
        for batch, (attribs, labels) in enumerate(dataloader):
            train_loss.append(self.trainOneBatch(attribs, labels, optimizer=optimizer))
        if respondEpoch:
            return sum(train_loss) / len(train_loss)
        else:
            return train_loss

    def testOneBatch(self, attribs, labels):
        """
        Found to be easier to look at the brains of the program
        if tested one batch at a time and could get the error after
        """
        with torch.no_grad():
            pred = self(attribs)
            loss = self.MSELoss(pred,labels.unsqueeze(1))
        return loss.item()
    
    def testAll(self, dataloader, respondEpoch = True):
        # TODO make this not bad
        self.eval()
        test_loss = []
        for batch, (attribs, labels) in enumerate(dataloader):
            test_loss.append(self.testOneBatch(attribs, labels))
        if respondEpoch:
            return sum(test_loss)/len(test_loss)
        else:
            return test_loss
    
    def makeConfMatrix(self, dataloader):
        confMatr = [[0 for a in range(0,10)] for a in range(0,10)]
        with torch.no_grad():
            for batch, (attribs, labels) in enumerate(dataloader):
                predictions = self(attribs)
                for i, sample_pred in enumerate(predictions):
                    predicted_label = torch.argmax(sample_pred)
                    label = torch.argmax(labels[i])
                    confMatr[label][predicted_label] += 1
        
        fig, ax = pyplot.subplots()
        pyplot.imshow(confMatr, cmap='Blues')
        # pyplot.colorbar()
        for row in range(len(confMatr)):
            for col in range(len(confMatr[row])):
                label = confMatr[row][col]
                ax.text(col, row, label, color='black', ha='center', va='center')
        ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
        ax.set_yticks([0,1,2,3,4,5,6,7,8,9])
        pyplot.ylabel('Correct Answer')
        pyplot.xlabel(f"Guessed answer")
        pyplot.title(f"10x10 Confusion Matrix for Mnist")
        pyplot.show()

def classify_mnist():
    #train
    train_attributes, train_labels = mountToTensorMnist('mnist_train.csv')
    train = CustomMnistDataset(train_attributes, train_labels)
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    #valid
    valid_attributes, valid_labels = mountToTensorMnist('mnist_valid.csv')
    valid = CustomMnistDataset(valid_attributes, valid_labels)
    valid_loader = DataLoader(valid, batch_size=64, shuffle=True)
    #test
    test_attributes, test_labels = mountToTensorMnist('mnist_test.csv')
    test = CustomMnistDataset(test_attributes, test_labels)
    test_loader = DataLoader(test, batch_size=64, shuffle=True)

    #uncomment any one of the lines and copy paste from the external file to see results
    # determineBestLearningRateMnist(test_loader, train_loader, valid_loader)
    # determineBestWeightDecayMnist(test_loader, train_loader, valid_loader)

    mnist_ff = MNIST(153)
    #using adam because some guy on medium recomended it
    optimizer = torch.optim.Adam(mnist_ff.parameters(), lr=0.001, weight_decay=0.00001)
    train_loss = []
    valid_loss = []
    test_loss = []
    epochs = 75
    epsilon = 1E-3
    for epoch in range(0,epochs):
        mnist_ff.train()
        train_loss.append(mnist_ff.trainAll(train_loader, optimizer=optimizer))
        mnist_ff.eval()
        valid_loss.append(mnist_ff.testAll(valid_loader))
        test_loss.append(mnist_ff.testAll(test_loader))
        if epoch == 0:
            continue
        #stop before we overfit
        if abs(valid_loss[epoch] - valid_loss[epoch-1]) < epsilon:
            print("manual break")
            break

    mnist_ff.makeConfMatrix(test_loader)
    pyplot.plot([i for i, val in enumerate(train_loss)],train_loss)
    pyplot.xlabel("Batch Number")
    pyplot.title(f"lr = {0.001}")
    pyplot.ylabel("Training Loss")
    pyplot.show()

    pyplot.plot([i for i, val in enumerate(test_loss)],test_loss)
    pyplot.xlabel("Batch Number")
    pyplot.title(f"lr = {0.001}")
    pyplot.ylabel("Test Loss")
    pyplot.show()

if __name__ == "__main__":
    print("F1 training")
    predict_races()
