#######################################################
#  Copy paste the following code with the respective  #
#           function call in the main file            #
#######################################################

def determineBestLearningRate(test_loader, train_loader, valid_loader):
    epochs = 40
    epsilon = 1E-6
    learningRate = [1E0, 1E-1, 1E-2,  1E-3, 1E-4, 1E-5, 1E-6, 1E-7, 1E-8, 1E-9]
    #iterate over learningRate
    fig, ax = pyplot.subplots(5, 2)
    fig.tight_layout(pad=2.0)
    for index, lr in enumerate(learningRate):
        print("working on lr: ", lr)
        pyplot.subplot(5, 2, index + 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        f1_ff = F1FeedForward().to(device)
        #using adam because some guy on medium recomended it
        optimizer = torch.optim.Adam(f1_ff.parameters(), lr=lr)
        train_loss = []
        valid_loss = []
        test_loss = []
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
        pyplot.plot([i for i, val in enumerate(test_loss)],test_loss)
        pyplot.xlabel("Epoch Number")
        pyplot.title(f"lr = {lr}")
        pyplot.ylabel("Test Loss")
    pyplot.show()

def determineBestWeightDecay(test_loader, train_loader, valid_loader):
    epochs = 40
    epsilon = 1E-6
    WeightDecayVec = [1E0, 1E-1, 1E-2,  1E-3, 1E-4, 1E-5, 1E-6, 1E-7, 1E-8, 1E-9]

    fig, ax = pyplot.subplots(5, 2)
    fig.tight_layout(pad=2.0)
    for index, wd in enumerate(WeightDecayVec):
        print("working on wd: ", wd)
        pyplot.subplot(5, 2, index + 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        f1_ff = F1FeedForward().to(device)
        #using adam because some guy on medium recomended it
        optimizer = torch.optim.Adam(f1_ff.parameters(), lr=0.001, weight_decay=wd)
        train_loss = []
        valid_loss = []
        test_loss = []
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
        pyplot.plot([i for i, val in enumerate(test_loss)],test_loss)
        pyplot.xlabel("Epoch Number")
        pyplot.title(f"weight decay = {wd}")
        pyplot.ylabel("Test Loss")
    pyplot.show()

def determineBestWeightDecay(test_loader, train_loader, valid_loader):
    epochs = 40
    epsilon = 1E-6
    MomentumVec = [1E0, 1E-1, 1E-2,  1E-3, 1E-4, 1E-5, 1E-6, 1E-7, 1E-8, 1E-9]

    fig, ax = pyplot.subplots(5, 2)
    fig.tight_layout(pad=2.0)
    for index, mom in enumerate(MomentumVec):
        print("working on wd: ", wd)
        pyplot.subplot(5, 2, index + 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        f1_ff = F1FeedForward().to(device)
        #using adam because some guy on medium recomended it
        optimizer = torch.optim.Adam(f1_ff.parameters(), lr=0.001, weight_decay=0.001, momentum=mom)
        train_loss = []
        valid_loss = []
        test_loss = []
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
        pyplot.plot([i for i, val in enumerate(test_loss)],test_loss)
        pyplot.xlabel("Epoch Number")
        pyplot.title(f"momentum = {mom}")
        pyplot.ylabel("Test Loss")
    pyplot.show()