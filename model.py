import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

inputSize = 28
hiddenSize = 128
outputSize = 10
numEpochs = 5
batchSize = 64
learningRate = 0.001

trainDataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testDataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
trainLoader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(dataset=testDataset, batch_size=batchSize, shuffle=False)

class MyRNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(MyRNN, self).__init__()
        self.hiddenSize = hiddenSize
        self.rnn = nn.RNN(inputSize, hiddenSize, batch_first=True)
        self.fc = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hiddenSize).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


model = MyRNN(inputSize, hiddenSize, outputSize)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

totalStep = len(trainLoader)
for epoch in range(numEpochs):
    for i, (images, labels) in enumerate(trainLoader):
        images = images.view(-1, 28, 28)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 300 == 0:
            print(f'Epoch [{epoch + 1}/{numEpochs}], Step [{i + 1}/{totalStep}], Loss: {loss.item():.4f}')

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testLoader:
        images = images.view(-1, 28, 28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += int((predicted == labels).sum())

    print(f'테스트 데이터 {len(testDataset)}개의 전체 정확도: {100 * correct / total:.2f} %')
