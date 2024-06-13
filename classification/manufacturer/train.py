import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders
from model import CNN
from pathlib import Path

data_root = Path("../../data/fgvc-aircraft-2013b")

trainloader, testloader, num_classes = get_dataloaders(data_root, batch_size=4, num_workers=2)
net = CNN(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training des Modells
for epoch in range(5):  # Beispiel: 5 Epochen
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:  # Alle 200 Mini-Batches ausgeben
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')
