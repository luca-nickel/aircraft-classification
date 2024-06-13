from dataset import get_dataloaders
from model import CNN
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

def train_model(trainloader, num_classes, net, epochs):
    print("Start training")
    print("Number of classes: ", num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    losses = []
    for epoch in range(epochs):  # Beispiel: 5 Epochen
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:  # Alle 200 Mini-Batches ausgeben
                avg_loss = running_loss / 200
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, avg_loss))
                losses.append(avg_loss)
                ax.clear()
                ax.plot(losses)
                ax.set_title("Training Loss")
                ax.set_xlabel("Batch")
                ax.set_ylabel("Loss")
                plt.pause(0.001)
                running_loss = 0.0

    plt.ioff()  # Turn off interactive mode
    plt.show()
    print('Finished Training')
    torch.save(net.state_dict(), 'model.pth')

def evaluate_model(testloader, num_classes, net):
    print("Start evaluation")
    net.load_state_dict(torch.load('model.pth'))

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Genauigkeit des Netzwerks auf den Testbildern: %d %%' % (
        100 * correct / total))

if __name__ == "__main__":
    data_root = Path("../../data/fgvc-aircraft-2013b")
    trainloader, testloader, num_classes = get_dataloaders(data_root, batch_size=4, num_workers=2)
    net = CNN(num_classes=num_classes)
    epochs = 20

    train_model(trainloader, num_classes, net, epochs)
    evaluate_model(testloader, num_classes, net)
