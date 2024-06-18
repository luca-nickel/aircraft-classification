from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch import load
from torch.utils.data import DataLoader

from preprocessing.transforms_service import TransformsService
from classification.dataloader.dataset import ClassificationDataset
from classification.model.model import Model
from classification.model.network.network import Network
from classification.trainer import Trainer


def main():
    train_dataset = ClassificationDataset(
        file="../data/input/fgvc-aircraft-2013b/data/images_manufacturer_trainval.txt",
        transforms=TransformsService("default_classification_pipeline_val").transforms,
        train_transforms=TransformsService(
            "default_classification_pipeline_train_extra"
        ).transforms,
    )
    test_dataset = ClassificationDataset(
        file="../data/input/fgvc-aircraft-2013b/data/images_manufacturer_test.txt",
        transforms=TransformsService("default_classification_pipeline_val").transforms,
        classes=train_dataset.classes,
    )

    net = Network(train_dataset[0][0].shape[0], len(train_dataset.classes))
    state_dict = load("runs/2024-06-16 17_33/model_saves/_012_.pt")
    net.load_state_dict(state_dict)
    model = Model(
        net,
        CrossEntropyLoss(),
        SGD(net.parameters(), lr=0.0001, weight_decay=0.3, momentum=0.9),
    )

    # trainer = Trainer(model, [train_dataset, test_dataset], batch_size=64)

    # trainer.train(100, inflation=1)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Call the plot_accuracy function of the model
    model.plot_accuracy(test_loader, test_dataset.classes)


if __name__ == "__main__":
    main()
