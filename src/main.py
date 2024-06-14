from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from preprocessing.transforms_service import TransformsService
from classification.dataloader.dataset import ClassificationDataset
from classification.model.model import Model
from classification.model.network.network import Network
from classification.trainer import Trainer


def main():
    train_dataset = ClassificationDataset(
        transforms=TransformsService("default_classification_pipeline").transforms
    )
    test_dataset = ClassificationDataset(
        file="../data/input/fgvc-aircraft-2013b/data/images_manufacturer_test.txt",
        transforms=TransformsService("default_classification_pipeline_val").transforms,
    )

    net = Network(train_dataset[0][0].shape[0], len(train_dataset.classes))
    model = Model(
        net,
        CrossEntropyLoss(),
        Adam(net.parameters(), lr=0.0001, weight_decay=1e-4),
    )

    trainer = Trainer(model, [train_dataset, test_dataset], batch_size=128)

    trainer.train(50, inflation=5)


if __name__ == "__main__":
    main()
