from pathlib import Path

from torchvision.datasets import FGVCAircraft

dataset = FGVCAircraft(Path("./data/input/"), download=True)
print(dataset)
