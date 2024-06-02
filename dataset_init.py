from pathlib import Path

from torchvision.datasets import FGVCAircraft

dataset = FGVCAircraft(Path("./data/input/"), download=False)
print(dataset)
