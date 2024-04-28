from pathlib import Path

from torchvision.datasets import FGVCAircraft

dataset = FGVCAircraft(Path("./data"), download=True)
print(dataset)
