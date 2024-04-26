from pathlib import Path

from torchvision.datasets import FGVCAircraft

dataset = FGVCAircraft(Path("."), download=True)
print(dataset)
