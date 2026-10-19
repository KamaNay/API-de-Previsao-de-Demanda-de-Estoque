import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent

FILES = [
    (ROOT / "src" / "features.py", ROOT / "src" / "lambda_api" / "features.py"),
    (ROOT / "artifacts" / "model.pkl", ROOT / "src" / "lambda_api" / "model.pkl"),
]

for src, dest in FILES:
    shutil.copy(src, dest)
    print(f"{src.name} copiado para {dest.parent}")
