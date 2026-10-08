import shutil
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "features.py"
DEST = Path(__file__).parent.parent / "src" / "lambda_api" / "features.py"

shutil.copy(SRC, DEST)  # qual função do módulo shutil copia um arquivo?
print(f"{SRC.name} copiado para {DEST.parent}")