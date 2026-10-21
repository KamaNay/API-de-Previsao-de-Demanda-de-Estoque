import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
BUILD = ROOT / "build"
SRC = ROOT / "src" / "lambda_api"
FUNCTION_FILES = ["handler.py", "db.py", "inference.py", "schemas.py", "features.py", "model.pkl"]


def run_sync():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "sync_shared_code.py")], check=True)


def clean(folder):
    for name in ("__pycache__", "tests", "test"):
        for path in list(folder.rglob(name)):
            if path.exists() and "numpy" not in path.parts:
                shutil.rmtree(path)
    for path in folder.rglob("*.pyi"):
        path.unlink()


def build_layer():
    shutil.rmtree(BUILD / "layer", ignore_errors=True)
    target = BUILD / "layer" / "python"
    target.mkdir(parents=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "-r", str(ROOT / "requirements-lambda.txt"),
        "--target", str(target),
        "--platform", "manylinux2014_x86_64",
        "--platform", "manylinux_2_28_x86_64",
        "--python-version", "3.12",
        "--only-binary", ":all:",
    ], check=True)
    clean(target)
    shutil.make_archive(str(BUILD / "layer"), "zip", root_dir=BUILD / "layer")


def build_function():
    # copia FUNCTION_FILES para uma pasta de staging e gera function.zip
    FILES = [SRC / f for f in FUNCTION_FILES]
    staging = BUILD / "function_staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for f in FILES:
        shutil.copy(f, staging)
        
    function_zip = BUILD / "function.zip"
    shutil.make_archive(str(function_zip.with_suffix("")), "zip", root_dir=staging)
    print(f"Function zip created at: {function_zip}")


def folder_size_mb(folder):
    total_size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    return total_size / (1024 * 1024)

def file_size_mb(file_path):
    return file_path.stat().st_size / (1024 * 1024)


LIMIT_MB = 250

if __name__ == "__main__":
    BUILD.mkdir(exist_ok=True)
    run_sync()
    build_layer()
    build_function()

    layer_mb = folder_size_mb(BUILD / "layer")
    function_mb = folder_size_mb(BUILD / "function_staging")
    total_mb = layer_mb + function_mb

    print(f"Layer (descompactado): {layer_mb:.1f} MB | zip: {file_size_mb(BUILD / 'layer.zip'):.1f} MB")
    print(f"Função (descompactada): {function_mb:.1f} MB | zip: {file_size_mb(BUILD / 'function.zip'):.1f} MB")
    print(f"Total descompactado: {total_mb:.1f} MB (limite {LIMIT_MB} MB)")

    if total_mb > LIMIT_MB:
        print("ERRO: pacote acima do limite da Lambda.")
        sys.exit(1)
