"""
Empacota o modelo YOLO (pesos + script de inferencia) no formato que o
SageMaker espera: um model.tar.gz com os pesos na raiz e o codigo de
inferencia dentro de code/.

Uso:
    python3 empacotar_modelo.py --pesos weights.pt --saida model.tar.gz
"""

import argparse
import os
import shutil
import tarfile
import tempfile


def empacotar(caminho_pesos: str, caminho_saida: str) -> None:
    pasta_atual = os.path.dirname(os.path.abspath(__file__))

    with tempfile.TemporaryDirectory() as tmp:
        pasta_code = os.path.join(tmp, "code")
        os.makedirs(pasta_code)

        # os pesos vao na RAIZ do pacote, com nome fixo "best.pt" — o
        # inference.py (model_fn) espera exatamente esse nome
        shutil.copy(caminho_pesos, os.path.join(tmp, "best.pt"))

        # script de inferencia + o modulo de agregacao que ele importa +
        # as dependencias extras (ultralytics, opencv headless)
        shutil.copy(os.path.join(pasta_atual, "inference.py"), os.path.join(pasta_code, "inference.py"))
        shutil.copy(os.path.join(pasta_atual, "agregacao.py"), os.path.join(pasta_code, "agregacao.py"))
        shutil.copy(
            os.path.join(pasta_atual, "requirements.txt"),
            os.path.join(pasta_code, "requirements.txt"),
        )

        with tarfile.open(caminho_saida, "w:gz") as tar:
            tar.add(tmp, arcname=".")

    tamanho_mb = os.path.getsize(caminho_saida) / (1024 * 1024)
    print(f"Pacote criado: {caminho_saida} ({tamanho_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empacota o modelo YOLO pro SageMaker")
    parser.add_argument("--pesos", required=True, help="Caminho pro arquivo weights.pt (best.pt)")
    parser.add_argument("--saida", default="model.tar.gz", help="Nome do arquivo de saida")
    args = parser.parse_args()
    empacotar(args.pesos, args.saida)