"""
Empacota os DOIS modelos do pipeline novo (segmentacao milho/nao-milho +
classificacao saudavel/nao_saudavel) junto com o script de inferencia, no
formato que o SageMaker espera: um model.tar.gz com os pesos na raiz e o
codigo de inferencia dentro de code/.

Uso:
    python3 empacotar_modelo.py \
        --pesos-segmentacao pesos_Yolo26.pt \
        --pesos-classificacao best.pt \
        --saida model.tar.gz
"""

import argparse
import os
import shutil
import tarfile
import tempfile


def empacotar(caminho_pesos_segmentacao: str, caminho_pesos_classificacao: str, caminho_saida: str) -> None:
    pasta_atual = os.path.dirname(os.path.abspath(__file__))

    with tempfile.TemporaryDirectory() as tmp:
        pasta_code = os.path.join(tmp, "code")
        os.makedirs(pasta_code)

        # os dois modelos vao na RAIZ do pacote, com nomes fixos que o
        # inference.py (model_fn) espera:
        #   corn_seg.pt -> estagio 1, segmentacao milho/nao-milho
        #   best.pt     -> estagio 3, classificacao saudavel/nao_saudavel,
        #                  roda em cima dos crops isolados do estagio 2
        shutil.copy(caminho_pesos_segmentacao, os.path.join(tmp, "corn_seg.pt"))
        shutil.copy(caminho_pesos_classificacao, os.path.join(tmp, "best.pt"))

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
    parser = argparse.ArgumentParser(
        description="Empacota os 2 modelos do pipeline (segmentacao + classificacao) pro SageMaker"
    )
    parser.add_argument(
        "--pesos-segmentacao", required=True,
        help="Caminho pro .pt do modelo milho/nao-milho (segmentacao, estagio 1 — ex.: pesos_Yolo26.pt)",
    )
    parser.add_argument(
        "--pesos-classificacao", required=True,
        help="Caminho pro .pt do modelo saudavel/nao_saudavel (classificacao, estagio 3 — ex.: best.pt)",
    )
    parser.add_argument("--saida", default="model.tar.gz", help="Nome do arquivo de saida")
    args = parser.parse_args()
    empacotar(args.pesos_segmentacao, args.pesos_classificacao, args.saida)