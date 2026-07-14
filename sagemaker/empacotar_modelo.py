# -*- coding: utf-8 -*-
"""
empacotar_modelo.py
--------------------
Empacota o checkpoint treinado (modelo_melhor.pth + metadados.json) junto
com o inference.py num model.tar.gz no formato que o Amazon SageMaker
espera para hospedar como endpoint, usando o container gerenciado de
PyTorch (nao precisamos construir nem manter nossa propria imagem Docker).

ESTRUTURA GERADA DENTRO DO model.tar.gz:
    model.tar.gz
    ├── modelo_melhor.pth
    ├── metadados.json
    └── code/
        └── inference.py

Isso NAO faz o deploy sozinho — so gera o artefato que depois e enviado
pro S3 e referenciado na criacao do SageMaker Model/Endpoint (proximo
passo, quando vocês decidirem que o modelo esta maduro o suficiente pra
valer o custo de um endpoint rodando).

Exemplo de uso:
    python3 empacotar_modelo.py \
        --checkpoint ../training/checkpoints/modelo_melhor.pth \
        --metadados ../training/checkpoints/metadados.json \
        --inference ./inference.py \
        --saida ./model.tar.gz
"""

import os
import json
import shutil
import tarfile
import argparse


def empacotar(checkpoint, metadados, inference, saida):
    for caminho, nome in [(checkpoint, "checkpoint"), (metadados, "metadados"), (inference, "inference.py")]:
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"{nome} não encontrado em: {caminho}")

    # confere que os metadados sao um JSON valido antes de empacotar —
    # melhor falhar aqui do que descobrir so quando o endpoint subir
    with open(metadados, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[EMPACOTAR] Metadados OK — backbone: {meta.get('backbone')}, "
          f"img_size: {meta.get('img_size')}, subtipo_treinado: {meta.get('subtipo_treinado')}")

    pasta_tmp = "_pacote_tmp"
    if os.path.exists(pasta_tmp):
        shutil.rmtree(pasta_tmp)
    os.makedirs(os.path.join(pasta_tmp, "code"))

    shutil.copy(checkpoint, os.path.join(pasta_tmp, "modelo_melhor.pth"))
    shutil.copy(metadados, os.path.join(pasta_tmp, "metadados.json"))
    shutil.copy(inference, os.path.join(pasta_tmp, "code", "inference.py"))

    with tarfile.open(saida, "w:gz") as tar:
        for nome in os.listdir(pasta_tmp):
            tar.add(os.path.join(pasta_tmp, nome), arcname=nome)

    shutil.rmtree(pasta_tmp)

    tamanho_mb = os.path.getsize(saida) / (1024 * 1024)
    print(f"[EMPACOTAR] Pacote criado: {saida} ({tamanho_mb:.1f} MB)")
    print("[EMPACOTAR] Próximo passo (quando decidirem fazer o deploy de verdade):")
    print(f"  aws s3 cp {saida} s3://<seu-bucket>/blick/model.tar.gz")


def main():
    parser = argparse.ArgumentParser(description="Empacota o modelo treinado no formato do SageMaker.")
    parser.add_argument("--checkpoint", default="../training/checkpoints/modelo_melhor.pth")
    parser.add_argument("--metadados", default="../training/checkpoints/metadados.json")
    parser.add_argument("--inference", default="./inference.py")
    parser.add_argument("--saida", default="./model.tar.gz")
    args = parser.parse_args()

    empacotar(args.checkpoint, args.metadados, args.inference, args.saida)


if __name__ == "__main__":
    main()