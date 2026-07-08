# -*- coding: utf-8 -*-
"""
triagem_s3.py
-------------
Baixa capturas novas do bucket S3 do Blick em lote e ja roda o modelo
treinado pra SUGERIR uma classe pra cada uma, organizando os arquivos em
pastas por classe prevista. Um humano ainda precisa auditar antes de essas
fotos virarem dado de treino/validacao de verdade — o script so troca
"arrastar cada foto uma por uma" por "conferir e corrigir sugestoes".

FLUXO:
    S3 (fotos novas do Klar) -> download em lote -> classificacao automatica
    -> pastas por classe sugerida (ou "revisar_confianca_baixa")
    -> [humano confere/corrige] -> mover pra data/raw/klar/<classe>/

IMPORTANTE: fotos que vieram do S3 (producao) nao tem rotulo nenhum — a
classificacao aqui e uma SUGESTAO do proprio modelo, nunca um rotulo
confirmado. Sempre confira antes de mover pra data/raw/klar, especialmente
nas classes que ja sabemos ter ponto cego (ex.: reflexo de sol confundido
com Virus_do_Estriamento — ver conversa sobre as fotos de campo).

Requer credenciais AWS configuradas (aws configure) com permissao de
leitura no bucket.

Exemplo de uso:
    python3 triagem_s3.py --bucket blick-capturas --prefixo capturas/2026-07 \
        --destino ../../data/raw/klar_novo --limiar-confianca 0.75
"""

import os
import argparse

import boto3
import torch
from PIL import Image
from torchvision import transforms

from inferir import carregar_modelo, montar_transform, classificar_imagem, montar_mascara_status

EXTENSOES_VALIDAS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def baixar_do_s3(bucket, prefixo, destino_bruto):
    """Baixa todos os objetos de imagem sob o prefixo informado."""
    os.makedirs(destino_bruto, exist_ok=True)
    s3 = boto3.client("s3")
    paginador = s3.get_paginator("list_objects_v2")

    baixados = []
    for pagina in paginador.paginate(Bucket=bucket, Prefix=prefixo):
        for obj in pagina.get("Contents", []):
            chave = obj["Key"]
            if not chave.lower().endswith(EXTENSOES_VALIDAS):
                continue

            nome_local = chave.replace("/", "__")
            caminho_local = os.path.join(destino_bruto, nome_local)

            if not os.path.exists(caminho_local):
                s3.download_file(bucket, chave, caminho_local)

            baixados.append(caminho_local)

    return baixados


def classificar_e_organizar(caminhos, destino, checkpoint, metadados_path, limiar_confianca):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, meta = carregar_modelo(checkpoint, metadados_path, device)
    transform = montar_transform(meta)
    mascara_status = montar_mascara_status(meta, device)

    contagem_por_pasta = {}

    for caminho in caminhos:
        try:
            r = classificar_imagem(caminho, modelo, transform, meta, device, mascara_status)
        except Exception as e:
            print(f"[TRIAGEM] Erro ao processar {caminho}: {e}")
            continue

        if r["confianca_status"] >= limiar_confianca:
            pasta_destino = r["status_previsto"]  # saudavel / praga / doenca
        else:
            pasta_destino = "revisar_confianca_baixa"

        pasta_completa = os.path.join(destino, pasta_destino)
        os.makedirs(pasta_completa, exist_ok=True)

        destino_final = os.path.join(pasta_completa, os.path.basename(caminho))
        os.replace(caminho, destino_final)

        contagem_por_pasta[pasta_destino] = contagem_por_pasta.get(pasta_destino, 0) + 1

        print(f"{os.path.basename(caminho)} -> {pasta_destino} "
              f"(status: {r['status_previsto']} {r['confianca_status']*100:.0f}%, "
              f"subtipo sugerido: {r['subtipo_previsto']})")

    return contagem_por_pasta


def main():
    parser = argparse.ArgumentParser(description="Baixa e pré-classifica capturas novas do S3.")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefixo", default="", help="Prefixo/pasta dentro do bucket (ex.: capturas/2026-07).")
    parser.add_argument("--destino", required=True, help="Pasta local onde organizar os resultados.")
    parser.add_argument("--checkpoint", default="./checkpoints/modelo_melhor.pth")
    parser.add_argument("--metadados", default="./checkpoints/metadados.json")
    parser.add_argument("--limiar-confianca", type=float, default=0.75,
                         help="Abaixo disso, a foto vai pra 'revisar_confianca_baixa' em vez de auto-classificada.")
    args = parser.parse_args()

    destino_bruto = os.path.join(args.destino, "_baixadas_do_s3")
    print(f"[TRIAGEM] Baixando de s3://{args.bucket}/{args.prefixo} ...")
    caminhos = baixar_do_s3(args.bucket, args.prefixo, destino_bruto)
    print(f"[TRIAGEM] {len(caminhos)} imagem(ns) baixada(s).")

    if not caminhos:
        print("[TRIAGEM] Nada novo pra classificar.")
        return

    contagem = classificar_e_organizar(
        caminhos, args.destino, args.checkpoint, args.metadados, args.limiar_confianca
    )

    print("\n" + "=" * 60)
    print("RESUMO DA TRIAGEM AUTOMÁTICA")
    print("=" * 60)
    for pasta, n in sorted(contagem.items()):
        print(f"  {pasta:<25} {n} imagem(ns)")
    print("\nPróximo passo: confira as pastas (principalmente 'revisar_confianca_baixa',")
    print("mas dê uma olhada por amostragem nas outras também) antes de mover")
    print("qualquer coisa pra data/raw/klar/<classe correta>/.")


if __name__ == "__main__":
    main()