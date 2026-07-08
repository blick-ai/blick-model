# -*- coding: utf-8 -*-
"""
avaliar_modelo.py
------------------
Roda o modelo treinado sobre o conjunto de validação e gera:
  1. Um relatório por classe (precisão, recall, f1, quantidade de exemplos)
  2. A matriz de confusão completa
  3. Um resumo direto: pra cada classe, qual é o erro mais comum do modelo
     (ex.: "Cercosporiose costuma ser confundida com Mancha_de_Turcicum")

MOTIVAÇÃO: a acurácia geral (ex.: "94% no subtipo") esconde um problema
comum em classificadores com muitas classes desbalanceadas — o modelo pode
estar "empurrando" as classes difíceis pra uma classe genérica (aqui,
Outra_Doenca ou Outra_Praga) em vez de realmente aprender a diferença
entre elas. Esse script existe pra detectar isso com números, não achismo.

LEITURA DO RESULTADO:
  - RECALL baixo numa classe = o modelo tem dificuldade de RECONHECER essa
    classe quando ela aparece de verdade (ex.: recall baixo em
    Cercosporiose = muitas fotos de cercosporiose sendo classificadas como
    outra coisa).
  - PRECISÃO baixa numa classe = o modelo está "chutando" essa classe com
    frequência quando na verdade era outra (ex.: precisão baixa em
    Outra_Doenca = o modelo está jogando coisas ali que não deveriam estar).
  - Se Outra_Doenca tiver RECALL alto mas PRECISÃO baixa, é sinal de que
    virou uma "gaveta de bagunça" pra onde o modelo empurra casos difíceis.

Exemplo de uso:
    python3 avaliar_modelo.py --manifest ../../data/manifest.csv
"""

import argparse
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from train import (
    carregar_e_dividir, BlickLeafDataset, montar_transforms,
    ClassificadorHierarquico, STATUS_LABELS,
)
import json


def carregar_modelo_e_meta(checkpoint_path, metadados_path, device):
    with open(metadados_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    modelo = ClassificadorHierarquico(
        n_subtipos=max(1, len(meta["classe_para_idx"])),
        backbone_nome=meta.get("backbone", "resnet18"),
        pretrained=False,
    )
    modelo.load_state_dict(torch.load(checkpoint_path, map_location=device))
    modelo.to(device)
    modelo.eval()
    return modelo, meta


def avaliar_confusao(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, meta = carregar_modelo_e_meta(args.checkpoint, args.metadados, device)

    _, val_df = carregar_e_dividir(args.manifest, val_fraction=args.val_fraction)

    classe_para_idx = {c: int(i) for c, i in meta["classe_para_idx"].items()}
    idx_para_classe = {int(i): c for i, c in meta["idx_para_classe"].items()}

    # classes na validacao que nao existiam no treino do checkpoint carregado
    # (pode acontecer se o manifesto mudou depois do treino) sao ignoradas,
    # com aviso, em vez de quebrar a avaliacao
    antes = len(val_df)
    val_df = val_df[val_df["classe"].isin(classe_para_idx.keys())].reset_index(drop=True)
    if len(val_df) < antes:
        print(f"[AVALIAR] {antes - len(val_df)} imagem(ns) da validação ignorada(s) "
              f"(classe não existia no treino deste checkpoint).")

    ds_val = BlickLeafDataset(val_df, classe_para_idx, montar_transforms(treino=False, img_size=meta["img_size"]))
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    verdadeiros, previstos = [], []
    with torch.no_grad():
        for imgs, _, y_subtipo in dl_val:
            imgs = imgs.to(device)
            _, saida_subtipo = modelo(imgs)
            previstos.extend(saida_subtipo.argmax(1).cpu().tolist())
            verdadeiros.extend(y_subtipo.tolist())

    nomes_classes = [idx_para_classe[i] for i in range(len(idx_para_classe))]

    print("\n" + "=" * 70)
    print("RELATÓRIO POR CLASSE (conjunto de validação)")
    print("=" * 70)
    print(classification_report(verdadeiros, previstos, labels=list(range(len(nomes_classes))),
                                 target_names=nomes_classes, zero_division=0))

    matriz = confusion_matrix(verdadeiros, previstos, labels=list(range(len(nomes_classes))))

    print("=" * 70)
    print("CONFUSÃO MAIS COMUM POR CLASSE")
    print("=" * 70)
    for i, nome in enumerate(nomes_classes):
        total_classe = matriz[i].sum()
        if total_classe == 0:
            continue
        acertos = matriz[i][i]
        erros = matriz[i].copy()
        erros[i] = 0
        if erros.sum() == 0:
            print(f"  {nome:<22} 100% acerto ({acertos}/{total_classe})")
            continue
        idx_erro_comum = erros.argmax()
        print(f"  {nome:<22} acerto {acertos}/{total_classe} ({acertos/total_classe*100:.0f}%) "
              f"— erro mais comum: confundido com '{nomes_classes[idx_erro_comum]}' "
              f"({erros[idx_erro_comum]}x)")

    # checagem direta do "viraram tudo Outra_Doenca/Outra_Praga?"
    for classe_generica in ("Outra_Doenca", "Outra_Praga"):
        if classe_generica not in classe_para_idx:
            continue
        idx_generica = classe_para_idx[classe_generica]
        previstos_como_generica = sum(1 for p in previstos if p == idx_generica)
        corretos_generica = sum(1 for v, p in zip(verdadeiros, previstos) if p == idx_generica and v == idx_generica)
        if previstos_como_generica > 0:
            precisao_generica = corretos_generica / previstos_como_generica
            print(f"\n[DIAGNÓSTICO] '{classe_generica}' foi prevista {previstos_como_generica}x, "
                  f"das quais {corretos_generica} realmente eram dessa classe "
                  f"(precisão {precisao_generica*100:.0f}%).")
            if precisao_generica < 0.5:
                print(f"  -> ALERTA: menos da metade das previsões de '{classe_generica}' estavam certas — "
                      f"sinal de que o modelo está empurrando classes difíceis pra essa gaveta genérica.")


def main():
    parser = argparse.ArgumentParser(description="Avalia o modelo com matriz de confusão por classe.")
    parser.add_argument("--manifest", default="../../data/manifest.csv")
    parser.add_argument("--checkpoint", default="./checkpoints/modelo_melhor.pth")
    parser.add_argument("--metadados", default="./checkpoints/metadados.json")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    avaliar_confusao(args)


if __name__ == "__main__":
    main()