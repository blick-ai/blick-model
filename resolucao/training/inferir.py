# -*- coding: utf-8 -*-
"""
inferir.py
----------
Roda o modelo ja treinado (checkpoints/modelo_melhor.pth) numa imagem ou
numa pasta inteira de imagens, e imprime a previsao de status_geral
(saudavel/praga/doenca) e do subtipo especifico, com as respectivas
confiancas.

USO PRINCIPAL PENSADO AGORA: triagem semi-automatica das fotos novas do
Klar que ainda nao tem rotulo confirmado. O modelo sugere um rotulo pra
cada foto; um humano confirma ou corrige antes de essas fotos virarem
dado de treino ou validacao de verdade — o modelo NUNCA deve validar a
si mesmo, ele so acelera a triagem manual.

Exemplos:
    # uma imagem so
    python3 inferir.py --imagem ../../foto_klar_01.jpg

    # uma pasta inteira, salvando um CSV pra revisar depois
    python3 inferir.py --pasta ../../fotos_klar_novas --saida-csv triagem.csv
"""

import os
import csv
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms

from train import ClassificadorHierarquico, STATUS_LABELS

EXTENSOES_VALIDAS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def carregar_modelo(checkpoint_path, metadados_path, device):
    with open(metadados_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    modelo = ClassificadorHierarquico(
        n_subtipos=max(1, len(meta["classe_para_idx"])),
        backbone_nome=meta.get("backbone", "resnet18"),
        pretrained=False,  # vamos carregar os pesos treinados, nao os do ImageNet
    )
    modelo.load_state_dict(torch.load(checkpoint_path, map_location=device))
    modelo.to(device)
    modelo.eval()
    return modelo, meta


def montar_transform(meta):
    img_size = meta["img_size"]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=meta["normalizacao_media"], std=meta["normalizacao_desvio"]),
    ])


def montar_mascara_status(meta, device):
    """
    Monta um vetor booleano por indice de subtipo, indicando se ele
    pertence a cada status geral. Usado pra restringir o argmax do
    subtipo ao grupo coerente com o status ja decidido.

    Se o metadados.json for de um treino antigo (sem "classe_para_status"),
    retorna None e a inferencia cai de volta pro comportamento antigo, sem
    a garantia de coerencia — funciona, so nao tem essa protecao.
    """
    if "classe_para_status" not in meta:
        return None

    idx_para_classe = meta["idx_para_classe"]
    classe_para_status = meta["classe_para_status"]
    n_subtipos = len(idx_para_classe)

    mascara = {}
    for status in STATUS_LABELS:
        vetor = torch.zeros(n_subtipos, dtype=torch.bool)
        for idx_str, classe in idx_para_classe.items():
            if classe_para_status.get(classe) == status:
                vetor[int(idx_str)] = True
        mascara[status] = vetor.to(device)
    return mascara


def classificar_imagem(caminho, modelo, transform, meta, device, mascara_status=None):
    img = Image.open(caminho).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        saida_status, saida_subtipo = modelo(x)
        prob_status = torch.softmax(saida_status, dim=1)[0]

    idx_status = int(prob_status.argmax().item())
    status_previsto = STATUS_LABELS[idx_status]

    resultado = {
        "caminho": caminho,
        "status_previsto": status_previsto,
        "confianca_status": round(prob_status[idx_status].item(), 4),
        "probabilidades_status": {
            status: round(prob_status[STATUS_LABELS.index(status)].item(), 4) for status in STATUS_LABELS
        },
    }

    # se o checkpoint foi treinado sem a cabeca de subtipo (--treinar-subtipo
    # nao usado), meta["idx_para_classe"] vem vazio — nao ha subtipo pra
    # reportar, e tudo bem, esse e o modo recomendado atualmente
    if not meta.get("subtipo_treinado", True) or not meta.get("idx_para_classe"):
        resultado["subtipo_previsto"] = None
        resultado["confianca_subtipo"] = None
        resultado["subtipo_sem_filtro"] = None
        return resultado

    prob_subtipo = torch.softmax(saida_subtipo, dim=1)[0]

    # subtipo "bruto": o que o modelo diria sem nenhuma restricao — pode
    # ser incoerente com o status_geral (ver bug do Ferrugem_Comum + praga)
    idx_subtipo_bruto = int(prob_subtipo.argmax().item())

    idx_subtipo = idx_subtipo_bruto
    if mascara_status is not None:
        vetor_mascara = mascara_status[status_previsto]
        if vetor_mascara.any():
            prob_subtipo_mascarada = prob_subtipo.masked_fill(~vetor_mascara, float("-inf"))
            idx_subtipo = int(prob_subtipo_mascarada.argmax().item())

    resultado["subtipo_previsto"] = meta["idx_para_classe"][str(idx_subtipo)]
    resultado["confianca_subtipo"] = round(prob_subtipo[idx_subtipo].item(), 4)
    resultado["subtipo_sem_filtro"] = meta["idx_para_classe"][str(idx_subtipo_bruto)]
    return resultado


def imprimir_resultado(r):
    print(f"\n{os.path.basename(r['caminho'])}")
    print(f"  status geral : {r['status_previsto']}  (confiança {r['confianca_status']*100:.1f}%)")
    if r["subtipo_previsto"] is not None:
        print(f"  subtipo      : {r['subtipo_previsto']}  (confiança {r['confianca_subtipo']*100:.1f}%)")
    detalhe = " | ".join(f"{status} {p*100:.1f}%" for status, p in r["probabilidades_status"].items())
    print(f"  detalhe      : {detalhe}")

    if r["subtipo_previsto"] is not None and r["subtipo_previsto"] != r["subtipo_sem_filtro"]:
        print(f"  (obs.: sem forçar coerência, o subtipo sugerido seria "
              f"'{r['subtipo_sem_filtro']}' — incoerente com o status geral, corrigido)")

    # confianca baixa merece atencao redobrada na revisao manual
    if r["confianca_status"] < 0.6:
        print("  ATENÇÃO: confiança baixa — revisar esta imagem manualmente com mais cuidado.")


def main():
    parser = argparse.ArgumentParser(description="Classifica imagem(ns) com o modelo já treinado do Blick.")
    parser.add_argument("--checkpoint", default="./checkpoints/modelo_melhor.pth")
    parser.add_argument("--metadados", default="./checkpoints/metadados.json")
    parser.add_argument("--imagem", default=None, help="Caminho de uma única imagem.")
    parser.add_argument("--pasta", default=None, help="Caminho de uma pasta com várias imagens.")
    parser.add_argument("--saida-csv", default=None, help="Se informado, salva os resultados em CSV (útil no modo --pasta).")
    args = parser.parse_args()

    if not args.imagem and not args.pasta:
        parser.error("Informe --imagem ou --pasta.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, meta = carregar_modelo(args.checkpoint, args.metadados, device)
    mascara_status = montar_mascara_status(meta, device)
    if mascara_status is None:
        print("[INFERIR] AVISO: metadados sem 'classe_para_status' (checkpoint antigo) — "
              "subtipo pode ficar incoerente com o status geral. Retreine ou reprocesse "
              "os metadados pra ativar a correção de coerência.")
    transform = montar_transform(meta)

    if args.imagem:
        caminhos = [args.imagem]
    else:
        caminhos = [
            os.path.join(args.pasta, nome)
            for nome in sorted(os.listdir(args.pasta))
            if nome.endswith(EXTENSOES_VALIDAS)
        ]
        print(f"[INFERIR] {len(caminhos)} imagem(ns) encontrada(s) em {args.pasta}.")

    resultados = []
    for caminho in caminhos:
        try:
            r = classificar_imagem(caminho, modelo, transform, meta, device, mascara_status)
        except Exception as e:
            print(f"\n{os.path.basename(caminho)}: erro ao processar ({e})")
            continue
        imprimir_resultado(r)
        resultados.append(r)

    if args.saida_csv and resultados:
        with open(args.saida_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
            writer.writeheader()
            writer.writerows(resultados)
        print(f"\n[INFERIR] Resultados salvos em {args.saida_csv} — revise antes de usar como rótulo definitivo.")


if __name__ == "__main__":
    main()