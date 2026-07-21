# -*- coding: utf-8 -*-
"""
train.py
--------
Treina o classificador fitossanitario do Blick a partir do manifesto
gerado pelo `data/prepare_dataset.py`.

ARQUITETURA: um unico backbone (ResNet34 pre-treinado no ImageNet) com
DUAS CABECAS de saida:
  1. status_geral  -> saudavel / praga / doenca         (robusta, bem populada)
  2. subtipo       -> a classe especifica (Cercosporiose, Ferrugem_Comum, ...)

Por que hierarquico e nao um classificador unico de ~20 classes: as classes
de subtipo tem volume bem desigual (de 113 a 6387 imagens — ver relatorio
de auditoria do prepare_dataset.py), entao a cabeca de subtipo vai errar
mais nas classes raras. Com a cabeca de status_geral treinada em paralelo
sobre grupos bem mais populosos, o sistema ainda da um resultado confiavel
("tem doenca") mesmo quando erra o subtipo exato — o que ja e util pro
dashboard (alerta + recomendacao generica), com o subtipo refinando a
recomendacao quando a confianca permitir.

MITIGACAO DO VIES DE FUNDO: a auditoria do prepare_dataset.py mostrou que
a maioria das classes de doenca/praga tem 80-98% de imagens com fundo de
estudio na BORDA, contra 20-30% nas imagens saudaveis — um atalho que o
modelo poderia aprender em vez da lesao em si. A mitigacao usada aqui e
fazer o RandomResizedCrop cortar bem fundo na imagem (scale minimo 0.4),
o que na maioria das vezes remove justamente a faixa de borda onde o fundo
de estudio aparece.

AVISO IMPORTANTE SOBRE VALIDACAO: por enquanto, o manifesto so tem imagens
de datasets de internet/professor (fundo de estudio), sem fotos de campo
reais do Klar. Isso significa que a acuracia de validacao reportada aqui
NAO garante desempenho em campo — ela mede se o modelo generaliza dentro do
mesmo estilo fotografico, nao para o estilo real de captura do rover. Assim
que houver fotos rotuladas do Klar (ou do PlantDoc), adicione a origem
delas em ORIGENS_VALIDACAO_CAMPO abaixo para que sirvam de validacao real.

Exemplo de uso:
    python3 train.py --manifest ../data/manifest.csv --epochs 15
"""

import os
import json
import time
import argparse
from collections import Counter

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
# Tamanho de entrada e backbone tem default leve (ResNet18, 160px) pensado
# para rodar em CPU numa maquina local. Se um dia o treino migrar pra GPU
# (Colab/SageMaker), use --backbone resnet34 --img-size 224 pra ter mais
# capacidade — a arquitetura aceita ambos sem mudar nada mais no codigo.
IMG_SIZE_PADRAO = 160

# Origens que representam fotos de campo reais (nao fundo de estudio).
# Se alguma origem COMECAR com um destes prefixos (case-insensitive), TODAS
# as imagens dela vao pra validacao (nunca pro treino) — sao poucas e
# preciosas, medem o que realmente importa: desempenho no estilo de
# captura do Klar. Prefixo em vez de nome exato de proposito: evita quebrar
# se um lote novo for organizado como "klar_novo", "klar_lote2", etc.
PREFIXOS_VALIDACAO_CAMPO = ("klar", "plantdoc")

STATUS_LABELS = ["saudavel", "praga", "doenca", "nao_milho"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BlickLeafDataset(Dataset):
    def __init__(self, df, classe_para_idx, transform):
        self.df = df.reset_index(drop=True)
        self.classe_para_idx = classe_para_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        linha = self.df.iloc[idx]
        img = Image.open(linha["caminho_processado"]).convert("RGB")
        img = self.transform(img)

        # -1 quando a classe so existe na validacao de campo, sem indice de
        # treino (ex.: Outra_Praga vindo so do Klar) — a imagem ainda entra
        # na avaliacao de status_geral, so fica de fora da metrica de subtipo
        idx_subtipo = self.classe_para_idx.get(linha["classe"], -1)
        idx_status = STATUS_LABELS.index(linha["status_geral"])
        return img, idx_status, idx_subtipo


def montar_transforms(treino: bool, img_size: int):
    if treino:
        # RandomResizedCrop com scale minimo agressivo (0.4) e a principal
        # mitigacao contra o vies de fundo de estudio detectado na auditoria:
        # na maior parte das vezes, corta a faixa de borda onde o fundo
        # aparece, forcando o modelo a olhar pra textura da lesao no centro.
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ---------------------------------------------------------------------------
# Modelo: backbone compartilhado + duas cabecas
# ---------------------------------------------------------------------------
class ClassificadorHierarquico(nn.Module):
    def __init__(self, n_subtipos, backbone_nome="resnet18", pretrained=True):
        super().__init__()

        if backbone_nome == "resnet18":
            pesos = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=pesos)
        elif backbone_nome == "resnet34":
            pesos = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=pesos)
        else:
            raise ValueError(f"backbone desconhecido: {backbone_nome!r} (use 'resnet18' ou 'resnet34')")

        n_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # remove a cabeca original, vira so extrator de features
        self.backbone = backbone

        self.cabeca_status = nn.Linear(n_features, len(STATUS_LABELS))
        self.cabeca_subtipo = nn.Linear(n_features, n_subtipos)

    def forward(self, x):
        features = self.backbone(x)
        return self.cabeca_status(features), self.cabeca_subtipo(features)


# ---------------------------------------------------------------------------
# Preparacao dos dados
# ---------------------------------------------------------------------------
FRACAO_TREINO_CAMPO_PADRAO = 0.7  # resto vai pra validacao
LIMIAR_CLASSE_MINIMA_CAMPO = 2    # classes de campo com ate esse tanto de exemplo vao INTEIRAS pro treino


def dividir_campo_treino_validacao(df_campo, fracao_treino, seed):
    """
    Divide as imagens de campo entre treino e validacao, em vez da regra
    antiga (100% pra validacao). Reservar tudo pra validacao parecia mais
    "seguro" pra medir desempenho real, mas na pratica significa que o
    modelo NUNCA ve um exemplo de campo durante o treino — inclusive
    "Saudavel" em fundo de mato, que e exatamente o caso que ele mais
    erra (ver conversa sobre as fotos do FOTOS.zip saindo como
    doenca/praga quando pareciam saudaveis a olho nu).

    Classes com ate LIMIAR_CLASSE_MINIMA_CAMPO exemplos vao INTEIRAS pro
    treino: com tao pouco dado, uma validacao de 1-2 imagem(ns) tem poder
    estatistico proximo de zero mesmo (foi o que aconteceu com Saudavel,
    n=2 — a divisao 70/30 mandou so 1 pro treino), entao o exemplo vale
    mais como sinal de treino do que como medida de validacao.
    """
    from sklearn.model_selection import train_test_split

    partes_treino, partes_val = [], []
    for _, grupo in df_campo.groupby("classe"):
        if len(grupo) <= LIMIAR_CLASSE_MINIMA_CAMPO or fracao_treino >= 1:
            partes_treino.append(grupo)
            continue
        if fracao_treino <= 0:
            partes_val.append(grupo)
            continue

        grupo_treino, grupo_val = train_test_split(
            grupo, train_size=fracao_treino, random_state=seed
        )
        if len(grupo_treino) == 0:
            grupo_treino, grupo_val = grupo.iloc[:1], grupo.iloc[1:]
        elif len(grupo_val) == 0:
            grupo_treino, grupo_val = grupo.iloc[:-1], grupo.iloc[-1:]

        partes_treino.append(grupo_treino)
        partes_val.append(grupo_val)

    treino_campo = pd.concat(partes_treino) if partes_treino else df_campo.iloc[0:0]
    val_campo = pd.concat(partes_val) if partes_val else df_campo.iloc[0:0]
    return treino_campo, val_campo


def carregar_e_dividir(manifest_path, val_fraction=0.15, seed=42, limit=None,
                        fracao_treino_campo=FRACAO_TREINO_CAMPO_PADRAO):
    df = pd.read_csv(manifest_path)

    antes = len(df)
    df = df[df["status_geral"] != "revisar"]
    df = df[df["caminho_processado"].notna()]
    print(f"[TREINO] {antes - len(df)} imagem(ns) excluida(s) (status 'revisar' ou sem imagem processada).")

    if limit:
        n_classes = df["classe"].nunique()
        por_classe = max(2, limit // n_classes)
        partes = [
            grupo.sample(min(len(grupo), por_classe), random_state=seed)
            for _, grupo in df.groupby("classe")
        ]
        df = pd.concat(partes, ignore_index=True)
        print(f"[TREINO] --limit ativo: usando amostra reduzida de {len(df)} imagem(ns) (smoke test).")

    print(f"[TREINO] {len(df)} imagem(ns) disponiveis para treino/validacao.")

    tem_campo = df["origem_dataset"].str.lower().str.startswith(PREFIXOS_VALIDACAO_CAMPO)
    n_campo = tem_campo.sum()

    if n_campo > 0:
        df_campo = df[tem_campo]
        df_resto = df[~tem_campo]

        treino_campo, val_campo = dividir_campo_treino_validacao(df_campo, fracao_treino_campo, seed)
        print(f"[TREINO] {n_campo} imagem(ns) de origem de campo: "
              f"{len(treino_campo)} para treino, {len(val_campo)} para validação "
              f"(fração configurável via --fracao-treino-campo, padrão {fracao_treino_campo:.0%}).")

        # o restante (internet/prof_wanderson) continua 100% no treino como
        # antes — o campo e que precisava de ajuste, nao essa parte
        treino_df = pd.concat([df_resto, treino_campo])
        val_df = val_campo
    else:
        print("\n" + "!" * 70)
        print("AVISO: nenhuma origem de campo (Klar/PlantDoc) encontrada no manifesto.")
        print("A validacao abaixo vai medir generalizacao DENTRO do mesmo estilo")
        print("fotografico dos datasets da internet/professor, NAO desempenho real")
        print("em campo. Trate a acuracia de validacao com cautela ate haver fotos")
        print("de campo rotuladas para validar de verdade.")
        print("!" * 70 + "\n")

        from sklearn.model_selection import train_test_split
        treino_df, val_df = train_test_split(
            df, test_size=val_fraction, random_state=seed, stratify=df["classe"]
        )

    return treino_df.reset_index(drop=True), val_df.reset_index(drop=True)


def calcular_pesos_classe(labels, n_classes):
    """Peso inversamente proporcional a frequencia, para compensar desbalanceamento."""
    contagem = Counter(labels)
    pesos = torch.zeros(n_classes)
    total = len(labels)
    for idx in range(n_classes):
        n = contagem.get(idx, 0)
        pesos[idx] = total / (n_classes * n) if n > 0 else 0.0
    return pesos


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------
def treinar(args):
    # fixa a semente aleatoria ANTES de qualquer coisa (inicializacao das
    # cabecas do modelo, embaralhamento do DataLoader) — sem isso, dois
    # treinos com o MESMO codigo e MESMO dado ainda davam resultados
    # diferentes so por causa da aleatoriedade da inicializacao dos
    # pesos, o que tornava impossivel saber se uma mudanca de codigo
    # realmente ajudou ou so foi sorte. Ver conversa sobre 2fc1ce60 e
    # 20260704_103223 "trocando de lado" entre rodadas sem mudanca de codigo.
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TREINO] Usando dispositivo: {device}")

    treino_df, val_df = carregar_e_dividir(
        args.manifest, val_fraction=args.val_fraction, limit=args.limit,
        fracao_treino_campo=args.fracao_treino_campo, seed=args.seed,
    )

    if args.treinar_subtipo:
        classes_ordenadas = sorted(treino_df["classe"].unique())
        classe_para_idx = {c: i for i, c in enumerate(classes_ordenadas)}
        idx_para_classe = {i: c for c, i in classe_para_idx.items()}

        # classes que so aparecem na validacao de campo (nao no treino) nao tem
        # indice de subtipo — mas continuam validas pra avaliar status_geral
        # (ver BlickLeafDataset.__getitem__, que usa -1 como sentinela nesse caso)
        faltantes = set(val_df["classe"].unique()) - set(classe_para_idx.keys())
        if faltantes:
            n_afetadas = val_df["classe"].isin(faltantes).sum()
            print(f"[TREINO] AVISO: {n_afetadas} imagem(ns) de classe(s) só na validação "
                  f"(sem exemplo no treino: {faltantes}) — mantidas na avaliação de "
                  f"status_geral, mas sem métrica de subtipo (nunca viram treino pra isso).")
    else:
        # subtipo desativado: todas as imagens recebem o sentinela -1 (ver
        # BlickLeafDataset.__getitem__), entao a metrica/perda de subtipo
        # simplesmente nao roda. O modelo treina so pro essencial: saudavel
        # / praga / doenca. Reative com --treinar-subtipo quando houver
        # dado suficiente por classe especifica pra isso valer a pena.
        print("[TREINO] Cabeça de subtipo DESATIVADA nesta rodada — treinando só "
              "saudável/praga/doença (use --treinar-subtipo pra reativar).")
        classes_ordenadas = []
        classe_para_idx = {}
        idx_para_classe = {}

    ds_treino = BlickLeafDataset(treino_df, classe_para_idx, montar_transforms(treino=True, img_size=args.img_size))
    ds_val = BlickLeafDataset(val_df, classe_para_idx, montar_transforms(treino=False, img_size=args.img_size))

    gerador = torch.Generator()
    gerador.manual_seed(args.seed)

    dl_treino = DataLoader(ds_treino, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True, generator=gerador)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True)

    labels_status_treino = [STATUS_LABELS.index(s) for s in treino_df["status_geral"]]
    pesos_status = calcular_pesos_classe(labels_status_treino, len(STATUS_LABELS)).to(device)

    if args.treinar_subtipo:
        labels_subtipo_treino = [classe_para_idx[c] for c in treino_df["classe"]]
        pesos_subtipo = calcular_pesos_classe(labels_subtipo_treino, len(classes_ordenadas)).to(device)
        criterio_subtipo = nn.CrossEntropyLoss(weight=pesos_subtipo)
    else:
        criterio_subtipo = None

    # com subtipo desativado, n_subtipos=1 e so um espaco reservado — essa
    # saida nunca e usada de verdade (loss e accuracy ficam de fora)
    n_subtipos_modelo = len(classes_ordenadas) if args.treinar_subtipo else 1
    modelo = ClassificadorHierarquico(n_subtipos=n_subtipos_modelo, backbone_nome=args.backbone).to(device)

    criterio_status = nn.CrossEntropyLoss(weight=pesos_status)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=args.epochs)

    melhor_acc_status = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    n_lotes = len(dl_treino)

    for epoca in range(1, args.epochs + 1):
        modelo.train()
        perda_total = 0.0
        tempo_inicio_epoca = time.time()

        for i, (imgs, y_status, y_subtipo) in enumerate(dl_treino, 1):
            imgs, y_status, y_subtipo = imgs.to(device), y_status.to(device), y_subtipo.to(device)

            otimizador.zero_grad()
            saida_status, saida_subtipo = modelo(imgs)

            # peso maior na cabeca de status: e a saida mais robusta e a
            # que o dashboard usa primeiro (alerta geral). Subtipo so entra
            # na conta quando --treinar-subtipo estiver ativo — senao,
            # y_subtipo e so o sentinela -1 e nem daria pra calcular a perda.
            perda = criterio_status(saida_status, y_status)
            if criterio_subtipo is not None:
                mascara_valida = y_subtipo >= 0
                if mascara_valida.any():
                    perda = perda + 0.5 * criterio_subtipo(saida_subtipo[mascara_valida], y_subtipo[mascara_valida])

            perda.backward()
            otimizador.step()
            perda_total += perda.item() * imgs.size(0)

            if i % 10 == 0 or i == n_lotes:
                decorrido = time.time() - tempo_inicio_epoca
                seg_por_lote = decorrido / i
                restante = seg_por_lote * (n_lotes - i)
                print(f"  [TREINO] lote {i}/{n_lotes} — perda: {perda.item():.4f} — "
                      f"~{restante/60:.1f} min restantes nesta época", end="\r")

        print()  # quebra a linha do \r antes do resumo da epoca
        scheduler.step()
        perda_media = perda_total / len(ds_treino)

        acc_status, acc_subtipo = avaliar(modelo, dl_val, device)
        texto_subtipo = f"{acc_subtipo*100:.1f}%" if args.treinar_subtipo else "desativado"
        print(f"[TREINO] Epoca {epoca}/{args.epochs} — perda: {perda_media:.4f} | "
              f"val status_geral: {acc_status*100:.1f}% | val subtipo: {texto_subtipo}")

        if acc_status > melhor_acc_status:
            melhor_acc_status = acc_status
            caminho_checkpoint = os.path.join(args.output_dir, "modelo_melhor.pth")
            torch.save(modelo.state_dict(), caminho_checkpoint)
            print(f"  [TREINO] Novo melhor modelo salvo em {caminho_checkpoint}")

    # mapeamento classe -> status geral, derivado direto do manifesto (nao
    # depende do modelo). Usado na inferencia pra FORCAR coerencia entre as
    # duas cabecas: o subtipo sugerido so pode vir do grupo que bate com o
    # status geral ja decidido (ver bug de "praga" + subtipo "Ferrugem_Comum"
    # discutido na conversa da triagem do S3). So faz sentido com subtipo ativo.
    if args.treinar_subtipo:
        classe_para_status = (
            pd.concat([treino_df, val_df])[["classe", "status_geral"]]
            .drop_duplicates()
            .set_index("classe")["status_geral"]
            .to_dict()
        )
    else:
        classe_para_status = {}

    # metadados necessarios para reconstruir o modelo no inference.py do SageMaker
    metadados = {
        "subtipo_treinado": args.treinar_subtipo,
        "status_labels": STATUS_LABELS,
        "classe_para_idx": classe_para_idx,
        "idx_para_classe": idx_para_classe,
        "classe_para_status": classe_para_status,
        "img_size": args.img_size,
        "backbone": args.backbone,
        "normalizacao_media": [0.485, 0.456, 0.406],
        "normalizacao_desvio": [0.229, 0.224, 0.225],
    }
    with open(os.path.join(args.output_dir, "metadados.json"), "w", encoding="utf-8") as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)

    # recarrega o MELHOR checkpoint (nao o da ultima epoca) pra listar
    # exatamente onde ele erra na validacao — fecha o ciclo sem precisar
    # rodar inferir.py numa pasta separada toda vez que quiser saber quais
    # fotos especificas o modelo confunde
    caminho_melhor = os.path.join(args.output_dir, "modelo_melhor.pth")
    if os.path.exists(caminho_melhor):
        modelo.load_state_dict(torch.load(caminho_melhor, map_location=device))
        modelo.eval()
        transform_val = montar_transforms(treino=False, img_size=args.img_size)

        erros = []
        with torch.no_grad():
            for _, linha in val_df.iterrows():
                img = Image.open(linha["caminho_processado"]).convert("RGB")
                x = transform_val(img).unsqueeze(0).to(device)
                saida_status, _ = modelo(x)
                previsto = STATUS_LABELS[saida_status.argmax(1).item()]
                real = linha["status_geral"]
                if previsto != real:
                    erros.append((os.path.basename(linha["caminho_processado"]), real, previsto, linha["origem_dataset"]))

        if erros:
            print(f"\n[TREINO] {len(erros)} imagem(ns) de validação onde o modelo (melhor checkpoint) errou:")
            for nome, real, previsto, origem in erros:
                print(f"  [{origem}] {nome} — real: {real}, previsto: {previsto}")
        else:
            print("\n[TREINO] Nenhum erro na validação com o melhor checkpoint — cuidado, "
                  "pode ser amostra pequena demais pra tirar conclusão.")

    print(f"\n[TREINO] Concluido. Melhor acurácia de status_geral na validação: {melhor_acc_status*100:.1f}%")
    print(f"[TREINO] Metadados salvos em {args.output_dir}/metadados.json")


def avaliar(modelo, dl_val, device):
    modelo.eval()
    acertos_status = acertos_subtipo = total = total_com_subtipo = 0

    with torch.no_grad():
        for imgs, y_status, y_subtipo in dl_val:
            imgs, y_status, y_subtipo = imgs.to(device), y_status.to(device), y_subtipo.to(device)
            saida_status, saida_subtipo = modelo(imgs)

            acertos_status += (saida_status.argmax(1) == y_status).sum().item()
            total += imgs.size(0)

            # -1 = classe sem indice de treino (ver BlickLeafDataset) — nao
            # entra na metrica de subtipo, mas ja contou acima pro status
            mascara_valida = y_subtipo >= 0
            if mascara_valida.any():
                acertos_subtipo += (saida_subtipo.argmax(1)[mascara_valida] == y_subtipo[mascara_valida]).sum().item()
                total_com_subtipo += mascara_valida.sum().item()

    if total == 0:
        return 0.0, 0.0
    acc_subtipo = acertos_subtipo / total_com_subtipo if total_com_subtipo > 0 else 0.0
    return acertos_status / total, acc_subtipo


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Treina o classificador hierárquico do Blick.")
    parser.add_argument("--manifest", default="../data/manifest.csv")
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet18",
                         help="resnet18 (padrão, mais leve/rápido em CPU) ou resnet34 (mais capacidade, use com GPU).")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE_PADRAO,
                         help=f"Tamanho da imagem de entrada (padrão: {IMG_SIZE_PADRAO}px, mais leve em CPU).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Usa uma amostra reduzida (N imagens no total, estratificada por classe) — "
                              "para testar rápido em CPU antes do treino completo.")
    parser.add_argument("--treinar-subtipo", action="store_true",
                         help="Ativa a cabeça de subtipo (doença/praga específica). Desativado por "
                              "padrão — o modelo treina só saudável/praga/doença, mais robusto com "
                              "o volume de dado por classe específica que temos hoje.")
    parser.add_argument("--fracao-treino-campo", type=float, default=FRACAO_TREINO_CAMPO_PADRAO,
                         help=f"Fração das fotos de campo (Klar/PlantDoc) usada no treino, o resto "
                              f"vai pra validação (padrão: {FRACAO_TREINO_CAMPO_PADRAO:.0%}). "
                              f"Use 0 pra voltar ao comportamento antigo (100%% validação).")
    parser.add_argument("--seed", type=int, default=42,
                         help="Semente aleatória — mesmo seed + mesmo dado + mesmo código = mesmo "
                              "resultado. Mude só se quiser testar sensibilidade a inicialização.")
    args = parser.parse_args()

    treinar(args)


if __name__ == "__main__":
    main()