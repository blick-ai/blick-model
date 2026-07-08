# -*- coding: utf-8 -*-
"""
prepare_dataset.py
-------------------
Unifica os datasets brutos de folhas de milho (PlantVillage, Kaggle,
Roboflow, capturas do Klar, etc.) em um manifesto unico, padroniza as
imagens para um formato comum e audita possiveis vieses de fundo/protocolo
de captura por classe antes de qualquer treino.

ESTRUTURA ESPERADA EM data/raw/:
    data/raw/<nome_do_dataset>/<classe>/*.jpg
    (ex.: data/raw/plantvillage/Healthy/Healthy__950_.jpg)

Se um dataset nao tiver subpastas por classe, a classe e inferida a partir
do prefixo do nome do arquivo antes de "__" (ex.: "Gray_Leaf_Spot__985.JPG").

Saidas:
    data/manifest.csv         -> uma linha por imagem, com metadados e
                                  as metricas de auditoria de viés
    data/processed/<classe>/  -> imagens padronizadas (mesmo tamanho,
                                  RGB, fundo tratado)
    Relatorio de auditoria de viés impresso no console ao final.

Exemplos:
    python3 prepare_dataset.py --raw-dir data/raw --out-dir data
    python3 prepare_dataset.py --raw-dir data/raw --only-audit
"""

import os
import csv
import argparse
import hashlib
import multiprocessing
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------
EXTENSOES_VALIDAS = (".jpg", ".jpeg", ".png")
TARGET_SIZE = 256            # lado do quadrado apos padronizacao
BORDA_PX = 12                 # espessura da faixa de borda analisada
LIMIAR_SATURACAO_PIXEL = 0.15  # pixel individual abaixo disso conta como "dessaturado" (cinza/preto)
LIMIAR_FRACAO_FUNDO = 0.15     # fracao da borda dessaturada acima disso = ha fundo de estudio visivel
LIMIAR_DIFERENCA_VIES = 0.35   # diferenca de proporcao entre classes que dispara alerta


# ---------------------------------------------------------------------------
# Estrutura de uma linha do manifesto
# ---------------------------------------------------------------------------
@dataclass
class ItemManifesto:
    caminho_original: str
    caminho_processado: Optional[str]
    classe: str
    classe_bruta: str
    status_geral: str
    origem_dataset: str
    largura: int
    altura: int
    proporcao: float
    tem_alpha: bool
    borda_fundo_estudio: bool
    borda_fracao_dessaturada: float


# ---------------------------------------------------------------------------
# Descoberta de classe e origem a partir do caminho do arquivo
# ---------------------------------------------------------------------------
def inferir_classe_e_origem(caminho, raw_dir):
    """
    Tenta extrair (classe, origem_dataset) da estrutura de pastas.
    Se nao houver subpasta de classe, usa o prefixo do nome do arquivo
    antes de "__" (convencao comum em datasets tipo PlantVillage).
    """
    relativo = os.path.relpath(caminho, raw_dir)
    partes = relativo.split(os.sep)

    origem_dataset = partes[0] if len(partes) > 1 else "desconhecido"

    if len(partes) >= 3:
        # data/raw/<dataset>/<classe>/arquivo.jpg
        classe = partes[-2]
    else:
        # sem subpasta de classe -> tenta pelo nome do arquivo
        nome = os.path.splitext(os.path.basename(caminho))[0]
        classe = nome.split("__")[0] if "__" in nome else "desconhecido"

    return classe.strip(), origem_dataset.strip()


# ---------------------------------------------------------------------------
# Normalizacao de taxonomia
# ---------------------------------------------------------------------------
# Datasets de fontes diferentes nomeiam a mesma classe biologica de jeitos
# diferentes (ex.: "Common Rust", "Common_Rust" e "Corn___Common_rust" sao
# a mesma doenca). Sem consolidar isso, o manifesto trata cada variacao de
# nome como uma classe separada. Ao adicionar um dataset novo, rode com
# --only-audit e cheque o aviso "CLASSES BRUTAS NAO MAPEADAS" no final —
# ele lista o que falta incluir aqui.

# Nomes de pasta que nao pertencem a este classificador (ex.: sobra do
# dataset do filtro planta/nao-planta do Klar) ou sao genericos demais
# para virar um rotulo confiavel sem contexto adicional.
CLASSES_DESCARTAR = {
    "Not Maize Leaf",
    "img",
    "maize",
    "maize-crops",
    "Plantas de milho",
    "Infected",   # generico demais - nao diz qual doenca/praga
    "mosquito",   # nao e praga de milho - contaminacao de dataset generico de insetos
    "sawfly",     # incomum em milho - mesma suspeita
}

MAPA_CLASSES = {
    # --- saudavel ---
    "Healthy": "Saudavel",
    "Healthy corn": "Saudavel",
    "Corn___healthy": "Saudavel",
    "Maize healthy": "Saudavel",

    # --- doencas fungicas do escopo original do artigo ---
    "Blight": "Mancha_de_Turcicum",
    "Corn___Northern_Leaf_Blight": "Mancha_de_Turcicum",
    "Northern Leaf Blight": "Mancha_de_Turcicum",
    "Maize leaf blight": "Mancha_de_Turcicum",
    "Milho (Corn) - Mancha_Turcicum (Northern Leaf Blight) - 1": "Mancha_de_Turcicum",

    "Gray Leaf Spot": "Cercosporiose",
    "Gray_Leaf_Spot": "Cercosporiose",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Cercosporiose",

    # --- praga do escopo original do artigo ---
    "Maize fall armyworm": "Lagarta_do_Cartucho",
    "armyworm": "Lagarta_do_Cartucho",  # REVISAR: confirmar especie (Spodoptera vs. Mythimna)

    # --- doencas novas incluidas na expansao de escopo ---
    "Common Rust": "Ferrugem_Comum",
    "Common_Rust": "Ferrugem_Comum",
    "Corn___Common_rust": "Ferrugem_Comum",

    "Maize streak virus": "Virus_do_Estriamento",  # bom exemplo de origem viral p/ o artigo

    "Milho (Corn) - Antracnose (Anthracnose) - 1": "Antracnose",
    "Milho (Corn) - Antracnose (Anthracnose) - Cropped": "Antracnose",

    "Milho (Corn) - Enfezamento (Bushy stunt) - 1": "Enfezamento",  # origem bacteriana/fitoplasma

    "Milho (Corn) - Ferrugem_Branca (Tropical rust) - 1": "Ferrugem_Branca",
    "Milho (Corn) - Ferrugem_Branca (Tropical rust) - Cropped": "Ferrugem_Branca",

    "Milho (Corn) - Ferrugem_Polissora (Southern corn rust) - 1": "Ferrugem_Polissora",

    "Milho (Corn) - Lixa (Scab) - 1": "Lixa",

    "Milho (Corn) - Mancha_Bipolaris (Southern corn leaf blight) - 1": "Mancha_Bipolaris",
    "Milho (Corn) - Mancha_Bipolaris (Southern corn leaf blight) - Cropped": "Mancha_Bipolaris",

    "Milho (Corn) - Mancha_Branca (Phaeosphaeria Leaf Spot) - 1": "Mancha_Branca",

    "Milho (Corn) - Mancha_Diplodia (Diplodia leaf streak) - 1": "Mancha_Diplodia",
    "Milho (Corn) - Mancha_Diplodia (Diplodia leaf streak) - Cropped": "Mancha_Diplodia",

    # --- pragas novas incluidas na expansao de escopo ---
    "aphids": "Pulgao",
    "Maize grasshoper": "Gafanhoto",
    "grasshopper": "Gafanhoto",
    "Maize leaf beetle": "Besouro",
    "beetle": "Besouro",
    "mites": "Acaro",
    "stem_borer": "Broca_do_Colmo",

    # --- suspeitas resolvidas ---
    "bollworm": "Lagarta_da_Espiga",  # praga real do milho (Helicoverpa spp.), promovida

    # --- ainda pendente: comparar visualmente com Cercosporiose antes de decidir ---
    # Inspecao visual manual (jul/2026): a pasta mistura padroes diferentes
    # entre si (pustulas arredondadas em alguns exemplos, estrias lineares
    # finas em outros, halo amarelo em estagio avancado em outros ainda) -
    # nao bate de forma consistente nem com Cercosporiose nem com
    # Ferrugem_Comum. Joga em Outra_Doenca em vez de arriscar contaminar
    # uma classe limpa. Se algum dia sobrar tempo, vale re-triar manualmente
    # imagem a imagem, ou usar o proprio modelo ja treinado nas classes
    # limpas pra pre-classificar essas 1239 imagens e revisar so as duvidosas.
    "Maize leaf spot": "Outra_Doenca",

    # --- auto-mapeamento: pastas ja nomeadas com o nome da classe final ---
    # Facilita organizar manualmente lotes novos (ex.: fotos do Klar ja
    # triadas) sem precisar inventar um nome de pasta bruto so pra bater
    # com alguma entrada acima. Basta nomear a pasta com o nome canonico
    # direto (ex.: data/raw/klar/Saudavel/*.jpg) que ja funciona.
    "Saudavel": "Saudavel",
    "Cercosporiose": "Cercosporiose",
    "Ferrugem_Comum": "Ferrugem_Comum",
    "Mancha_de_Turcicum": "Mancha_de_Turcicum",
    "Virus_do_Estriamento": "Virus_do_Estriamento",
    "Ferrugem_Branca": "Ferrugem_Branca",
    "Mancha_Bipolaris": "Mancha_Bipolaris",
    "Outra_Doenca": "Outra_Doenca",
    "Outra_Praga": "Outra_Praga",  # bucket generico p/ praga visivel mas sem especie identificada
    "Lagarta_do_Cartucho": "Lagarta_do_Cartucho",
    "Lagarta_da_Espiga": "Lagarta_da_Espiga",
    "Acaro": "Acaro",
    "Besouro": "Besouro",
    "Broca_do_Colmo": "Broca_do_Colmo",
    "Gafanhoto": "Gafanhoto",
    "Pulgao": "Pulgao",
}


# ---------------------------------------------------------------------------
# Status geral (nivel hierarquico superior): saudavel / praga / doenca
# ---------------------------------------------------------------------------
# Usado pelo train.py para treinar um classificador de dois niveis: primeiro
# o status geral (robusto, classes bem populosas), depois o subtipo
# especifico (mais fragil nas classes raras). Ver conversa sobre hierarquia.
GRUPO_STATUS = {
    "Saudavel": "saudavel",

    "Cercosporiose": "doenca",
    "Ferrugem_Comum": "doenca",
    "Mancha_de_Turcicum": "doenca",
    "Virus_do_Estriamento": "doenca",
    "Ferrugem_Branca": "doenca",
    "Mancha_Bipolaris": "doenca",
    "Outra_Doenca": "doenca",

    "Lagarta_do_Cartucho": "praga",
    "Lagarta_da_Espiga": "praga",
    "Acaro": "praga",
    "Besouro": "praga",
    "Broca_do_Colmo": "praga",
    "Gafanhoto": "praga",
    "Pulgao": "praga",
    "Outra_Praga": "praga",
}


def obter_status_geral(classe):
    """
    Retorna "saudavel" / "praga" / "doenca" para uma classe canonica, ou
    "revisar" para classes ainda pendentes de decisao (prefixo REVISAR_ ou
    NAO_MAPEADA_) — nesse caso, o train.py deve excluir a imagem do
    treino ate a classe ser resolvida em GRUPO_STATUS.
    """
    if classe in GRUPO_STATUS:
        return GRUPO_STATUS[classe]
    return "revisar"


# Classes com poucas imagens demais para treinar isoladamente por enquanto
# (3 a 33 exemplos). Agrupadas num rotulo generico "Outra_Doenca" ate que
# haja mais dado — a classe_bruta de cada imagem continua guardando a
# doenca especifica original, entao e facil "promover" qualquer uma de
# volta a sua propria classe assim que o volume justificar, sem reprocessar
# nada: so tirar o nome daqui.
CLASSES_RARAS_PARA_AGRUPAR = frozenset({
    "Lixa",
    "Enfezamento",
    "Ferrugem_Polissora",
    "Mancha_Diplodia",
    "Mancha_Branca",
    "Antracnose",
})


# Versoes em minusculo, montadas uma unica vez, pra permitir lookup
# case-insensitive sem duplicar o dicionario inteiro escrito a mao acima.
# Isso existe porque ja tivemos DUAS vezes o mesmo tipo de bug: pasta
# criada como "outra_doenca" ou "Outra_doenca" em vez de "Outra_Doenca"
# fazia cair silenciosamente em NAO_MAPEADA. Com isso, qualquer variacao
# de maiuscula/minuscula no nome da pasta passa a funcionar igual.
_CLASSES_DESCARTAR_LOWER = {c.lower() for c in CLASSES_DESCARTAR}
_MAPA_CLASSES_LOWER = {k.lower(): v for k, v in MAPA_CLASSES.items()}


def normalizar_classe(classe_bruta):
    """
    Aplica o MAPA_CLASSES para consolidar nomes de pasta diferentes na
    mesma classe biologica, e agrupa classes raras demais em "Outra_Doenca".
    A comparacao e case-insensitive (ver comentario acima de _MAPA_CLASSES_LOWER).

    Retorna None se a classe deve ser descartada, ou o nome canonico. Se a
    classe bruta nao estiver no mapa, ela NAO e descartada silenciosamente:
    fica marcada com o prefixo "NAO_MAPEADA__", pra ser detectada depois
    (em `construir_manifesto`) e aparecer no relatorio final.

    Observacao: essa funcao nao mantem nenhum estado entre chamadas de
    proposito — isso permite rodar em processos separados (multiprocessing)
    sem precisar sincronizar nada entre eles.
    """
    classe_bruta_lower = classe_bruta.lower()

    if classe_bruta_lower in _CLASSES_DESCARTAR_LOWER:
        return None

    if classe_bruta_lower in _MAPA_CLASSES_LOWER:
        classe = _MAPA_CLASSES_LOWER[classe_bruta_lower]
        if classe in CLASSES_RARAS_PARA_AGRUPAR:
            return "Outra_Doenca"
        return classe

    return f"NAO_MAPEADA__{classe_bruta}"


# ---------------------------------------------------------------------------
# Metricas de auditoria de viés de fundo
# ---------------------------------------------------------------------------
def _rgb_para_saturacao(faixa_rgb_0_255):
    """
    Calcula a saturacao HSV (0-1) de um array Nx3 de pixels RGB (0-255),
    sem depender de OpenCV. Formula padrao: S = (max-min)/max (max>0).
    """
    arr = faixa_rgb_0_255 / 255.0
    maximo = arr.max(axis=1)
    minimo = arr.min(axis=1)
    saturacao = np.where(maximo > 0, (maximo - minimo) / np.clip(maximo, 1e-6, None), 0.0)
    return saturacao


def analisar_borda(img_rgb):
    """
    Analisa uma faixa ao redor da borda da imagem. Fundo de estudio (mesa
    cinza, fundo preto solido, etc.) e tipicamente dessaturado (tons de
    cinza), enquanto folha verde tem saturacao de cor bem mais alta.

    Importante: o fundo de estudio muitas vezes aparece so em UM canto ou
    lado da foto (a folha foi apoiada de forma nao-simetrica), entao a
    media de saturacao do perimetro inteiro dilui esse sinal. Por isso a
    metrica usada e a FRACAO de pixels da borda que sao individualmente
    dessaturados, nao a media global.

    Retorna (tem_fundo_de_estudio_visivel: bool, fracao_dessaturada: float)
    """
    arr = np.asarray(img_rgb, dtype=np.float32)
    h, w, _ = arr.shape
    b = min(BORDA_PX, h // 4, w // 4)
    if b < 2:
        return False, 0.0

    faixa = np.concatenate([
        arr[:b, :, :].reshape(-1, 3),
        arr[-b:, :, :].reshape(-1, 3),
        arr[:, :b, :].reshape(-1, 3),
        arr[:, -b:, :].reshape(-1, 3),
    ], axis=0)

    saturacoes = _rgb_para_saturacao(faixa)
    fracao_dessaturada = float(np.mean(saturacoes < LIMIAR_SATURACAO_PIXEL))
    return fracao_dessaturada > LIMIAR_FRACAO_FUNDO, round(fracao_dessaturada, 3)


# ---------------------------------------------------------------------------
# Padronizacao da imagem
# ---------------------------------------------------------------------------
def padronizar_imagem(caminho, tamanho=TARGET_SIZE):
    """
    Abre a imagem, resolve transparencia (compoe sobre cinza neutro em vez
    de deixar preto/transparente, que vira um atalho artificial pro
    modelo) e redimensiona com letterbox, preservando a proporcao original.
    """
    img = Image.open(caminho)
    tem_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)

    if tem_alpha:
        img = img.convert("RGBA")
        fundo = Image.new("RGB", img.size, (128, 128, 128))  # cinza neutro
        fundo.paste(img, mask=img.split()[-1])
        img = fundo
    else:
        img = img.convert("RGB")

    largura, altura = img.size

    # letterbox: redimensiona mantendo proporcao, preenche com cinza neutro
    escala = tamanho / max(largura, altura)
    nova_largura, nova_altura = int(largura * escala), int(altura * escala)
    img_redim = img.resize((nova_largura, nova_altura), Image.BILINEAR)

    tela = Image.new("RGB", (tamanho, tamanho), (128, 128, 128))
    offset = ((tamanho - nova_largura) // 2, (tamanho - nova_altura) // 2)
    tela.paste(img_redim, offset)

    return tela, img, tem_alpha, (largura, altura)


# ---------------------------------------------------------------------------
# Worker de um unico arquivo (roda em processo separado)
# ---------------------------------------------------------------------------
def _processar_um_arquivo(params):
    """
    Processa uma unica imagem de ponta a ponta: infere classe/origem,
    normaliza a classe, padroniza a imagem e roda a auditoria de fundo.

    Precisa ser uma funcao de modulo (nao aninhada) para ser "picklable" e
    poder rodar dentro de um multiprocessing.Pool.

    Retorna sempre uma tupla (status, payload):
      status="ok"         payload=ItemManifesto
      status="descartada" payload=None
      status="erro"       payload=mensagem de erro (str)
    """
    caminho, raw_dir, processed_dir, salvar_processadas = params

    classe_bruta, origem = inferir_classe_e_origem(caminho, raw_dir)
    classe = normalizar_classe(classe_bruta)

    if classe is None:
        return ("descartada", None)

    try:
        tela, img_original, tem_alpha, (largura, altura) = padronizar_imagem(caminho)
    except Exception as e:
        return ("erro", f"{caminho}: {e}")

    img_rgb_original = img_original.convert("RGB") if img_original.mode != "RGB" else img_original
    borda_fundo_estudio, fracao_dessaturada = analisar_borda(img_rgb_original)

    caminho_proc = None
    if salvar_processadas:
        pasta_classe = os.path.join(processed_dir, classe)
        os.makedirs(pasta_classe, exist_ok=True)
        # hash do caminho ORIGINAL completo (nao so o nome do arquivo) garante
        # nome de saida unico mesmo quando duas pastas brutas diferentes tem
        # arquivos com o mesmo nome (ex.: "1.jpg" em "Common Rust" E em
        # "Common_Rust", que caem na mesma classe Ferrugem_Comum). Sem isso,
        # a segunda imagem processada sobrescreve a primeira silenciosamente
        # no disco, mesmo aparecendo como duas linhas distintas no manifesto.
        digest = hashlib.md5(caminho.encode("utf-8")).hexdigest()[:10]
        nome_base = os.path.splitext(os.path.basename(caminho))[0]
        nome_saida = f"{origem}__{digest}__{nome_base}.jpg"
        caminho_proc = os.path.join(pasta_classe, nome_saida)
        tela.save(caminho_proc, quality=95)
        caminho_proc = os.path.abspath(caminho_proc)

    item = ItemManifesto(
        caminho_original=os.path.abspath(caminho),
        caminho_processado=caminho_proc,
        classe=classe,
        classe_bruta=classe_bruta,
        status_geral=obter_status_geral(classe),
        origem_dataset=origem,
        largura=largura,
        altura=altura,
        proporcao=round(largura / altura, 3) if altura else 0.0,
        tem_alpha=tem_alpha,
        borda_fundo_estudio=borda_fundo_estudio,
        borda_fracao_dessaturada=fracao_dessaturada,
    )
    return ("ok", item)


# ---------------------------------------------------------------------------
# Construcao do manifesto (em paralelo)
# ---------------------------------------------------------------------------
def construir_manifesto(raw_dir, out_dir, salvar_processadas=True, n_workers=None):
    processed_dir = os.path.join(out_dir, "processed")

    arquivos = [
        os.path.join(raiz, nome)
        for raiz, _, nomes in os.walk(raw_dir)
        for nome in nomes
        if nome.lower().endswith(EXTENSOES_VALIDAS)
    ]

    print(f"[MANIFESTO] {len(arquivos)} imagem(ns) encontrada(s) em {raw_dir}.")

    n_workers = n_workers or os.cpu_count() or 4
    print(f"[MANIFESTO] Processando com {n_workers} processo(s) em paralelo...")

    tarefas = [(caminho, raw_dir, processed_dir, salvar_processadas) for caminho in arquivos]

    itens = []
    descartadas = 0
    erros = []
    total = len(tarefas)

    # chunksize maior reduz overhead de comunicacao entre processos quando
    # ha muitos arquivos pequenos (nosso caso: dezenas de milhares de imagens)
    chunksize = max(1, total // (n_workers * 20)) if total else 1

    with multiprocessing.Pool(processes=n_workers) as pool:
        for i, (status, payload) in enumerate(
            pool.imap_unordered(_processar_um_arquivo, tarefas, chunksize=chunksize), 1
        ):
            if status == "ok":
                itens.append(payload)
            elif status == "descartada":
                descartadas += 1
            else:
                erros.append(payload)

            if i % 500 == 0 or i == total:
                print(f"  [MANIFESTO] {i}/{total} processadas...")

    nao_mapeadas = sorted({
        item.classe_bruta for item in itens if item.classe.startswith("NAO_MAPEADA__")
    })

    print(f"[MANIFESTO] {descartadas} imagem(ns) descartada(s) por estarem em CLASSES_DESCARTAR.")
    if erros:
        print(f"[MANIFESTO] {len(erros)} imagem(ns) ignorada(s) por erro ao abrir. Ex.: {erros[0]}")
    if nao_mapeadas:
        print("\n" + "!" * 70)
        print("CLASSES BRUTAS NAO MAPEADAS (adicione ao MAPA_CLASSES ou a CLASSES_DESCARTAR):")
        for nome in nao_mapeadas:
            print(f"  - {nome!r}")
        print("!" * 70 + "\n")

    return itens


def salvar_manifesto_csv(itens, caminho_csv):
    os.makedirs(os.path.dirname(caminho_csv), exist_ok=True)
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(itens[0]).keys()))
        writer.writeheader()
        for item in itens:
            writer.writerow(asdict(item))
    print(f"[MANIFESTO] Salvo em {caminho_csv} ({len(itens)} linhas).")


# ---------------------------------------------------------------------------
# Auditoria de viés: fundo uniforme correlacionado com a classe?
# ---------------------------------------------------------------------------
def auditar_vies(itens):
    """
    Para cada (origem_dataset, classe), calcula a proporcao de imagens com
    fundo de estudio na borda (baixa saturacao), a contagem absoluta de
    imagens, e a composicao por classe bruta original (util pra flagrar
    fusoes de pastas tipo "-1"/"-Cropped" que podem ser a mesma foto
    duas vezes sob nomes diferentes).
    """
    from collections import defaultdict

    grupos = defaultdict(list)
    brutas_por_grupo = defaultdict(lambda: defaultdict(int))
    total_por_classe = defaultdict(int)

    for item in itens:
        chave = (item.origem_dataset, item.classe)
        grupos[chave].append(item.borda_fundo_estudio)
        brutas_por_grupo[chave][item.classe_bruta] += 1
        total_por_classe[item.classe] += 1

    proporcao_por_grupo = {
        chave: sum(valores) / len(valores)
        for chave, valores in grupos.items()
    }
    contagem_por_grupo = {chave: len(valores) for chave, valores in grupos.items()}

    origens = {chave[0] for chave in proporcao_por_grupo}

    print("\n" + "=" * 70)
    print("RESUMO POR CLASSE (todas as origens somadas)")
    print("=" * 70)
    for classe, total in sorted(total_por_classe.items(), key=lambda x: x[1]):
        print(f"  {classe:<25} {total:>6} imagens")

    print("\n" + "=" * 70)
    print("AUDITORIA DE VIÉS DE FUNDO POR DATASET / CLASSE")
    print("=" * 70)

    alertas = []
    classes_poucas_imagens = sorted(
        (classe, total) for classe, total in total_por_classe.items() if total < 100
    )

    for origem in sorted(origens):
        classes_da_origem = {
            chave[1]: prop for chave, prop in proporcao_por_grupo.items()
            if chave[0] == origem
        }
        print(f"\n[{origem}]")
        for classe, prop in sorted(classes_da_origem.items()):
            n = contagem_por_grupo[(origem, classe)]
            print(f"  {classe:<25} {n:>6} imagens   fundo de estúdio em {prop*100:5.1f}%")

            # se a classe consolida mais de uma pasta bruta, mostra a composicao
            # (ex.: pega o caso de "-1" vs "-Cropped" sendo fundidos na mesma classe)
            brutas = brutas_por_grupo[(origem, classe)]
            if len(brutas) > 1:
                for nome_bruto, n_bruto in sorted(brutas.items(), key=lambda x: -x[1]):
                    print(f"      ↳ {n_bruto:>6}  de \"{nome_bruto}\"")

        if len(classes_da_origem) > 1:
            valores = list(classes_da_origem.values())
            diferenca = max(valores) - min(valores)
            if diferenca >= LIMIAR_DIFERENCA_VIES:
                alertas.append((origem, classes_da_origem, diferenca))

    print("\n" + "-" * 70)
    if alertas:
        print("ALERTAS — possível correlação espúria entre fundo e classe:\n")
        for origem, classes_da_origem, diferenca in alertas:
            print(f"  [{origem}] diferença de {diferenca*100:.1f} pontos percentuais entre classes.")
            print(f"    -> Verifique manualmente antes de treinar com essa fonte.")
    else:
        print("Nenhum viés de fundo evidente encontrado com o limiar atual.")

    if classes_poucas_imagens:
        print("\n" + "-" * 70)
        print("AVISO — classes com menos de 100 imagens NO TOTAL (todas as origens):\n")
        for classe, total in classes_poucas_imagens:
            print(f"  {classe}: apenas {total} imagens")

    print("=" * 70 + "\n")

    return alertas


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unifica, padroniza e audita viés dos datasets de milho do Blick."
    )
    parser.add_argument("--raw-dir", default="data/raw",
                         help="Pasta com os datasets brutos (organizados por dataset/classe).")
    parser.add_argument("--out-dir", default="data",
                         help="Pasta de saida (manifest.csv e processed/ serao criados aqui).")
    parser.add_argument("--only-audit", action="store_true",
                         help="Não salva imagens padronizadas, só gera o manifesto e a auditoria.")
    parser.add_argument("--workers", type=int, default=None,
                         help="Número de processos em paralelo (padrão: todos os núcleos disponíveis).")
    args = parser.parse_args()

    itens = construir_manifesto(
        args.raw_dir, args.out_dir,
        salvar_processadas=not args.only_audit,
        n_workers=args.workers,
    )

    if not itens:
        print("[MANIFESTO] Nenhuma imagem valida encontrada. Verifique --raw-dir.")
        return

    salvar_manifesto_csv(itens, os.path.join(args.out_dir, "manifest.csv"))
    auditar_vies(itens)


if __name__ == "__main__":
    main()