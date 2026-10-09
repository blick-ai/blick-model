"""
inference.py — script de inferencia customizado pro SageMaker.

Pipeline novo de 3 estagios (decisao do professor, substitui o modelo
unico de deteccao direta que tinhamos antes):

  ESTAGIO 1 - segmentacao milho/nao-milho (corn_seg.pt, antigo
  pesos_Yolo26.pt): roda sobre a foto INTEIRA e devolve os poligonos de
  cada planta de milho que aparece nela. Se nao achar NENHUMA planta, a
  foto inteira ja e "nao_milho" e a gente nem chama o estagio 2/3.

  ESTAGIO 2 - isolamento por planta (replica a logica do
  isolate_corn_plants_corrigido.py, mas em memoria, sem salvar arquivo em
  disco): pra cada poligono do estagio 1, apaga tudo que nao e a planta
  (fundo preto) e recorta no bounding box da mascara, com uma margem.
  Vira um crop isolado por planta.

  ESTAGIO 3 - classificacao saudavel/nao_saudavel (best.pt, modelo de
  CLASSIFICACAO, nao deteccao): roda em cima de CADA crop isolado do
  estagio 2, uma planta por vez.

As classificacoes individuais (uma por planta) sao juntadas num status
geral unico pra foto pela regra em agregacao.py (por area da planta doente,
ver la). Cada item de analise_por_planta agora tambem traz area_pct (area do
poligono em % da foto) e aninhada (poligono quase todo dentro de outro
maior), pra API poder aplicar/ajustar a regra. Tambem traz contorno
(poligono simplificado, pontos [x, y] em 0..1) e caixa ([x1, y1, x2, y2] em
0..1) de cada planta, pro front desenhar as marcacoes sobre a foto. O
contrato antigo continua valendo: so ganhou campos.
"""

import base64
import io
import json
import os

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from agregacao import MAPA_CLASSES, agregar_classificacoes

# abaixo desse valor, uma planta detectada pelo estagio 1 e descartada
# por ser ruido demais pra confiar (nao isola, nao classifica)
LIMIAR_CONFIANCA_SEGMENTACAO = 0.25

# mesmo valor usado no gerar_anotacoes_milho.py (etapa de treino) - evita
# isolar/classificar manchas minusculas que o modelo de segmentacao
# acertou por acaso
LIMIAR_AREA_MINIMA_PLANTA = 400

# mesma margem usada no isolate_corn_plants_corrigido.py, em pixels, ao
# redor do recorte de cada planta isolada
PADDING_ISOLAMENTO = 15

# poligono com pelo menos essa fracao da area dentro de outro maior e
# considerado aninhado (duplicado/fragmento da mesma planta)
LIMITE_ANINHADA = 0.90


def model_fn(model_dir):
    """Chamado uma vez, quando o container "acorda" (cold start). Carrega
    os DOIS modelos - segmentacao (estagio 1) e classificacao (estagio 3)."""
    modelo_segmentacao = YOLO(os.path.join(model_dir, "corn_seg.pt"))
    modelo_classificacao = YOLO(os.path.join(model_dir, "best.pt"))
    return {"segmentacao": modelo_segmentacao, "classificacao": modelo_classificacao}


def input_fn(request_body, content_type):
    # o blick-api manda a imagem em bytes crus, com esse content-type —
    # e o formato real que descobrimos em producao, testando de verdade
    if content_type in ("application/x-image", "image/jpeg", "image/png", "image/jpg"):
        imagem = Image.open(io.BytesIO(request_body)).convert("RGB")
        return imagem

    # fallback: JSON com base64 (formato original que eu tinha assumido
    # antes de testar contra o backend real — mantido por flexibilidade,
    # caso algum outro chamador use esse formato)
    if content_type == "application/json":
        corpo = json.loads(request_body)
        imagem_base64 = corpo.get("imagem_base64") or corpo.get("image")
        if not imagem_base64:
            raise ValueError("Corpo da requisicao precisa ter 'imagem_base64'")
        image_bytes = base64.b64decode(imagem_base64)
        imagem = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return imagem

    raise ValueError(f"Content-Type nao suportado: {content_type}")


def _isolar_planta(imagem_bgr, poligono_px, padding=PADDING_ISOLAMENTO):
    """Mesma logica do isolate_corn_plants_corrigido.py: apaga (preto)
    tudo que nao e a planta e recorta no bounding box da mascara, com
    margem - so que aqui tudo em memoria, sem ler/escrever arquivo."""
    altura, largura = imagem_bgr.shape[:2]
    mascara = np.zeros((altura, largura), dtype=np.uint8)
    pontos = np.round(poligono_px).astype(np.int32)
    cv2.fillPoly(mascara, [pontos], 255)

    isolada = cv2.bitwise_and(imagem_bgr, imagem_bgr, mask=mascara)

    x, y, w, h = cv2.boundingRect(mascara)
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(largura, x + w + padding), min(altura, y + h + padding)
    return isolada[y1:y2, x1:x2]


# maximo de pontos guardados por contorno (o poligono do YOLO tem centenas)
MAX_PONTOS_CONTORNO = 40


def _contorno_normalizado(poligono, largura, altura, max_pontos=MAX_PONTOS_CONTORNO):
    """Simplifica o poligono (Douglas-Peucker) e devolve os pontos em 0..1
    (x/largura, y/altura), pra o front desenhar em cima da foto em qualquer
    tamanho. Aumenta a tolerancia ate caber em max_pontos."""
    pts = np.round(poligono).astype(np.int32).reshape(-1, 1, 2)
    perimetro = cv2.arcLength(pts, True)
    eps = 0.002 * perimetro
    simples = cv2.approxPolyDP(pts, eps, True)
    while len(simples) > max_pontos and eps < perimetro:
        eps *= 1.4
        simples = cv2.approxPolyDP(pts, eps, True)
    simples = simples.reshape(-1, 2).astype(float)
    return [[round(float(x) / largura, 4), round(float(y) / altura, 4)] for x, y in simples]


def _caixa_normalizada(poligono, largura, altura):
    """[x1, y1, x2, y2] do retangulo que envolve o poligono, em 0..1."""
    x1, y1 = poligono.min(axis=0)
    x2, y2 = poligono.max(axis=0)
    return [round(float(x1) / largura, 4), round(float(y1) / altura, 4),
            round(float(x2) / largura, 4), round(float(y2) / altura, 4)]


def _marcar_aninhadas(poligonos, largura, altura, limite=LIMITE_ANINHADA):
    """Pra cada poligono, True se >= limite da sua area esta dentro de OUTRO
    poligono maior (mesma planta/folha contada duas vezes). Usa mascaras
    reduzidas (lado maior = 400 px) so pra ficar barato."""
    esc = 400 / max(largura, altura)
    mascaras = []
    for pol in poligonos:
        m = np.zeros((int(altura * esc) + 1, int(largura * esc) + 1), dtype=np.uint8)
        cv2.fillPoly(m, [np.round(pol * esc).astype(np.int32)], 1)
        mascaras.append(m.astype(bool))
    areas = [m.sum() for m in mascaras]

    aninhadas = []
    for k in range(len(poligonos)):
        dentro = False
        if areas[k]:
            for j in range(len(poligonos)):
                if j != k and areas[j] > areas[k]:
                    if (mascaras[k] & mascaras[j]).sum() / areas[k] >= limite:
                        dentro = True
                        break
        aninhadas.append(dentro)
    return aninhadas


def _resposta_sem_planta(motivo="sem_planta_detectada"):
    probabilidades = {"saudavel": 0.0, "nao_saudavel": 0.0, "nao_milho": 1.0}
    return {
        "status_geral": "nao_milho",
        "confianca_status_geral": 1.0,
        "probabilidades": probabilidades,
        "subtipo": None,
        "confianca_subtipo": None,
        "quantidade_deteccoes": 0,
        "motivo_decisao": motivo,
        "analise_por_planta": [],
    }


def predict_fn(imagem, modelos):
    modelo_segmentacao = modelos["segmentacao"]
    modelo_classificacao = modelos["classificacao"]

    # ESTAGIO 1 — segmentacao milho/nao-milho na foto inteira
    resultado_segmentacao = modelo_segmentacao.predict(
        imagem, conf=LIMIAR_CONFIANCA_SEGMENTACAO, verbose=False
    )[0]

    if resultado_segmentacao.masks is None or len(resultado_segmentacao.masks.data) == 0:
        return _resposta_sem_planta()

    poligonos = resultado_segmentacao.masks.xy  # coordenadas de pixel, resolucao original

    # imagem em BGR (convencao OpenCV) pra poder recortar - a mesma usada
    # pelo isolate_corn_plants_corrigido.py quando le com cv2.imread
    imagem_rgb = np.array(imagem)
    imagem_bgr = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR)

    altura_img, largura_img = imagem_bgr.shape[:2]
    classificacoes = []
    analise_por_planta = []
    poligonos_usados = []

    for poligono in poligonos:
        if len(poligono) < 3:
            continue

        xs, ys = poligono[:, 0], poligono[:, 1]
        area = (xs.max() - xs.min()) * (ys.max() - ys.min())
        if area < LIMIAR_AREA_MINIMA_PLANTA:
            continue

        # ESTAGIO 2 — isola essa planta (fundo preto + recorte)
        crop_bgr = _isolar_planta(imagem_bgr, poligono)
        if crop_bgr is None or crop_bgr.size == 0:
            continue

        # ESTAGIO 3 — classifica o crop isolado (saudavel/nao_saudavel)
        resultado_classificacao = modelo_classificacao.predict(crop_bgr, verbose=False)[0]
        probs = resultado_classificacao.probs
        if probs is None:
            continue

        idx_classe = int(probs.top1)
        confianca = float(probs.top1conf)
        nome_classe = modelo_classificacao.names[idx_classe]  # "healthy" | "unhealthy"
        classe_traduzida = MAPA_CLASSES.get(nome_classe)
        if classe_traduzida is None:
            # classe desconhecida (nao deveria acontecer, mas nao trava por isso)
            continue

        area_pct = 100.0 * cv2.contourArea(poligono.astype(np.float32)) / (largura_img * altura_img)
        classificacoes.append({
            "classe": classe_traduzida,
            "confianca": confianca,
            "area_pct": area_pct,
        })
        analise_por_planta.append({
            "classe": classe_traduzida,
            "confianca": round(confianca, 4),
            "area_pct": round(area_pct, 3),
        })
        poligonos_usados.append(poligono)

    if not classificacoes:
        # o estagio 1 achou poligonos, mas nenhum sobreviveu aos filtros
        # (area minima / recorte vazio) — trata igual a "sem planta"
        return _resposta_sem_planta(motivo="poligonos_descartados_pelo_filtro")

    aninhadas = _marcar_aninhadas(poligonos_usados, largura_img, altura_img)
    for item, an, cls_, pol in zip(analise_por_planta, aninhadas, classificacoes, poligonos_usados):
        item["aninhada"] = bool(an)
        cls_["aninhada"] = bool(an)
        # marcacao da planta na foto (coordenadas 0..1), pro front desenhar
        item["contorno"] = _contorno_normalizado(pol, largura_img, altura_img)
        item["caixa"] = _caixa_normalizada(pol, largura_img, altura_img)

    status_geral, confianca_status_geral, probabilidades, motivo = agregar_classificacoes(classificacoes)

    return {
        "status_geral": status_geral,
        "confianca_status_geral": round(float(confianca_status_geral), 4),
        "probabilidades": probabilidades,
        "subtipo": None,
        "confianca_subtipo": None,
        # campos extras, so pra auditoria/debug — nao fazem parte do
        # contrato minimo que o blick-api le, mas nao atrapalham
        "quantidade_deteccoes": len(classificacoes),
        "motivo_decisao": motivo,
        "analise_por_planta": analise_por_planta,
    }


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"