"""
inference.py — script de inferencia customizado pro SageMaker.

Le a foto, roda o modelo YOLO (deteccao de objeto: healthy/disease/pest,
podendo ter varias caixas por imagem, ja que uma foto pode ter varias
plantas de milho), aplica a regra de "alerta inteligente" (agregacao.py)
pra virar UM status geral por foto, e devolve no MESMO formato que o
backend ja espera de um modelo de classificacao — o contrato entre
SageMaker e blick-api nao muda, so o que acontece por dentro daqui.
"""

import base64
import io
import json
import os

from PIL import Image
from ultralytics import YOLO

from agregacao import MAPA_CLASSES_ROBOFLOW, agregar_deteccoes

# abaixo desse valor, uma deteccao e descartada por ser ruido demais pra
# confiar (nao entra nem na contagem nem na agregacao)
LIMIAR_CONFIANCA_DETECCAO = 0.25


def model_fn(model_dir):
    """Chamado uma vez, quando o container "acorda" (cold start)."""
    caminho_pesos = os.path.join(model_dir, "best.pt")
    modelo = YOLO(caminho_pesos)
    return modelo


def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Content-Type nao suportado: {content_type}")

    corpo = json.loads(request_body)
    # aceita tanto "imagem_base64" (nosso padrao) quanto "image" (generico)
    imagem_base64 = corpo.get("imagem_base64") or corpo.get("image")
    if not imagem_base64:
        raise ValueError("Corpo da requisicao precisa ter 'imagem_base64'")

    image_bytes = base64.b64decode(imagem_base64)
    imagem = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return imagem


def predict_fn(imagem, modelo):
    resultados = modelo.predict(imagem, conf=LIMIAR_CONFIANCA_DETECCAO, verbose=False)
    r = resultados[0]

    deteccoes = []
    for classe_idx, confianca in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        nome_original = modelo.names[int(classe_idx)]  # "healthy" | "disease" | "pest"
        classe_traduzida = MAPA_CLASSES_ROBOFLOW.get(nome_original)
        if classe_traduzida is None:
            # classe desconhecida (nao deveria acontecer, mas nao trava por isso)
            continue
        deteccoes.append({"classe": classe_traduzida, "confianca": float(confianca)})

    status_geral, confianca_status_geral, probabilidades, motivo = agregar_deteccoes(deteccoes)

    return {
        "status_geral": status_geral,
        "confianca_status_geral": round(float(confianca_status_geral), 4),
        "probabilidades": probabilidades,
        "subtipo": None,
        "confianca_subtipo": None,
        # campos extras, so pra auditoria/debug — nao fazem parte do
        # contrato minimo que o blick-api le, mas nao atrapalham
        "quantidade_deteccoes": len(deteccoes),
        "motivo_decisao": motivo,
    }


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"