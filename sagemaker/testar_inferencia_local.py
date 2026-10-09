"""
Roda o inference.py NO SEU COMPUTADOR, sem SageMaker, sem AWS e sem tocar em
nada de producao. Serve pra ver as marcacoes de verdade antes de fazer deploy.

Uso (PowerShell, dentro da pasta sagemaker, tudo numa linha so):
  python testar_inferencia_local.py --foto C:\\caminho\\foto.jpg

Opcoes:
  --seg pesos_Yolo26.pt    pesos da segmentacao (vira corn_seg.pt)
  --cls best.pt            pesos da classificacao
  --saida marcado.jpg      foto com os contornos desenhados (padrao: marcado.jpg)
  --sem-preprocessar       nao gira pelo EXIF nem reduz pra 1600 px (a API faz os dois)

Precisa de: pip install ultralytics opencv-python pillow numpy
"""
import argparse
import io
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference  # noqa: E402

COR = {"saudavel": (80, 175, 76), "nao_saudavel": (80, 80, 199)}  # BGR


def preprocessar_como_a_api(caminho):
    """Mesmo que a API faz antes de chamar o endpoint: gira pelo EXIF e
    reduz pra no maximo 1600 px, em JPEG."""
    im = ImageOps.exif_transpose(Image.open(caminho)).convert("RGB")
    im.thumbnail((1600, 1600))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def desenhar(bytes_img, resposta, saida):
    arr = cv2.imdecode(np.frombuffer(bytes_img, np.uint8), cv2.IMREAD_COLOR)
    h, w = arr.shape[:2]
    fs = max(0.7, w / 1300)                 # escala do texto conforme o tamanho da foto
    linha = max(3, round(3 * fs))
    plantas = [p for p in resposta.get("analise_por_planta", []) if p.get("contorno")]

    # 1) preenchimento transparente + contorno (aninhada: linha mais fina)
    camada = arr.copy()
    pol = []
    for p in plantas:
        pts = np.array([[x * w, y * h] for x, y in p["contorno"]], np.int32).reshape(-1, 1, 2)
        pol.append(pts)
        cv2.fillPoly(camada, [pts], COR.get(p["classe"], (138, 137, 139)))
    arr = cv2.addWeighted(camada, 0.18, arr, 0.82, 0)
    for p, pts in zip(plantas, pol):
        cor = COR.get(p["classe"], (138, 137, 139))
        esp = max(2, linha // 2) if p.get("aninhada") else linha
        cv2.polylines(arr, [pts], True, (0, 0, 0), esp + 2, cv2.LINE_AA)
        cv2.polylines(arr, [pts], True, cor, esp, cv2.LINE_AA)

    # 2) rotulos: fundo colorido, texto branco, dentro da foto e sem se sobrepor
    usados = []
    for p in sorted(plantas, key=lambda q: (q["caixa"][1], q["caixa"][0])):
        if p.get("aninhada"):
            continue                          # repeticao de outra planta: so o contorno
        nome = "Saudavel" if p["classe"] == "saudavel" else "Nao saudavel"
        txt = nome
        (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fs, 2)
        bw, bh = tw + 12, th + base + 8
        x = int(min(max(p["caixa"][0] * w, 0), w - bw))
        y = int(p["caixa"][1] * h) - bh          # logo acima da caixa
        if y < 0:
            y = int(p["caixa"][1] * h) + 2       # sem espaco em cima: vai pra dentro
        for _ in range(30):                      # empurra pra baixo se bater em outro rotulo
            if any(x < ux + uw and ux < x + bw and y < uy + uh and uy < y + bh for ux, uy, uw, uh in usados):
                y += bh + 2
            else:
                break
        y = int(min(y, h - bh))
        usados.append((x, y, bw, bh))
        cor = COR.get(p["classe"], (138, 137, 139))
        cv2.rectangle(arr, (x, y), (x + bw, y + bh), cor, -1)
        cv2.rectangle(arr, (x, y), (x + bw, y + bh), (0, 0, 0), 1)
        cv2.putText(arr, txt, (x + 6, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fs, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(saida, arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--foto", required=True)
    ap.add_argument("--seg", default="pesos_Yolo26.pt")
    ap.add_argument("--cls", default="best.pt")
    ap.add_argument("--saida", default="marcado.jpg")
    ap.add_argument("--sem-preprocessar", action="store_true")
    a = ap.parse_args()

    for c in (a.foto, a.seg, a.cls):
        if not os.path.exists(c):
            raise SystemExit(f"nao achei: {c}")

    with tempfile.TemporaryDirectory() as pasta:
        shutil.copy(a.seg, os.path.join(pasta, "corn_seg.pt"))
        shutil.copy(a.cls, os.path.join(pasta, "best.pt"))
        modelos = inference.model_fn(pasta)

    if a.sem_preprocessar:
        dados = open(a.foto, "rb").read()
    else:
        dados = preprocessar_como_a_api(a.foto)

    imagem = inference.input_fn(dados, "application/x-image")
    resposta = inference.predict_fn(imagem, modelos)
    # o que o SageMaker devolve de verdade: passa pelo output_fn
    corpo, _ = inference.output_fn(resposta, "application/json")

    print(f"status_geral: {resposta['status_geral']} (confianca {resposta['confianca_status_geral']})")
    print(f"plantas: {len(resposta['analise_por_planta'])}   tamanho da resposta: {len(corpo) / 1024:.1f} KB")
    for i, p in enumerate(resposta["analise_por_planta"], 1):
        pontos = len(p.get("contorno") or [])
        print(f"  {i}. {p['classe']:13s} conf {p['confianca']:.3f}  area {p.get('area_pct')}%  "
              f"aninhada={p.get('aninhada')}  contorno={pontos} pontos  caixa={p.get('caixa')}")
        if not p.get("contorno") or not p.get("caixa"):
            print("     ATENCAO: sem contorno/caixa nessa planta (inference.py antigo?)")
    desenhar(dados, resposta, a.saida)
    print(f"foto marcada salva em: {a.saida}")


if __name__ == "__main__":
    main()