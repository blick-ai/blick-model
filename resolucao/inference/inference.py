# -*- coding: utf-8 -*-
"""
inference.py
------------
Script de inferencia para hospedar o classificador fitossanitario do Blick
como um endpoint do Amazon SageMaker, usando o container gerenciado de
PyTorch (nao precisamos construir nem manter nossa propria imagem Docker
de serving — so entregamos os pesos + este script).

ESTRUTURA ESPERADA NO model.tar.gz enviado ao SageMaker:
    model.tar.gz
    ├── modelo_melhor.pth
    ├── metadados.json
    └── code/
        └── inference.py   <- este arquivo

O SageMaker chama as 4 funcoes abaixo nesta ordem, uma vez por requisicao
(exceto model_fn, que roda uma vez quando o endpoint sobe):
    model_fn      -> carrega o modelo e os metadados na memoria
    input_fn      -> converte o corpo bruto da requisicao (bytes de imagem) em algo utilizavel
    predict_fn    -> roda a inferencia de verdade
    output_fn     -> converte o resultado em JSON pra devolver na resposta

Esse arquivo e propositalmente AUTOCONTIDO (a classe do modelo esta
duplicada aqui, nao importada de train.py) porque o container do SageMaker
so tem acesso ao que estiver dentro de code/ no model.tar.gz — nao ao
resto do repositorio.
"""

import os
import io
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

STATUS_LABELS = ["saudavel", "praga", "doenca"]


# ---------------------------------------------------------------------------
# Mesma arquitetura do train.py — duplicada de proposito, ver docstring acima
# ---------------------------------------------------------------------------
class ClassificadorHierarquico(nn.Module):
    def __init__(self, n_subtipos, backbone_nome="resnet18", pretrained=False):
        super().__init__()

        if backbone_nome == "resnet18":
            pesos = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=pesos)
        elif backbone_nome == "resnet34":
            pesos = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=pesos)
        else:
            raise ValueError(f"backbone desconhecido: {backbone_nome!r}")

        n_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.cabeca_status = nn.Linear(n_features, len(STATUS_LABELS))
        self.cabeca_subtipo = nn.Linear(n_features, n_subtipos)

    def forward(self, x):
        features = self.backbone(x)
        return self.cabeca_status(features), self.cabeca_subtipo(features)


# ---------------------------------------------------------------------------
# 1. model_fn — roda UMA VEZ quando o endpoint sobe, nao a cada requisicao
# ---------------------------------------------------------------------------
def model_fn(model_dir):
    with open(os.path.join(model_dir, "metadados.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    modelo = ClassificadorHierarquico(
        n_subtipos=max(1, len(meta["classe_para_idx"])),
        backbone_nome=meta.get("backbone", "resnet18"),
        pretrained=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(
        os.path.join(model_dir, "modelo_melhor.pth"), map_location=device
    )
    modelo.load_state_dict(state_dict)
    modelo.to(device)
    modelo.eval()

    transform = transforms.Compose([
        transforms.Resize((meta["img_size"], meta["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=meta["normalizacao_media"], std=meta["normalizacao_desvio"]),
    ])

    # mascara de coerencia: restringe o subtipo ao grupo compatibel com o
    # status geral ja decidido (ver bug "praga" + subtipo "Ferrugem_Comum"
    # descoberto na triagem das capturas reais do Klar)
    mascara_status = None
    if "classe_para_status" in meta:
        idx_para_classe = meta["idx_para_classe"]
        classe_para_status = meta["classe_para_status"]
        n_subtipos = len(idx_para_classe)
        mascara_status = {}
        for status in STATUS_LABELS:
            vetor = torch.zeros(n_subtipos, dtype=torch.bool)
            for idx_str, classe in idx_para_classe.items():
                if classe_para_status.get(classe) == status:
                    vetor[int(idx_str)] = True
            mascara_status[status] = vetor.to(device)

    return {
        "modelo": modelo, "meta": meta, "transform": transform,
        "device": device, "mascara_status": mascara_status,
    }


# ---------------------------------------------------------------------------
# 2. input_fn — converte o corpo bruto da requisicao numa imagem PIL
# ---------------------------------------------------------------------------
def input_fn(request_body, content_type="application/x-image"):
    if content_type in ("application/x-image", "application/octet-stream", "image/jpeg", "image/png"):
        return Image.open(io.BytesIO(request_body)).convert("RGB")

    raise ValueError(f"content_type nao suportado: {content_type!r}. Envie bytes de imagem (jpeg/png).")


# ---------------------------------------------------------------------------
# 3. predict_fn — a inferencia de verdade
# ---------------------------------------------------------------------------
def predict_fn(input_object, model_artifacts):
    modelo = model_artifacts["modelo"]
    meta = model_artifacts["meta"]
    transform = model_artifacts["transform"]
    device = model_artifacts["device"]
    mascara_status = model_artifacts["mascara_status"]

    x = transform(input_object).unsqueeze(0).to(device)

    with torch.no_grad():
        saida_status, saida_subtipo = modelo(x)
        prob_status = torch.softmax(saida_status, dim=1)[0]
        prob_subtipo = torch.softmax(saida_subtipo, dim=1)[0]

    idx_status = int(prob_status.argmax().item())
    status_previsto = STATUS_LABELS[idx_status]

    idx_subtipo = int(prob_subtipo.argmax().item())
    if mascara_status is not None:
        vetor_mascara = mascara_status[status_previsto]
        if vetor_mascara.any():
            prob_subtipo_mascarada = prob_subtipo.masked_fill(~vetor_mascara, float("-inf"))
            idx_subtipo = int(prob_subtipo_mascarada.argmax().item())

    return {
        "status_geral": status_previsto,
        "confianca_status_geral": round(prob_status[idx_status].item(), 4),
        "subtipo": meta["idx_para_classe"][str(idx_subtipo)],
        "confianca_subtipo": round(prob_subtipo[idx_subtipo].item(), 4),
        "probabilidades_status_geral": {
            STATUS_LABELS[i]: round(prob_status[i].item(), 4) for i in range(len(STATUS_LABELS))
        },
    }


# ---------------------------------------------------------------------------
# 4. output_fn — serializa o resultado como JSON pra resposta HTTP
# ---------------------------------------------------------------------------
def output_fn(prediction, accept="application/json"):
    if accept != "application/json":
        raise ValueError(f"accept nao suportado: {accept!r}. Este endpoint so retorna application/json.")
    return json.dumps(prediction, ensure_ascii=False), accept