# -*- coding: utf-8 -*-
"""
agregacao.py — repositorio: blick-model (sagemaker/)
-----------------------------------------------------
Agrega as classificacoes individuais de cada planta isolada numa
captura para decidir o status_geral final da captura inteira.

Regra de decisao (v3, por AREA): a captura e nao_saudavel se existir pelo
menos uma planta classificada como nao_saudavel que
  - tenha confianca >= CONFIANCA_MINIMA_DOENTE e
  - ocupe pelo menos AREA_MINIMA_DOENTE_PCT da foto, e
  - se estiver aninhada em outro poligono maior (folha dentro de uma planta
    que o modelo chamou de saudavel), so conta se for grande e muito
    confiante (ANINHADA_*): e como uma folha com adesivo aparece.

Motivo: contar plantas nao separa foto com adesivo (1 poligono grande
doente) de foto limpa (varios fragmentos finos de folha seca, cada um
com 0.1 a 0.5% da foto). Calibrado em 2026-10-06 com fotos reais: limpas
ate 0.51%, com adesivo a partir de 0.97%.

Sem area_pct nos itens (formato antigo), cai na regra v2: pelo menos
LIMIAR_MINIMO_DOENTES plantas doentes. A API (blick-api) reaplica a mesma
regra a partir de analise_por_planta, entao este status e so o de origem.
"""

MAPA_CLASSES = {
    "healthy": "saudavel",
    "unhealthy": "nao_saudavel",
    # aliases legados, mantidos por seguranca caso o modelo antigo
    # ainda retorne esses rotulos em algum cenario
    "disease": "nao_saudavel",
    "pest": "nao_saudavel",
}

# alias mantido por compatibilidade com código anterior
MAPA_CLASSES_ROBOFLOW = MAPA_CLASSES

AREA_MINIMA_DOENTE_PCT = 0.8
CONFIANCA_MINIMA_DOENTE = 0.75
ANINHADA_AREA_MINIMA_PCT = 1.5
ANINHADA_CONFIANCA_MINIMA = 0.90

# regra antiga (v2), usada so quando nao ha area_pct
LIMIAR_MINIMO_DOENTES = 2


def agregar_classificacoes(classificacoes):
    """
    classificacoes: lista de dicts {"classe": str, "confianca": float,
    "area_pct": float (opcional), "aninhada": bool (opcional)}, um item por
    planta isolada e classificada na captura.

    Retorna: (vencedora, confianca, probabilidades, motivo)
    """
    if not classificacoes:
        return (
            "nao_milho",
            1.0,
            {"saudavel": 0.0, "nao_saudavel": 0.0, "nao_milho": 1.0},
            "sem_planta_detectada",
        )

    total = len(classificacoes)

    normalizadas = [
        {
            "classe": MAPA_CLASSES.get(c["classe"], c["classe"]),
            "confianca": c["confianca"],
            "area_pct": c.get("area_pct"),
            "aninhada": bool(c.get("aninhada")),
        }
        for c in classificacoes
    ]

    doentes = [c for c in normalizadas if c["classe"] == "nao_saudavel"]
    saudaveis = [c for c in normalizadas if c["classe"] == "saudavel"]
    n_doentes = len(doentes)

    if any(c["area_pct"] is not None for c in normalizadas):
        uteis = [
            c for c in doentes
            if (
                c["confianca"] >= ANINHADA_CONFIANCA_MINIMA
                and (c["area_pct"] or 0) >= ANINHADA_AREA_MINIMA_PCT
            ) or (
                not c["aninhada"]
                and c["confianca"] >= CONFIANCA_MINIMA_DOENTE
                and (c["area_pct"] or 0) >= AREA_MINIMA_DOENTE_PCT
            )
        ]
        eh_doente = bool(uteis)
        confianca_doente = max((c["confianca"] for c in uteis), default=0.0)
        motivo_doente = f"{len(uteis)}_doente_util_de_{n_doentes}_doentes_{total}_plantas"
        motivo_saudavel = f"nenhuma_doente_util_{n_doentes}_doentes_{total}_plantas"
    else:
        eh_doente = n_doentes >= LIMIAR_MINIMO_DOENTES
        confianca_doente = max((c["confianca"] for c in doentes), default=0.0)
        motivo_doente = f"{n_doentes}_de_{total}_plantas_nao_saudaveis"
        motivo_saudavel = f"menos_de_{LIMIAR_MINIMO_DOENTES}_plantas_doentes_{n_doentes}_de_{total}"

    if eh_doente:
        vencedora = "nao_saudavel"
        confianca = confianca_doente
        motivo = motivo_doente
    else:
        vencedora = "saudavel"
        confianca = (
            sum(c["confianca"] for c in saudaveis) / len(saudaveis)
            if saudaveis
            else 0.5
        )
        motivo = motivo_saudavel

    probabilidades = {
        "saudavel": 1.0 - (n_doentes / total),
        "nao_saudavel": n_doentes / total,
        "nao_milho": 0.0,
    }

    return (vencedora, confianca, probabilidades, motivo)