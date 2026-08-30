"""
Regra de "alerta inteligente" — transforma varias deteccoes por objeto
(YOLO, uma imagem pode ter varias plantas/caixas) num status geral unico
pra planta/foto, priorizando nao deixar passar sinal de alerta:

1. Conta quantas caixas de cada classe (saudavel/praga/doenca)
2. Se (praga + doenca) tiver MAIS caixas que saudavel -> vence a classe
   de alerta com mais caixas entre praga/doenca
3. Senao (saudavel tem mais ou igual caixas) -> se a caixa de MAIOR
   confianca entre TODAS as deteccoes for de praga/doenca, ela vence
   mesmo em minoria (nao deixa uma deteccao forte se diluir)
4. Caso nenhuma das duas condicoes acima dispare -> saudavel vence
"""

MAPA_CLASSES_ROBOFLOW = {
    "healthy": "saudavel",
    "disease": "doenca",
    "pest": "praga",
}


def agregar_deteccoes(deteccoes):
    """
    deteccoes: lista de dicts {"classe": "saudavel"|"praga"|"doenca", "confianca": float}
    Retorna: (status_geral, confianca_do_status, probabilidades_por_classe)
    """
    if not deteccoes:
        # nenhuma deteccao na imagem inteira -> nao tem planta de milho
        # reconhecivel (esse modelo assume que a etapa 1 ja confirmou
        # que e milho; se nao achou nada aqui, e sinal de imagem ruim)
        probabilidades_vazias = {"saudavel": 0.0, "praga": 0.0, "doenca": 0.0, "nao_milho": 1.0}
        return "nao_milho", 0.0, probabilidades_vazias, "sem_deteccao"

    contagem = {"saudavel": 0, "praga": 0, "doenca": 0}
    maior_confianca_por_classe = {"saudavel": 0.0, "praga": 0.0, "doenca": 0.0}

    for d in deteccoes:
        contagem[d["classe"]] += 1
        if d["confianca"] > maior_confianca_por_classe[d["classe"]]:
            maior_confianca_por_classe[d["classe"]] = d["confianca"]

    caixas_alerta = contagem["praga"] + contagem["doenca"]
    caixas_saudavel = contagem["saudavel"]

    if caixas_alerta > caixas_saudavel:
        # entre praga/doenca: primeiro quem tem MAIS caixas vence; se
        # empatar em quantidade, quem tiver MAIOR confianca vence; se
        # empatar nos dois, doenca vence por padrao (desempate final)
        if contagem["praga"] > contagem["doenca"]:
            vencedora = "praga"
        elif contagem["doenca"] > contagem["praga"]:
            vencedora = "doenca"
        else:
            # empate na quantidade de caixas -> decide por confianca
            if maior_confianca_por_classe["praga"] > maior_confianca_por_classe["doenca"]:
                vencedora = "praga"
            else:
                # doenca vence tanto se tiver confianca maior quanto em empate total
                vencedora = "doenca"
        motivo = "quantidade"
    else:
        # saudavel tem mais ou igual caixas -> confere se alguma deteccao
        # de alerta tem confianca MAIOR que a maior confianca de saudavel
        maior_alerta = max(maior_confianca_por_classe["praga"], maior_confianca_por_classe["doenca"])
        if maior_alerta > maior_confianca_por_classe["saudavel"]:
            # mesmo criterio de desempate: doenca vence em empate de confianca
            vencedora = "praga" if maior_confianca_por_classe["praga"] > maior_confianca_por_classe["doenca"] else "doenca"
            motivo = "confianca"
        else:
            vencedora = "saudavel"
            motivo = "saudavel_predominante"

    confianca_vencedora = maior_confianca_por_classe[vencedora]
    total_caixas = sum(contagem.values())
    probabilidades = {
        classe: round(contagem[classe] / total_caixas, 4) for classe in contagem
    }
    probabilidades["nao_milho"] = 0.0

    return vencedora, confianca_vencedora, probabilidades, motivo