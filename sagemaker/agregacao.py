"""
Regra de "alerta inteligente" — transforma varias deteccoes por objeto
(YOLO, uma imagem pode ter varias plantas/caixas) num status geral unico
pra planta/foto, priorizando nao deixar passar sinal de alerta.

Taxonomia simplificada pra 2 classes (decisao do projeto: praga e
doenca deixaram de ser distinguidas, viram "nao_saudavel" junto):

1. Conta quantas caixas de cada classe (saudavel / nao_saudavel)
2. Quem tiver MAIS caixas vence
3. Em empate de quantidade -> quem tiver a caixa de MAIOR confianca
   individual vence (nao deixa uma deteccao forte se diluir num empate)
4. Em empate total (quantidade E confianca) -> nao_saudavel vence, por
   seguranca (prefere falso alerta a deixar passar problema real)
"""

# "pest" mapeia pra nao_saudavel tambem, de proposito — mesmo que o
# modelo atual (2 classes) nao tenha mais essa saida, mapear ela aqui
# garante que, se um modelo futuro voltar a distinguir praga como
# categoria propria, o sistema nao quebra nem ignora silenciosamente
# essas deteccoes — elas so caem automaticamente em "nao_saudavel"
MAPA_CLASSES_ROBOFLOW = {
    "healthy": "saudavel",
    "disease": "nao_saudavel",
    "pest": "nao_saudavel",
}


def agregar_deteccoes(deteccoes):
    """
    deteccoes: lista de dicts {"classe": "saudavel"|"nao_saudavel", "confianca": float}
    Retorna: (status_geral, confianca_do_status, probabilidades_por_classe, motivo)
    """
    if not deteccoes:
        # nenhuma deteccao na imagem inteira -> nao tem planta de milho
        # reconhecivel (esse modelo assume que a etapa 1 ja confirmou
        # que e milho; se nao achou nada aqui, e sinal de imagem ruim)
        probabilidades_vazias = {"saudavel": 0.0, "nao_saudavel": 0.0, "nao_milho": 1.0}
        return "nao_milho", 0.0, probabilidades_vazias, "sem_deteccao"

    contagem = {"saudavel": 0, "nao_saudavel": 0}
    maior_confianca_por_classe = {"saudavel": 0.0, "nao_saudavel": 0.0}

    for d in deteccoes:
        contagem[d["classe"]] += 1
        if d["confianca"] > maior_confianca_por_classe[d["classe"]]:
            maior_confianca_por_classe[d["classe"]] = d["confianca"]

    if contagem["nao_saudavel"] > contagem["saudavel"]:
        vencedora = "nao_saudavel"
        motivo = "quantidade"
    elif contagem["saudavel"] > contagem["nao_saudavel"]:
        vencedora = "saudavel"
        motivo = "quantidade"
    else:
        # empate na quantidade de caixas -> decide por confianca; em
        # empate total, nao_saudavel vence por seguranca
        if maior_confianca_por_classe["saudavel"] > maior_confianca_por_classe["nao_saudavel"]:
            vencedora = "saudavel"
        else:
            vencedora = "nao_saudavel"
        motivo = "confianca"

    confianca_vencedora = maior_confianca_por_classe[vencedora]
    total_caixas = sum(contagem.values())
    probabilidades = {
        classe: round(contagem[classe] / total_caixas, 4) for classe in contagem
    }
    probabilidades["nao_milho"] = 0.0

    return vencedora, confianca_vencedora, probabilidades, motivo