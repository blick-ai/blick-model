# -*- coding: utf-8 -*-
"""
deploy_endpoint.py
-------------------
Sobe o model.tar.gz (gerado pelo empacotar_modelo.py) como um endpoint
real do SageMaker, usando o container gerenciado de PyTorch da AWS.

Usa boto3 direto (API de baixo nivel do SageMaker), de proposito, em vez
do SDK de alto nivel `sagemaker` — esse SDK teve uma reestruturacao grande
de versao recente demais pra eu garantir que o codigo funciona sem poder
testar contra uma conta AWS de verdade. boto3 e mais verboso mas e estavel
ha anos, sem essa incerteza.

IMPORTANTE — conta academica com permissoes limitadas: TODOS os recursos
criados aqui (Model, EndpointConfig, Endpoint) recebem automaticamente as
tags obrigatorias abaixo.

Exemplo de uso:
    python3 deploy_endpoint.py \
        --model-tar ./model.tar.gz \
        --role-arn arn:aws:iam::<conta>:role/<role-do-sagemaker> \
        --endpoint-name blick-classificador \
        --bucket blick-capturas-tcc-640168426886-us-east-1-an
"""

import argparse
import os
import time

import boto3

TAGS = [
    {"Key": "creator", "Value": "LUCASCRAPINO_22006672"},
    {"Key": "environment", "Value": "GRADUACAO"},
    {"Key": "group", "Value": "CMD04"},
    {"Key": "owner", "Value": "BOSSINI"},
    {"Key": "project", "Value": "TCC"},
]

CONTA_DLC_POR_REGIAO = {
    "us-east-1": "763104351884",
    "us-east-2": "763104351884",
    "us-west-1": "763104351884",
    "us-west-2": "763104351884",
}


def montar_image_uri(region, framework_version="2.1.0", py_version="py310"):
    conta_dlc = CONTA_DLC_POR_REGIAO.get(region)
    if not conta_dlc:
        raise ValueError(
            f"Regiao {region!r} nao esta na tabela local de contas DLC — confira "
            f"https://github.com/aws/deep-learning-containers/blob/master/available_images.md "
            f"e adicione a conta certa em CONTA_DLC_POR_REGIAO."
        )
    return (
        f"{conta_dlc}.dkr.ecr.{region}.amazonaws.com/"
        f"pytorch-inference:{framework_version}-cpu-{py_version}"
    )


def enviar_para_s3_se_necessario(caminho_ou_uri, bucket, region):
    if caminho_ou_uri.startswith("s3://"):
        return caminho_ou_uri

    if not os.path.exists(caminho_ou_uri):
        raise FileNotFoundError(f"model.tar.gz não encontrado: {caminho_ou_uri}")

    s3 = boto3.client("s3", region_name=region)
    chave = f"blick/modelos/{int(time.time())}/model.tar.gz"
    print(f"[DEPLOY] Enviando {caminho_ou_uri} para s3://{bucket}/{chave} ...")
    s3.upload_file(caminho_ou_uri, bucket, chave)
    return f"s3://{bucket}/{chave}"


def deploy(model_tar, role_arn, endpoint_name, bucket, region, instance_type, modo, memoria_mb):
    model_s3_uri = enviar_para_s3_se_necessario(model_tar, bucket, region)
    image_uri = montar_image_uri(region)
    print(f"[DEPLOY] Container gerenciado: {image_uri}")
    print(f"[DEPLOY] Modo: {modo}"
          + (f" ({memoria_mb}MB)" if modo == "serverless" else f" ({instance_type})"))

    sm = boto3.client("sagemaker", region_name=region)
    timestamp = time.strftime("%Y%m%d%H%M%S")
    model_name = f"blick-modelo-{timestamp}"
    config_name = f"blick-config-{timestamp}"

    print(f"[DEPLOY] Criando Model '{model_name}'...")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_s3_uri,
        },
        ExecutionRoleArn=role_arn,
        Tags=TAGS,
    )

    if modo == "serverless":
        # SEM custo quando parado — so cobra pelo tempo de cada chamada.
        # Ideal pra uso esporadico de teste/TCC, ao contrario do endpoint
        # tradicional (ml.t2.medium etc.), que cobra por hora rodando
        # mesmo sem nenhuma requisicao chegando.
        variant = {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "ServerlessConfig": {
                "MemorySizeInMB": memoria_mb,
                "MaxConcurrency": 1,
            },
        }
    else:
        variant = {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstanceType": instance_type,
        }

    print(f"[DEPLOY] Criando EndpointConfig '{config_name}'...")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[variant],
        Tags=TAGS,
    )

    endpoint_existe = False
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        endpoint_existe = True
    except sm.exceptions.ClientError:
        pass

    if endpoint_existe:
        print(f"[DEPLOY] Endpoint '{endpoint_name}' já existe — atualizando pra nova versão do modelo...")
        sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    else:
        print(f"[DEPLOY] Criando Endpoint '{endpoint_name}' (isso demora alguns minutos)...")
        if modo == "realtime":
            print("[DEPLOY] IMPORTANTE: modo 'realtime' fica RODANDO (e sendo cobrado) até vocês "
                  "deletarem — não esqueçam de derrubar depois do teste se não for usar de imediato.")
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name, Tags=TAGS)

    print("[DEPLOY] Aguardando o endpoint ficar 'InService'...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name, WaiterConfig={"Delay": 15, "MaxAttempts": 60})

    print(f"\n[DEPLOY] Endpoint '{endpoint_name}' no ar.")
    print(f"[DEPLOY] Configure SAGEMAKER_ENDPOINT_NAME={endpoint_name} no blick_api.")
    if modo == "serverless":
        print("[DEPLOY] Modo serverless: primeira chamada depois de um tempo parado pode "
              "demorar alguns segundos a mais (cold start) — normal, não é erro.")


def main():
    parser = argparse.ArgumentParser(description="Faz o deploy do modelo como endpoint SageMaker.")
    parser.add_argument("--model-tar", required=True,
                         help="Caminho local do model.tar.gz OU um s3://... já enviado.")
    parser.add_argument("--role-arn", required=True,
                         help="ARN da execution role do SageMaker (peça pro professor se não souber qual usar).")
    parser.add_argument("--endpoint-name", default="blick-classificador")
    parser.add_argument("--bucket", required=True, help="Bucket usado só se --model-tar for um caminho local.")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--modo", choices=["serverless", "realtime"], default="serverless",
                         help="'serverless' (padrão) só cobra pelo tempo de cada chamada, sem custo "
                              "parado — melhor opção pra uso esporádico de teste/TCC. 'realtime' fica "
                              "sempre ligado (mais caro, mas sem cold start).")
    parser.add_argument("--memoria-mb", type=int, default=3072, choices=[1024, 2048, 3072, 4096, 5120, 6144],
                         help="Memória do modo serverless — 3072MB é de sobra pro nosso modelo (resnet18/160px).")
    parser.add_argument("--instance-type", default="ml.t2.medium",
                         help="Só usado no modo 'realtime'. ml.t2.medium é a opção mais barata típica — "
                              "confira antes quais tipos a conta acadêmica permite.")
    args = parser.parse_args()

    deploy(args.model_tar, args.role_arn, args.endpoint_name, args.bucket, args.region,
           args.instance_type, args.modo, args.memoria_mb)


if __name__ == "__main__":
    main()