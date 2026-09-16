# API de Previsão de Demanda de Estoque

Sistema de previsão de demanda de vendas usando séries temporais e machine learning, construído como um pipeline reprodutível — da análise exploratória ao modelo pronto para servir previsões. O objetivo não é a sofisticação matemática do modelo, e sim a engenharia de dados por trás dele: features sem vazamento de informação, validação temporal correta e comparação honesta contra um baseline.

Dataset: [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only) (Kaggle) — 5 anos de vendas diárias (2013–2017), 10 lojas × 50 itens, 913.000 registros.

## Resultados

Modelo avaliado no último trimestre de 2017 (holdout fora da amostra de treino), comparado contra um baseline ingênuo (previsão = venda do mesmo dia da semana anterior):

| Modelo | MAE | RMSE |
|---|---|---|
| Baseline (naive, t-7) | 9,08 | 12,05 |
| **XGBoost** | **5,95** | **7,71** |

**Redução de ~34,5% no erro absoluto médio (MAE)** em relação ao baseline.

## Principais insights da análise exploratória

- **Padrão semanal claro:** vendas crescem de segunda-feira (média 41,4) até domingo (média 62,1) — o dia de maior movimento.
- **Sazonalidade anual:** pico de vendas em julho (média 67,0), possivelmente ligado ao fim férias escolares; vale de vendas em janeiro e dezembro.
- **Tendência de crescimento:** aumento de ~35% na média de vendas entre 2013 e 2017, com desaceleração no ritmo de crescimento nos últimos anos.
- **Lojas têm hierarquia estável:** a loja 2 é a de maior volume em todos os 5 anos; a loja 7 é a de menor volume em todos os 5 anos — todas as lojas crescem juntas ano a ano, sugerindo um efeito estrutural de loja independente da tendência geral do mercado.
- **Demanda regular, não intermitente:** 0% dos dias têm venda zero, o que valida o uso de um modelo de regressão direto (sem necessidade de técnicas para demanda esparsa).
- **Outlier investigado, não descartado:** a maior venda registrada (231 unidades) ocorreu na loja 2, num domingo de julho — a convergência exata dos três fatores de maior volume identificados na análise, não um erro de dado.

## Principais decisões técnicas

- **Split temporal, não aleatório:** treino com tudo antes de 01/10/2017, teste com o último trimestre. Embaralhar os dados antes de separar treino/teste vazaria informação do futuro para o passado, o que não reflete o uso real do modelo (prever o futuro a partir do passado).
- **`store` e `item` tratados como variáveis categóricas**, não como inteiros contínuos — evita que o modelo assuma uma relação de ordem entre os códigos que não existe na realidade (loja 8 não é "duas vezes" a loja 4).
- **Features de lag e rolling calculadas agrupando por `(store, item)`**, com o `shift` aplicado antes do `rolling`. Sem esse agrupamento, uma janela móvel poderia misturar dados do fim de uma série com o início de outra, ou incluir o próprio valor do dia que está sendo previsto — dois tipos distintos de vazamento de dados.
- **Baseline avaliado na mesma janela de teste do modelo**, para garantir uma comparação justa — calcular o erro do baseline num período diferente do usado para o XGBoost tornaria os números incomparáveis.
- **Sem `early_stopping_rounds` nesta fase:** usar o conjunto de teste como `eval_set` durante o treino permitiria que o modelo "espiasse" o teste na escolha do número de árvores, contaminando a avaliação final. Usar esse recurso corretamente exigiria um terceiro conjunto de validação, retirado de dentro do período de treino.
- **Caminhos de arquivo resolvidos via `pathlib` a partir da localização do script**, não como strings relativas — evita que o código quebre dependendo de onde é executado (terminal na raiz do projeto vs. dentro de `src/`).
- **Modelo exportado como um bundle** (`{"model": ..., "feature_columns": ..., "test_start": ...}`), não como objeto puro — garante que qualquer ambiente que carregue o `.pkl` (ex: uma função serverless) saiba exatamente quais colunas usar e em qual ordem, sem depender do código-fonte original.

## Como rodar

```bash
pip install -r requirements.txt
python src/train.py
```

Isso gera `artifacts/model.pkl` (modelo + metadados) e `artifacts/metrics.json` (métricas de avaliação).

## Estrutura do projeto

```
.
├── artifacts/
│   ├── metrics.json            # métricas de avaliação
│   └── model.pkl               # modelo final + metadados (não versionado no Git)
├── data/
│   └── train.csv              # dataset bruto (não versionado no Git)
├── notebooks/
│   └── 01_eda.ipynb           # análise exploratória
├── src/
│   ├── features.py            # engenharia de features (calendário, lag, rolling)
│   └── train.py                # split temporal, baseline, treino, avaliação, export
├── requirements.txt
├── .gitignore
└── README.md
```

## Próximos passos

Implantação como API serverless na AWS, dentro do free tier permanente (sem custo, sem necessidade de gerenciar infraestrutura ativa):
- **AWS Lambda + Function URL** servindo o modelo para inferência
- **Amazon S3** como data lake para os dados brutos
- **EventBridge** disparando automação diária de simulação de novos dados
- **Supabase (PostgreSQL)** armazenando previsões para monitoramento de acurácia (MAE/RMSE) ao longo do tempo
