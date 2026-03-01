# Arquitetura do Projeto

## Estrutura de Diretórios

```
fiap-mlet3-fase5/
├── DATATHON/                          # Dados brutos e dicionário de dados
│   ├── BASE DE DADOS PEDE 2024 - DATATHON.xlsx
│   ├── Dicionário Dados Datathon.pdf
│   └── Bases antigas/                 # Datasets de referência anteriores
├── src/                               # Código-fonte modularizado
│   ├── preprocessing/                 # Pipeline de pré-processamento
│   │   ├── data_loader.py             # Carregamento de dados (Excel)
│   │   ├── cleaner.py                 # Limpeza: nulos, tipos, texto
│   │   ├── feature_engineering.py     # Criação de features + encoding
│   │   └── pipeline.py               # Orquestração (load → clean → engineer)
│   ├── training/                      # Treinamento do modelo
│   │   ├── train.py                   # CV, seleção, treino final
│   │   └── run_training.py            # CLI: pipeline completo de treino
│   ├── evaluation/                    # Avaliação do modelo
│   │   └── evaluate.py               # Métricas (F1, Recall, AUC-ROC, etc.)
│   ├── api/                           # API REST
│   │   ├── app.py                     # FastAPI (/health, /predict)
│   │   └── schemas.py                 # Schemas Pydantic (request/response)
│   ├── monitoring/                    # Monitoramento
│   │   └── drift_detector.py          # Drift com Evidently AI
│   └── utils/                         # Utilitários
│       ├── constants.py               # Constantes e mapeamentos
│       └── logging_config.py          # Loguru: console + arquivo + JSONL
├── tests/                             # 13 arquivos de teste (89% cobertura)
│   ├── conftest.py                    # Fixtures compartilhadas
│   ├── test_data_loader.py
│   ├── test_cleaner.py
│   ├── test_feature_engineering.py
│   ├── test_pipeline.py
│   ├── test_training.py
│   ├── test_run_training.py
│   ├── test_evaluation.py
│   ├── test_api.py
│   ├── test_schemas.py
│   ├── test_monitoring.py
│   ├── test_constants.py
│   └── test_logging.py
├── dashboards/                        # Dashboard Streamlit
│   └── app.py                         # Simulador + Modelo + Monitoramento + Drift
├── models/                            # Artefatos do modelo
│   ├── model_v1.joblib                # RandomForest treinado
│   ├── pipeline_v1.joblib             # StandardScaler fitted
│   ├── reference_data.parquet         # Dados de referência (drift)
│   └── training_metrics.json          # Métricas de treino (dashboard)
├── logs/                              # Logs e predições
│   ├── app.log                        # Log geral da aplicação
│   └── predictions.jsonl              # Log de predições (monitoramento)
├── docs/                              # Documentação adicional
│   ├── arquitetura.md                 # Estrutura e diagrama MLOps
│   └── modelos.md                     # Tabela de modelos e métricas
├── Dockerfile                         # Imagem Docker (Python 3.12-slim)
├── docker-compose.yml                 # API (8000) + Dashboard (8501)
├── requirements.txt                   # Dependências Python
├── pyproject.toml                     # Config do projeto e pytest
└── README.md                          # Documentação principal
```

---

## Diagrama do Pipeline MLOps

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE MLOps                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │  DADOS   │───▶│ LIMPEZA  │───▶│  ENGENHARIA   │───▶│   TREINAMENTO    │  │
│  │  BRUTOS  │    │          │    │  DE FEATURES   │    │                  │  │
│  │ (Excel)  │    │ cleaner  │    │  feature_eng   │    │ CV 5-Fold + 3   │  │
│  │          │    │ .py      │    │  .py           │    │ modelos          │  │
│  └──────────┘    └──────────┘    └───────────────┘    └────────┬─────────┘  │
│       │                                                        │            │
│       │  data_loader.py                                        ▼            │
│       │                                                ┌──────────────┐     │
│       │                                                │  SELEÇÃO DO  │     │
│       │                                                │  MELHOR      │     │
│       │                                                │  MODELO      │     │
│       │                                                │  (F1-Score)  │     │
│       │                                                └──────┬───────┘     │
│       │                                                       │             │
│       │                                                       ▼             │
│       │                                              ┌─────────────────┐    │
│       │                                              │  AVALIAÇÃO      │    │
│       │                                              │  (Test Set 20%) │    │
│       │                                              │  evaluate.py    │    │
│       │                                              └────────┬────────┘    │
│       │                                                       │             │
│       │                                                       ▼             │
│       │                                              ┌─────────────────┐    │
│       │                                              │  SERIALIZAÇÃO   │    │
│       │                                              │  model.joblib   │    │
│       │                                              │  scaler.joblib  │    │
│       │                                              │  ref_data.pq    │    │
│       │                                              │  metrics.json   │    │
│       │                                              └────────┬────────┘    │
│       │                                                       │             │
├───────┼───────────────────────────────────────────────────────┼─────────────┤
│       │                  DEPLOY & SERVING                     │             │
│       │                                                       ▼             │
│       │  ┌─────────────────────────────────────────────────────────────┐    │
│       │  │                    Docker Compose                           │    │
│       │  │  ┌──────────────────────┐  ┌────────────────────────────┐  │    │
│       │  │  │  API (FastAPI)       │  │  Dashboard (Streamlit)     │  │    │
│       │  │  │  :8000               │  │  :8501                     │  │    │
│       │  │  │                      │  │                            │  │    │
│       │  │  │  /health             │  │  🎯 Simulador de Predição  │  │    │
│       │  │  │  /predict            │  │  🧠 Sobre o Modelo         │  │    │
│       │  │  │  /docs (Swagger)     │  │  📊 Monitoramento          │  │    │
│       │  │  │                      │  │  🔍 Detecção de Drift      │  │    │
│       │  │  └──────────┬───────────┘  └─────────────┬──────────────┘  │    │
│       │  │             │                            │                 │    │
│       │  └─────────────┼────────────────────────────┼─────────────────┘    │
│       │                │                            │                      │
├───────┼────────────────┼────────────────────────────┼──────────────────────┤
│       │         MONITORAMENTO                       │                      │
│       │                │                            │                      │
│       │                ▼                            │                      │
│       │  ┌──────────────────────┐                   │                      │
│       │  │  Logs (Loguru)       │───────────────────┘                      │
│       │  │  • app.log           │                                          │
│       │  │  • predictions.jsonl │──▶ Evidently AI (Drift Report)           │
│       │  └──────────────────────┘                                          │
│       │                                                                    │
└───────┴────────────────────────────────────────────────────────────────────┘
```

---

## Fluxo de Dados

1. **Ingestão** — `data_loader.py` carrega o arquivo Excel do PEDE 2024 (860 alunos, 42 colunas)
2. **Limpeza** — `cleaner.py` remove registros com Matemática e Português ambos nulos, preenche nulos restantes com mediana, normaliza textos
3. **Feature Engineering** — `feature_engineering.py` cria o target binário (`Defas_bin`), encoda categorias (ordinal, binário), remove features com data leakage
4. **Treinamento** — `train.py` avalia 3 modelos candidatos com validação cruzada estratificada 5-fold, seleciona o melhor por F1-Score
5. **Avaliação** — `evaluate.py` gera métricas completas no conjunto de teste (split 80/20 estratificado)
6. **Serialização** — modelo (`joblib`), scaler (`joblib`), dados de referência (`parquet`) e métricas (`json`) são salvos em `models/`
7. **Serving** — FastAPI serve o modelo via `/predict` com validação Pydantic, retornando predição, probabilidade e nível de risco
8. **Monitoramento** — cada predição é logada em `predictions.jsonl` com timestamp, input, output e latência
9. **Drift** — Evidently AI compara distribuições treino vs. produção; dashboard Streamlit visualiza os resultados

---

## Stack Tecnológica

| Componente        | Tecnologia                 |
| ----------------- | -------------------------- |
| Linguagem         | Python 3.12                |
| Frameworks de ML  | scikit-learn, pandas, numpy |
| API               | FastAPI + Uvicorn          |
| Serialização      | joblib                     |
| Testes            | pytest + pytest-cov        |
| Empacotamento     | Docker + Docker Compose    |
| Deploy            | Local (Docker)             |
| Monitoramento     | Evidently AI + Streamlit   |
| Logging           | Loguru                     |
| Visualização      | Plotly                     |

---

## Serviços Docker

| Serviço     | Porta | Descrição                              |
| ----------- | ----- | -------------------------------------- |
| `api`       | 8000  | API REST (FastAPI) com health check    |
| `dashboard` | 8501  | Dashboard interativo (Streamlit)       |

Ambos os serviços compartilham volumes de `logs/` e `models/` para comunicação via artefatos.
