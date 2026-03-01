# 🔮 Passos Mágicos — Predição de Defasagem Escolar

Pipeline completa de MLOps para predição do risco de defasagem escolar de estudantes da Associação Passos Mágicos.

---

## 1. Visão Geral do Projeto

### Objetivo
Desenvolver um modelo preditivo capaz de estimar o **risco de defasagem escolar** de cada estudante da Associação Passos Mágicos, utilizando dados educacionais do período 2022-2024 (PEDE 2024).

A defasagem escolar ocorre quando um aluno está em uma fase/série abaixo da esperada para sua idade. Identificar precocemente alunos em risco permite intervenções educacionais e psicopedagógicas direcionadas.

### Solução Proposta
Construção de uma pipeline completa de Machine Learning com as melhores práticas de MLOps:
- **Pré-processamento** de dados com tratamento de valores ausentes e encoding de variáveis categóricas
- **Feature engineering** com exclusão de features que causam data leakage (IAN, INDE 22, Fase)
- **Treinamento e seleção** do melhor modelo via validação cruzada estratificada (5-fold)
- **Deploy** via API REST (FastAPI) empacotada em Docker
- **Monitoramento** contínuo com detecção de drift (Evidently AI) e dashboard interativo (Streamlit)
- **Testes unitários** com 89% de cobertura (acima do mínimo de 80%)

### Stack Tecnológica
| Componente | Tecnologia |
|------------|------------|
| Linguagem | Python 3.12 |
| Frameworks de ML | scikit-learn, pandas, numpy |
| API | FastAPI + Uvicorn |
| Serialização | joblib |
| Testes | pytest + pytest-cov |
| Empacotamento | Docker + Docker Compose |
| Deploy | Local (Docker) |
| Monitoramento | Evidently AI + Streamlit |
| Logging | Loguru |

---

## 2. Estrutura do Projeto

```
fiap-mlet3-fase5/
├── DATATHON/                          # Dados brutos e documentação do datathon
│   ├── BASE DE DADOS PEDE 2024 - DATATHON.xlsx
│   ├── Dicionário Dados Datathon.pdf
│   └── Bases antigas/                 # Datasets de referência anteriores
├── src/                               # Código-fonte modularizado
│   ├── preprocessing/                 # Pipeline de pré-processamento
│   │   ├── data_loader.py             # Carregamento dos dados
│   │   ├── cleaner.py                 # Limpeza e tratamento de nulos
│   │   ├── feature_engineering.py     # Engenharia de atributos
│   │   └── pipeline.py               # Orquestração do pipeline
│   ├── training/                      # Treinamento do modelo
│   │   ├── train.py                   # CV, seleção e treino final
│   │   └── run_training.py            # Script CLI para treino
│   ├── evaluation/                    # Avaliação do modelo
│   │   └── evaluate.py               # Métricas e relatórios
│   ├── api/                           # API REST
│   │   ├── app.py                     # Aplicação FastAPI
│   │   └── schemas.py                 # Schemas Pydantic (request/response)
│   ├── monitoring/                    # Monitoramento de drift
│   │   └── drift_detector.py          # Detecção de drift com Evidently
│   └── utils/                         # Utilitários
│       ├── constants.py               # Constantes e configurações
│       └── logging_config.py          # Configuração de logs
├── tests/                             # Testes unitários (100 testes, 89% cobertura)
│   ├── conftest.py                    # Fixtures compartilhadas
│   ├── test_data_loader.py
│   ├── test_cleaner.py
│   ├── test_feature_engineering.py
│   ├── test_pipeline.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   ├── test_api.py
│   ├── test_schemas.py
│   ├── test_monitoring.py
│   ├── test_constants.py
│   ├── test_logging.py
│   └── test_run_training.py
├── dashboards/                        # Dashboard de monitoramento
│   └── app.py                         # Aplicação Streamlit
├── models/                            # Artefatos do modelo treinado
│   ├── model_v1.joblib                # Modelo serializado
│   ├── pipeline_v1.joblib             # Scaler serializado
│   └── reference_data.parquet         # Dados de referência para drift
├── logs/                              # Logs da aplicação e predições
├── Dockerfile                         # Imagem Docker da API
├── docker-compose.yml                 # Orquestração API + Dashboard
├── requirements.txt                   # Dependências Python
├── pyproject.toml                     # Configuração do projeto e testes
└── README.md                          # Esta documentação
```

---

## 3. Instruções de Deploy

### Pré-requisitos
- Python 3.10+
- Docker e Docker Compose (para deploy containerizado)
- pip

### Instalação Local

```bash
# Clonar o repositório
git clone <repo-url>
cd fiap-mlet3-fase5

# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependências
pip install -r requirements.txt
```

### Treinar o Modelo

```bash
python -m src.training.run_training
```

O script executa automaticamente:
1. Carregamento e limpeza dos dados
2. Feature engineering
3. Validação cruzada com 3 modelos (LogisticRegression, RandomForest, GradientBoosting)
4. Seleção do melhor modelo por F1-Score
5. Treino final e salvamento dos artefatos em `models/`

### Executar a API Localmente

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

A API estará disponível em `http://localhost:8000`.
Documentação interativa (Swagger): `http://localhost:8000/docs`.

### Deploy com Docker

```bash
# Build e execução
docker-compose up --build

# Ou apenas a API
docker build -t passos-magicos-api .
docker run -p 8000:8000 -v ./models:/app/models -v ./logs:/app/logs passos-magicos-api
```

Serviços disponíveis:
- **API**: `http://localhost:8000`
- **Dashboard de Monitoramento**: `http://localhost:8501`

### Executar Testes

```bash
# Rodar todos os testes com cobertura
python -m pytest tests/ -v

# Verificar cobertura detalhada
python -m pytest tests/ --cov=src --cov-report=html
# Abrir htmlcov/index.html no navegador
```

---

## 4. Exemplos de Chamadas à API

### Health Check

```bash
curl http://localhost:8000/health
```

**Resposta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Predição de Risco

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "IAA": 8.5,
    "IEG": 7.0,
    "IPS": 6.5,
    "IDA": 6.0,
    "IPV": 7.3,
    "Matem": 6.5,
    "Portug": 7.0,
    "Idade 22": 14,
    "Ano ingresso": 2020,
    "Gênero": "Menina",
    "Instituição de ensino": "Escola Pública",
    "Pedra 22": "Ametista",
    "Atingiu PV": "Não",
    "Indicado": "Não",
    "Rec Psicologia": "Sem limitações",
    "Destaque IEG": "Destaque: A sua boa entrega das lições de casa.",
    "Destaque IDA": "Destaque: As suas boas notas na Passos Mágicos.",
    "Destaque IPV": "Destaque: A sua boa integração aos Princípios Passos Mágicos."
  }'
```

**Resposta:**
```json
{
  "prediction": 0,
  "probability": 0.2341,
  "risk_level": "Baixo",
  "message": "O estudante apresenta baixo risco de defasagem escolar."
}
```

### Via Python

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "IAA": 4.0, "IEG": 3.5, "IPS": 5.0, "IDA": 3.0,
    "IPV": 4.2, "Matem": 3.0, "Portug": 4.0,
    "Idade 22": 16, "Ano ingresso": 2021,
    "Gênero": "Menino",
    "Instituição de ensino": "Escola Pública",
    "Pedra 22": "Quartzo",
    "Atingiu PV": "Não", "Indicado": "Não",
    "Rec Psicologia": "Requer avaliação",
    "Destaque IEG": "Melhorar: Melhorar a sua entrega de lições de casa.",
    "Destaque IDA": "Melhorar: Empenhar-se mais nas aulas e avaliações.",
    "Destaque IPV": "Melhorar: Integrar-se mais aos Princípios Passos Mágicos.",
})
print(response.json())
# {"prediction": 1, "probability": 0.8721, "risk_level": "Alto",
#  "message": "O estudante apresenta alto risco de defasagem escolar. Intervenção recomendada."}
```

---

## 5. Etapas do Pipeline de Machine Learning

### 5.1 Pré-processamento dos Dados
- Carregamento do dataset XLSX (860 alunos, 42 colunas)
- Remoção de registros com Matemática e Português ambos ausentes (2 linhas)
- Preenchimento de valores nulos restantes com mediana
- Normalização de espaços em colunas de texto

### 5.2 Engenharia de Features
- **Variável target**: `Defas_bin` — binarização da coluna `Defas` (1 = em risco quando Defas < 0, 0 = no nível adequado)
- **Features derivadas**: `Anos_no_programa` = 2022 - Ano de ingresso
- **Encoding ordinal**: `Pedra 22` (Quartzo=0 < Ágata=1 < Ametista=2 < Topázio=3)
- **Encoding binário**: Gênero, Atingiu PV, Indicado, Destaques (IEG, IDA, IPV)
- **Encoding categórico**: Instituição de ensino, Rec Psicologia
- **Features excluídas** (data leakage):
  - `IAN` — Indicador de Adequação de Nível (correlação -0.98 com target, é basicamente a mesma informação)
  - `INDE 22` — Índice composto que inclui IAN na sua fórmula
  - `Fase` — combinado com Idade, reconstrói deterministicamente o target
  - `Inglês` — 67% de valores nulos

### 5.3 Treinamento e Validação
- Split estratificado 80/20 (treino/teste)
- Validação cruzada 5-fold estratificada
- StandardScaler para normalização das features
- Modelos candidatos: LogisticRegression, RandomForest, GradientBoosting
- Métrica de seleção: **F1-Score** (balanceamento entre precisão e recall)

### 5.4 Seleção de Modelo
| Modelo | F1 (CV) | Recall (CV) | AUC-ROC (CV) |
|--------|---------|-------------|--------------|
| LogisticRegression | 0.858 | 0.818 | 0.893 |
| **RandomForest** | **0.913** | **0.952** | **0.898** |
| GradientBoosting | 0.906 | 0.942 | 0.908 |

**Modelo selecionado: RandomForest** — melhor F1-Score (0.913) com alto Recall (0.952).

### 5.5 Avaliação Final (Test Set)
| Métrica | Valor |
|---------|-------|
| Accuracy | 0.837 |
| F1-Score | 0.888 |
| Recall | 0.925 |
| Precision | 0.854 |
| AUC-ROC | 0.917 |

### Por que este modelo é confiável para produção?

1. **Recall alto (92.5%)**: Em um contexto social onde deixar de identificar um aluno em risco tem consequências sérias, o modelo prioriza minimizar falsos negativos. Apenas 7.5% dos alunos em risco não são identificados.

2. **Sem data leakage**: Features que codificam diretamente o target (IAN, INDE 22, Fase) foram removidas após análise de correlação, garantindo que o modelo aprende padrões genuínos.

3. **Validação robusta**: F1-Score de 0.913 (±0.023) em validação cruzada 5-fold demonstra consistência e baixa variância.

4. **Features interpretáveis**: Os principais preditores (Idade, Pedra, notas de Matemática e Português, IDA) são indicadores educacionais compreensíveis para educadores.

5. **Feature importances**:
   - Idade 22 (31.0%) — idade determina em grande parte a fase ideal esperada
   - Pedra 22 (14.9%) — classificação de desempenho geral
   - Matemática (10.1%) e IDA (10.0%) — desempenho acadêmico direto
   - IPV (7.0%) e Português (6.0%) — indicadores complementares

### 5.6 Monitoramento
- Logs de todas as predições em formato JSONL para análise posterior
- Dashboard Streamlit com visualização de distribuição de predições e volume
- Detecção de data drift via Evidently AI comparando dados de referência (treino) vs. produção
- Alertas visuais quando drift é detectado em features

---

## 6. Sobre a Associação Passos Mágicos

A Associação Passos Mágicos tem uma trajetória de 32 anos transformando a vida de crianças e jovens de baixa renda no município de Embu-Guaçu por meio da educação. O PEDE (Pesquisa de Desenvolvimento Educacional) é o instrumento de avaliação que monitora o desenvolvimento dos alunos ao longo dos anos.

Saiba mais: [passosmagicos.org.br](https://passosmagicos.org.br/quem-somos/)
