# Modelos e Métricas

## Modelos Candidatos

Três modelos foram avaliados usando **validação cruzada estratificada 5-fold** com `StandardScaler`:

| Modelo                  | Descrição                        | Hiperparâmetros Principais                                      |
| ----------------------- | -------------------------------- | --------------------------------------------------------------- |
| **Logistic Regression** | Modelo linear com regularização  | `max_iter=1000`, `class_weight=balanced`                        |
| **Random Forest**       | Ensemble de árvores (bagging)    | `n_estimators=200`, `max_depth=10`, `class_weight=balanced`     |
| **Gradient Boosting**   | Ensemble de árvores (boosting)   | `n_estimators=200`, `max_depth=5`, `learning_rate=0.1`          |

---

## Resultados da Validação Cruzada (5-Fold)

| Modelo                   | F1-Score  | Recall    | Precision | AUC-ROC   | Accuracy  |
| ------------------------ | --------- | --------- | --------- | --------- | --------- |
| Logistic Regression      | 0.858     | 0.818     | 0.901     | 0.893     | 0.872     |
| **Random Forest** ✅     | **0.913** | **0.952** | 0.878     | 0.898     | **0.894** |
| Gradient Boosting        | 0.906     | 0.942     | 0.873     | **0.908** | 0.892     |

**Modelo selecionado:** Random Forest — melhor F1-Score (0.913) com altíssimo Recall (0.952).

---

## Métricas no Conjunto de Teste (20%)

| Métrica    | Valor |
| ---------- | ----- |
| Accuracy   | 0.837 |
| F1-Score   | 0.888 |
| Recall     | 0.925 |
| Precision  | 0.854 |
| AUC-ROC    | 0.917 |

---

## Métrica de Seleção: F1-Score

O **F1-Score** foi escolhido como métrica principal por ser a média harmônica entre Precision e Recall. No contexto da Associação Passos Mágicos:

- **Recall alto é crítico**: um falso negativo (aluno em risco não identificado) pode significar a perda da janela de intervenção educacional
- **Precision importa**: falsos positivos geram alocação desnecessária de recursos, mas o custo é menor que não identificar um aluno em risco
- **F1-Score equilibra ambos**, favorecendo modelos com bom recall sem sacrificar totalmente a precision

---

## Feature Importances (Random Forest)

| Feature             | Importância (%) |
| ------------------- | --------------- |
| Idade 22            | 31.0%           |
| Pedra 22            | 14.9%           |
| Matemática          | 10.1%           |
| IDA                 | 10.0%           |
| IPV                 | 7.0%            |
| Português           | 6.0%            |
| Anos no programa    | 5.2%            |
| IEG                 | 4.1%            |
| IAA                 | 3.8%            |
| IPS                 | 3.0%            |
| Rec Psicologia      | 1.5%            |
| Gênero              | 1.1%            |
| Instituição         | 0.9%            |
| Atingiu PV          | 0.5%            |
| Indicado            | 0.4%            |
| Destaque IEG        | 0.3%            |
| Destaque IDA        | 0.1%            |
| Destaque IPV        | 0.1%            |

---

## Features Utilizadas (18)

### Numéricas (9)

| Feature           | Descrição                                       |
| ----------------- | ----------------------------------------------- |
| IAA               | Indicador de Auto Avaliação                     |
| IEG               | Indicador de Engajamento                        |
| IPS               | Indicador Psicossocial                          |
| IDA               | Indicador de Desempenho Acadêmico               |
| IPV               | Indicador de Ponto de Virada                    |
| Matem             | Nota de Matemática                              |
| Portug            | Nota de Português                               |
| Idade 22          | Idade do aluno em 2022                          |
| Anos_no_programa  | 2022 − Ano de ingresso (feature derivada)       |

### Categóricas (9, encodadas)

| Feature              | Encoding                                                     |
| -------------------- | ------------------------------------------------------------ |
| Gênero               | Menina=0, Menino=1                                           |
| Instituição de ensino | Escola Pública=0, Rede Decisão=1, Escola JP II=2            |
| Pedra 22             | Quartzo=0, Ágata=1, Ametista=2, Topázio=3 (ordinal)         |
| Atingiu PV           | Não=0, Sim=1                                                 |
| Indicado             | Não=0, Sim=1                                                 |
| Rec Psicologia       | Sem limitações=0, Não atendido=1, ..., Requer avaliação=4   |
| Destaque IEG_bin     | Melhorar=0, Destaque=1                                       |
| Destaque IDA_bin     | Melhorar=0, Destaque=1                                       |
| Destaque IPV_bin     | Melhorar=0, Destaque=1                                       |

---

## Features Excluídas (Data Leakage)

| Feature     | Motivo da Exclusão                                                              |
| ----------- | ------------------------------------------------------------------------------- |
| **IAN**     | Correlação de −0.98 com o target — codifica diretamente a adequação de nível    |
| **INDE 22** | Índice composto que inclui IAN na fórmula                                       |
| **Fase**    | Combinada com Idade, reconstrói deterministicamente o target                    |
| **Inglês**  | 67% de valores nulos — feature esparsa demais para ser útil                     |

---

## Justificativa de Confiabilidade

1. **Recall de 92.5%** — apenas 7.5% dos alunos em risco não são detectados
2. **Remoção de data leakage** — modelo aprende padrões genuínos, não atalhos estatísticos
3. **Validação cruzada** — F1 = 0.913 ± 0.023 demonstra baixa variância entre folds
4. **Split estratificado** — mantém proporção de classes em treino/teste (80/20)
5. **Interpretabilidade** — features mais importantes são indicadores educacionais conhecidos pelos educadores
