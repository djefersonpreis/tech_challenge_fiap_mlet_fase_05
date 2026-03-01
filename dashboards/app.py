"""Streamlit dashboard for model monitoring, prediction simulator, and model info."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
REFERENCE_DATA_PATH = MODELS_DIR / "reference_data.parquet"
MODEL_PATH = MODELS_DIR / "model_v1.joblib"
PIPELINE_PATH = MODELS_DIR / "pipeline_v1.joblib"
METRICS_PATH = MODELS_DIR / "training_metrics.json"

API_URL = "http://api:8000"  # docker-compose service name

st.set_page_config(
    page_title="Passos Mágicos - Dashboard",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 Passos Mágicos — Dashboard")

# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────


@st.cache_resource
def load_model_artifacts():
    """Load model and scaler once."""
    model, scaler = None, None
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(PIPELINE_PATH)
    except FileNotFoundError:
        pass
    return model, scaler


def load_predictions() -> pd.DataFrame:
    """Load prediction logs."""
    log_path = LOGS_DIR / "predictions.jsonl"
    if not log_path.exists():
        return pd.DataFrame()
    records = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_reference() -> pd.DataFrame:
    """Load reference data."""
    if REFERENCE_DATA_PATH.exists():
        return pd.read_parquet(REFERENCE_DATA_PATH)
    return pd.DataFrame()


def load_training_metrics() -> dict | None:
    """Load saved training metrics JSON."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────
#  Tabs
# ──────────────────────────────────────────────

tab_predict, tab_model, tab_monitor, tab_drift = st.tabs([
    "🎯 Simulador de Predição",
    "🧠 Sobre o Modelo",
    "📊 Monitoramento",
    "🔍 Detecção de Drift",
])

# ══════════════════════════════════════════════
#  TAB 1 — Simulador de Predição
# ══════════════════════════════════════════════

with tab_predict:
    st.header("🎯 Simulador de Predição")
    st.markdown(
        "Preencha os dados do estudante abaixo para simular uma predição de risco "
        "de defasagem escolar. A chamada é feita diretamente à API `/predict`."
    )

    with st.form("predict_form"):
        st.subheader("Indicadores Numéricos")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            iaa = st.number_input("IAA (Auto Avaliação)", 0.0, 10.0, 7.0, 0.1)
            ieg = st.number_input("IEG (Engajamento)", 0.0, 10.0, 6.5, 0.1)
        with col2:
            ips = st.number_input("IPS (Psicossocial)", 0.0, 10.0, 6.0, 0.1)
            ida = st.number_input("IDA (Desempenho Acadêmico)", 0.0, 10.0, 5.5, 0.1)
        with col3:
            ipv = st.number_input("IPV (Ponto de Virada)", 0.0, 10.0, 6.0, 0.1)
            matem = st.number_input("Matemática", 0.0, 10.0, 5.0, 0.1)
        with col4:
            portug = st.number_input("Português", 0.0, 10.0, 6.0, 0.1)

        st.subheader("Dados do Estudante")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            idade = st.number_input("Idade (em 2022)", 6, 25, 14, 1)
            ano_ingresso = st.number_input("Ano de Ingresso", 2016, 2022, 2020, 1)
        with col6:
            genero = st.selectbox("Gênero", ["Menina", "Menino"])
            instituicao = st.selectbox(
                "Instituição de Ensino",
                ["Escola Pública", "Rede Decisão", "Escola JP II"],
            )
        with col7:
            pedra = st.selectbox("Pedra 2022", ["Quartzo", "Ágata", "Ametista", "Topázio"])
            atingiu_pv = st.selectbox("Atingiu Ponto de Virada?", ["Não", "Sim"])
        with col8:
            indicado = st.selectbox("Indicado para bolsa?", ["Não", "Sim"])
            rec_psicologia = st.selectbox(
                "Rec. Psicologia",
                ["Sem limitações", "Não atendido", "Não indicado",
                 "Não avaliado", "Requer avaliação"],
            )

        st.subheader("Destaques")
        col9, col10, col11 = st.columns(3)
        with col9:
            destaque_ieg = st.selectbox(
                "Destaque IEG", ["Destaque", "Melhorar"], key="d_ieg",
            )
        with col10:
            destaque_ida = st.selectbox(
                "Destaque IDA", ["Destaque", "Melhorar"], key="d_ida",
            )
        with col11:
            destaque_ipv = st.selectbox(
                "Destaque IPV", ["Destaque", "Melhorar"], key="d_ipv",
            )

        submitted = st.form_submit_button(
            "🚀 Realizar Predição", use_container_width=True,
        )

    if submitted:
        payload = {
            "IAA": iaa, "IEG": ieg, "IPS": ips, "IDA": ida, "IPV": ipv,
            "Matem": matem, "Portug": portug,
            "Idade 22": idade, "Ano ingresso": ano_ingresso,
            "Gênero": genero,
            "Instituição de ensino": instituicao,
            "Pedra 22": pedra,
            "Atingiu PV": atingiu_pv,
            "Indicado": indicado,
            "Rec Psicologia": rec_psicologia,
            "Destaque IEG": f"{destaque_ieg}: texto exemplo.",
            "Destaque IDA": f"{destaque_ida}: texto exemplo.",
            "Destaque IPV": f"{destaque_ipv}: texto exemplo.",
        }

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                prob = result["probability"]
                risk = result["risk_level"]

                color_map = {"Baixo": "green", "Médio": "orange", "Alto": "red"}
                emoji_map = {"Baixo": "✅", "Médio": "⚠️", "Alto": "🚨"}

                st.markdown("---")
                st.subheader("Resultado da Predição")

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric(
                        "Predição",
                        "Em risco" if result["prediction"] == 1 else "Sem risco",
                    )
                with r2:
                    st.metric("Probabilidade de Risco", f"{prob:.1%}")
                with r3:
                    st.metric("Nível de Risco", f"{emoji_map[risk]} {risk}")

                st.info(result["message"])

                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%"},
                    title={"text": "Probabilidade de Defasagem"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color_map[risk]},
                        "steps": [
                            {"range": [0, 30], "color": "#d4edda"},
                            {"range": [30, 70], "color": "#fff3cd"},
                            {"range": [70, 100], "color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.8,
                            "value": prob * 100,
                        },
                    },
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 Payload enviado à API"):
                    st.json(payload)
                with st.expander("📋 Resposta da API"):
                    st.json(result)
            else:
                st.error(f"Erro na API (HTTP {resp.status_code}): {resp.text}")
        except requests.ConnectionError:
            st.error(
                "Não foi possível conectar à API. "
                "Verifique se o serviço está rodando (`docker-compose up`)."
            )
        except Exception as e:
            st.error(f"Erro inesperado: {e}")


# ══════════════════════════════════════════════
#  TAB 2 — Sobre o Modelo
# ══════════════════════════════════════════════

with tab_model:
    st.header("🧠 Sobre o Modelo")

    model, scaler = load_model_artifacts()
    metrics = load_training_metrics()

    # --- Model overview -------------------------------------------------
    st.subheader("Visão Geral")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown("""
        | Item | Detalhe |
        |------|---------|
        | **Problema** | Classificação binária — risco de defasagem escolar |
        | **Target** | `Defas_bin` (1 = em risco, 0 = adequado) |
        | **Modelo Selecionado** | Random Forest Classifier |
        | **Métrica de Seleção** | F1-Score (equilíbrio precisão/recall) |
        | **Serialização** | joblib |
        | **Scaling** | StandardScaler |
        """)
    with info_col2:
        st.markdown("""
        | Hiperparâmetro | Valor |
        |----------------|-------|
        | `n_estimators` | 200 |
        | `max_depth` | 10 |
        | `min_samples_split` | 5 |
        | `min_samples_leaf` | 2 |
        | `class_weight` | balanced |
        | `random_state` | 42 |
        """)

    # --- CV results -----------------------------------------------------
    st.subheader("Comparação de Modelos (Validação Cruzada 5-Fold)")

    # Default values (used when training_metrics.json is absent)
    cv_data = {
        "Modelo": ["Logistic Regression", "Random Forest ✅", "Gradient Boosting"],
        "F1-Score": [0.858, 0.913, 0.906],
        "Recall": [0.818, 0.952, 0.942],
        "Precision": [0.901, 0.878, 0.873],
        "AUC-ROC": [0.893, 0.898, 0.908],
        "Accuracy": [0.872, 0.894, 0.892],
    }
    if metrics and "cv_results" in metrics:
        cv = metrics["cv_results"]
        # Only override defaults if the saved CV data has the expected keys
        first_model_metrics = next(iter(cv.values()), {})
        required_keys = {"f1", "recall", "precision", "roc_auc", "accuracy"}
        if required_keys.issubset(first_model_metrics.keys()):
            cv_data = {
                "Modelo": [], "F1-Score": [], "Recall": [],
                "Precision": [], "AUC-ROC": [], "Accuracy": [],
            }
            for name, m in cv.items():
                is_best = name == metrics.get("best_model", "")
                cv_data["Modelo"].append(f"{name} ✅" if is_best else name)
                cv_data["F1-Score"].append(round(np.mean(m["f1"]), 4))
                cv_data["Recall"].append(round(np.mean(m["recall"]), 4))
                cv_data["Precision"].append(round(np.mean(m["precision"]), 4))
                cv_data["AUC-ROC"].append(round(np.mean(m["roc_auc"]), 4))
                cv_data["Accuracy"].append(round(np.mean(m["accuracy"]), 4))

    cv_df = pd.DataFrame(cv_data)
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

    # Bar chart comparison
    fig = go.Figure()
    for metric_name in ["F1-Score", "Recall", "AUC-ROC"]:
        fig.add_trace(go.Bar(
            name=metric_name,
            x=cv_df["Modelo"],
            y=cv_df[metric_name],
            text=cv_df[metric_name].apply(lambda v: f"{v:.3f}"),
            textposition="auto",
        ))
    fig.update_layout(
        title="Comparação de Modelos — Métricas de Validação Cruzada",
        barmode="group",
        yaxis=dict(range=[0.7, 1.0]),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Test set metrics -----------------------------------------------
    st.subheader("Métricas no Conjunto de Teste (20%)")

    test_metrics = {
        "Accuracy": 0.837, "F1-Score": 0.888, "Recall": 0.925,
        "Precision": 0.854, "AUC-ROC": 0.917,
    }
    if metrics and "test_metrics" in metrics:
        tm = metrics["test_metrics"]
        test_metrics = {
            "Accuracy": tm.get("accuracy", 0),
            "F1-Score": tm.get("f1", 0),
            "Recall": tm.get("recall", 0),
            "Precision": tm.get("precision", 0),
            "AUC-ROC": tm.get("roc_auc", 0),
        }

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Accuracy", f"{test_metrics['Accuracy']:.3f}")
    with m2:
        st.metric("F1-Score", f"{test_metrics['F1-Score']:.3f}")
    with m3:
        st.metric("Recall", f"{test_metrics['Recall']:.3f}")
    with m4:
        st.metric("Precision", f"{test_metrics['Precision']:.3f}")
    with m5:
        st.metric("AUC-ROC", f"{test_metrics['AUC-ROC']:.3f}")

    # --- Dataset info ---------------------------------------------------
    if metrics and "dataset" in metrics:
        ds = metrics["dataset"]
        st.subheader("Dados de Treinamento")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Total de Amostras", ds.get("total_samples", "—"))
        with d2:
            st.metric("Amostras de Treino", ds.get("train_samples", "—"))
        with d3:
            st.metric("Amostras de Teste", ds.get("test_samples", "—"))

    # --- Feature importance ---------------------------------------------
    st.subheader("Importância das Features")
    if model is not None and hasattr(model, "feature_importances_"):
        from src.utils.constants import ALL_FEATURES

        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": ALL_FEATURES,
            "Importância": importances,
        }).sort_values("Importância", ascending=True)

        fig = px.bar(
            feat_df, x="Importância", y="Feature",
            orientation="h",
            title="Feature Importances (Random Forest)",
            color="Importância",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Modelo não disponível para extrair importância das features.")

    # --- Why reliable ---------------------------------------------------
    st.subheader("Por que este modelo é confiável?")
    st.markdown("""
    1. **Recall alto (92.5%)** — Em contexto social, deixar de identificar um aluno em risco
       tem consequências sérias. O modelo minimiza falsos negativos.
    2. **Sem data leakage** — Features que codificam diretamente o target (`IAN`, `INDE 22`,
       `Fase`) foram removidas após análise de correlação.
    3. **Validação robusta** — F1-Score de 0.913 (±0.023) em validação cruzada 5-fold com
       estratificação demonstra consistência.
    4. **Features interpretáveis** — Os principais preditores (Idade, Pedra, Matemática, IDA)
       são indicadores conhecidos pelos educadores.
    """)

    # --- Features used --------------------------------------------------
    with st.expander("📋 Features utilizadas no modelo (18)"):
        st.markdown(
            "**Numéricas (9):** IAA, IEG, IPS, IDA, IPV, Matem, Portug, "
            "Idade 22, Anos_no_programa"
        )
        st.markdown(
            "**Categóricas (9):** Gênero, Instituição de ensino, Pedra 22, "
            "Atingiu PV, Indicado, Rec Psicologia, Destaque IEG_bin, "
            "Destaque IDA_bin, Destaque IPV_bin"
        )
        st.markdown("---")
        st.markdown(
            "**Features excluídas (data leakage):** IAN (corr=-0.98), "
            "INDE 22 (contém IAN), Fase (reconstrói target com Idade), "
            "Inglês (67% nulos)"
        )


# ══════════════════════════════════════════════
#  TAB 3 — Monitoramento
# ══════════════════════════════════════════════

with tab_monitor:
    st.header("📊 Monitoramento de Predições")

    predictions_df = load_predictions()

    if predictions_df.empty:
        st.info(
            "Nenhuma predição registrada ainda. Use o **Simulador de Predição** "
            "ou faça chamadas à API para gerar dados."
        )
    else:
        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Predições", len(predictions_df))
        with col2:
            risk_count = (
                predictions_df["prediction"].sum()
                if "prediction" in predictions_df.columns
                else 0
            )
            total = len(predictions_df)
            st.metric(
                "Alunos em Risco", f"{int(risk_count)} ({risk_count / total:.0%})",
            )
        with col3:
            if "probability" in predictions_df.columns:
                avg_prob = predictions_df["probability"].mean()
                st.metric("Probabilidade Média", f"{avg_prob:.2%}")
        with col4:
            if "latency_ms" in predictions_df.columns:
                avg_latency = predictions_df["latency_ms"].mean()
                st.metric("Latência Média", f"{avg_latency:.1f}ms")

        # Charts
        st.subheader("Distribuição de Predições")
        col_left, col_right = st.columns(2)

        with col_left:
            if "risk_level" in predictions_df.columns:
                risk_dist = predictions_df["risk_level"].value_counts()
                fig = px.pie(
                    values=risk_dist.values,
                    names=risk_dist.index,
                    title="Distribuição por Nível de Risco",
                    color=risk_dist.index,
                    color_discrete_map={
                        "Baixo": "#2ecc71",
                        "Médio": "#f39c12",
                        "Alto": "#e74c3c",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            if "probability" in predictions_df.columns:
                fig = px.histogram(
                    predictions_df, x="probability", nbins=20,
                    title="Distribuição de Probabilidades",
                    labels={"probability": "Probabilidade de Risco"},
                    color_discrete_sequence=["#3498db"],
                )
                st.plotly_chart(fig, use_container_width=True)

        # Timeline
        if "timestamp" in predictions_df.columns and len(predictions_df) > 1:
            st.subheader("Predições ao Longo do Tempo")
            daily = (
                predictions_df.set_index("timestamp")
                .resample("h")["prediction"]
                .agg(["count", "mean"])
            )
            daily.columns = ["count", "risk_rate"]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily.index, y=daily["count"],
                name="Total requests", marker_color="#3498db",
            ))
            fig.add_trace(go.Scatter(
                x=daily.index, y=daily["risk_rate"],
                name="Taxa de risco", yaxis="y2",
                line=dict(color="#e74c3c", width=2),
            ))
            fig.update_layout(
                title="Volume de Requisições e Taxa de Risco",
                yaxis=dict(title="Número de requests"),
                yaxis2=dict(
                    title="Taxa de risco", overlaying="y",
                    side="right", range=[0, 1],
                ),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Latency over time
        if (
            "latency_ms" in predictions_df.columns
            and "timestamp" in predictions_df.columns
        ):
            st.subheader("Latência ao Longo do Tempo")
            fig = px.scatter(
                predictions_df, x="timestamp", y="latency_ms",
                title="Latência por Requisição",
                labels={"latency_ms": "Latência (ms)", "timestamp": "Horário"},
                color_discrete_sequence=["#9b59b6"],
            )
            fig.add_hline(
                y=predictions_df["latency_ms"].mean(),
                line_dash="dash", line_color="red",
                annotation_text=(
                    f"Média: {predictions_df['latency_ms'].mean():.1f}ms"
                ),
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4 — Detecção de Drift
# ══════════════════════════════════════════════

with tab_drift:
    st.header("🔍 Detecção de Drift")

    predictions_df = load_predictions()
    reference_df = load_reference()

    if reference_df.empty:
        st.warning(
            "Dados de referência não encontrados. Execute o treinamento do modelo.",
        )
    elif predictions_df.empty:
        st.info(
            "Sem dados de produção para comparar. "
            "Faça predições via Simulador ou API.",
        )
    else:
        feature_cols = [c for c in reference_df.columns if c != "target"]

        if "input" in predictions_df.columns:
            try:
                prod_features = pd.json_normalize(predictions_df["input"])
                common_cols = [
                    c for c in feature_cols if c in prod_features.columns
                ]

                if common_cols:
                    selected_feature = st.selectbox(
                        "Selecione uma feature para comparar:", common_cols,
                    )

                    ref_values = reference_df[selected_feature].dropna()
                    prod_values = prod_features[selected_feature].dropna()

                    if (
                        len(prod_values) > 0
                        and ref_values.dtype in ["float64", "int64"]
                    ):
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(
                            x=ref_values, name="Referência (treino)",
                            opacity=0.7, marker_color="#3498db",
                        ))
                        fig.add_trace(go.Histogram(
                            x=prod_values, name="Produção",
                            opacity=0.7, marker_color="#e74c3c",
                        ))
                        fig.update_layout(
                            title=f"Distribuição: {selected_feature}",
                            barmode="overlay", height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Stats comparison table
                        stats_col1, stats_col2 = st.columns(2)
                        with stats_col1:
                            st.markdown("**Referência (treino)**")
                            st.dataframe(
                                ref_values.describe().to_frame("valor").T,
                                use_container_width=True,
                            )
                        with stats_col2:
                            st.markdown("**Produção**")
                            st.dataframe(
                                prod_values.describe().to_frame("valor").T,
                                use_container_width=True,
                            )
            except Exception as e:
                st.error(f"Erro ao processar dados de produção: {e}")

        # Evidently report
        drift_report_path = LOGS_DIR / "drift_report.html"
        if st.button("🔬 Gerar Relatório Completo de Drift (Evidently)"):
            try:
                from src.monitoring.drift_detector import generate_drift_report

                with st.spinner("Gerando relatório de drift..."):
                    result = generate_drift_report()
                if "error" not in result:
                    st.success(
                        f"Drift detectado: "
                        f"**{'Sim' if result.get('dataset_drift') else 'Não'}** | "
                        f"Colunas com drift: "
                        f"{result.get('number_of_drifted_columns', 0)}/"
                        f"{result.get('number_of_columns', 0)}"
                    )
                    if result.get("drifted_columns"):
                        st.warning(
                            f"Colunas com drift: "
                            f"{', '.join(result['drifted_columns'])}"
                        )
                else:
                    st.warning(result["error"])
            except Exception as e:
                st.error(f"Erro ao gerar relatório: {e}")

        if drift_report_path.exists():
            with open(drift_report_path, "r") as f:
                st.components.v1.html(f.read(), height=800, scrolling=True)
