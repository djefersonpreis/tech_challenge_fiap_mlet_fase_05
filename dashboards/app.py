"""Streamlit dashboard for model monitoring and drift detection."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
REFERENCE_DATA_PATH = MODELS_DIR / "reference_data.parquet"

st.set_page_config(
    page_title="Passos Mágicos - Monitoramento",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 Passos Mágicos - Monitoramento do Modelo")
st.markdown("Dashboard de acompanhamento de predições e detecção de drift.")


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


# --- Sidebar ---
st.sidebar.header("Filtros")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)

# --- Load data ---
predictions_df = load_predictions()
reference_df = load_reference()

# --- Metrics ---
st.header("📊 Métricas de Uso")

if predictions_df.empty:
    st.info("Nenhuma predição registrada ainda. Faça chamadas à API para ver dados aqui.")
else:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Predições", len(predictions_df))

    with col2:
        risk_count = predictions_df["prediction"].sum() if "prediction" in predictions_df.columns else 0
        st.metric("Alunos em Risco", int(risk_count))

    with col3:
        if "probability" in predictions_df.columns:
            avg_prob = predictions_df["probability"].mean()
            st.metric("Probabilidade Média", f"{avg_prob:.2%}")

    with col4:
        if "latency_ms" in predictions_df.columns:
            avg_latency = predictions_df["latency_ms"].mean()
            st.metric("Latência Média", f"{avg_latency:.1f}ms")

    # --- Distribution charts ---
    st.header("📈 Distribuição de Predições")

    col_left, col_right = st.columns(2)

    with col_left:
        if "risk_level" in predictions_df.columns:
            risk_dist = predictions_df["risk_level"].value_counts()
            fig = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                title="Distribuição por Nível de Risco",
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
                predictions_df,
                x="probability",
                nbins=20,
                title="Distribuição de Probabilidades",
                labels={"probability": "Probabilidade de Risco"},
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Timeline ---
    if "timestamp" in predictions_df.columns and len(predictions_df) > 1:
        st.header("📅 Predições ao Longo do Tempo")
        daily = predictions_df.set_index("timestamp").resample("h")["prediction"].agg(["count", "mean"])
        daily.columns = ["count", "risk_rate"]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily.index, y=daily["count"], name="Total requests"))
        fig.add_trace(go.Scatter(x=daily.index, y=daily["risk_rate"], name="Taxa de risco", yaxis="y2"))
        fig.update_layout(
            title="Volume de Requisições e Taxa de Risco",
            yaxis=dict(title="Número de requests"),
            yaxis2=dict(title="Taxa de risco", overlaying="y", side="right", range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Drift Detection ---
st.header("🔍 Detecção de Drift")

if reference_df.empty:
    st.warning("Dados de referência não encontrados. Execute o treinamento do modelo.")
elif predictions_df.empty:
    st.info("Sem dados de produção para comparar. Faça predições via API.")
else:
    # Compare feature distributions
    feature_cols = [c for c in reference_df.columns if c != "target"]

    # Extract input features from predictions
    if "input" in predictions_df.columns:
        try:
            prod_features = pd.json_normalize(predictions_df["input"])
            common_cols = [c for c in feature_cols if c in prod_features.columns]

            if common_cols:
                selected_feature = st.selectbox("Selecione uma feature para comparar:", common_cols)

                ref_values = reference_df[selected_feature].dropna()
                prod_values = prod_features[selected_feature].dropna()

                if len(prod_values) > 0 and ref_values.dtype in ["float64", "int64"]:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=ref_values, name="Referência (treino)", opacity=0.7))
                    fig.add_trace(go.Histogram(x=prod_values, name="Produção", opacity=0.7))
                    fig.update_layout(
                        title=f"Distribuição: {selected_feature}",
                        barmode="overlay",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao processar dados de produção: {e}")

    # Evidently report button
    drift_report_path = LOGS_DIR / "drift_report.html"
    if st.button("Gerar Relatório Completo de Drift (Evidently)"):
        try:
            from src.monitoring.drift_detector import generate_drift_report

            result = generate_drift_report()
            if "error" not in result:
                st.success(
                    f"Drift detectado: {'Sim' if result.get('dataset_drift') else 'Não'} | "
                    f"Colunas com drift: {result.get('number_of_drifted_columns', 0)}/{result.get('number_of_columns', 0)}"
                )
                if result.get("drifted_columns"):
                    st.warning(f"Colunas com drift: {', '.join(result['drifted_columns'])}")
            else:
                st.warning(result["error"])
        except Exception as e:
            st.error(f"Erro ao gerar relatório: {e}")

    if drift_report_path.exists():
        with open(drift_report_path, "r") as f:
            st.components.v1.html(f.read(), height=800, scrolling=True)
