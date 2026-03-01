"""Pydantic schemas for the API request/response models."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input schema for the prediction endpoint.

    Contains the raw feature values for a student.
    """

    iaa: float = Field(..., alias="IAA", description="Indicador de Auto Avaliação")
    ieg: float = Field(..., alias="IEG", description="Indicador de Engajamento")
    ips: float = Field(..., alias="IPS", description="Indicador Psicossocial")
    ida: float = Field(..., alias="IDA", description="Indicador de Desempenho Acadêmico")
    ipv: float = Field(..., alias="IPV", description="Indicador de Ponto de Virada")
    matem: float = Field(..., alias="Matem", description="Nota de Matemática")
    portug: float = Field(..., alias="Portug", description="Nota de Português")
    idade_22: int = Field(..., alias="Idade 22", description="Idade do aluno em 2022")
    ano_ingresso: int = Field(..., alias="Ano ingresso", description="Ano de ingresso no programa")
    genero: str = Field(..., alias="Gênero", description="Gênero: Menina ou Menino")
    instituicao: str = Field(..., alias="Instituição de ensino", description="Instituição de ensino")
    pedra_22: str = Field(..., alias="Pedra 22", description="Classificação Pedra 2022")
    atingiu_pv: str = Field(..., alias="Atingiu PV", description="Atingiu Ponto de Virada: Sim/Não")
    indicado: str = Field(..., alias="Indicado", description="Indicado para bolsa: Sim/Não")
    rec_psicologia: str = Field(..., alias="Rec Psicologia", description="Recomendação da Psicologia")
    destaque_ieg: str = Field(..., alias="Destaque IEG", description="Destaque ou Melhorar em IEG")
    destaque_ida: str = Field(..., alias="Destaque IDA", description="Destaque ou Melhorar em IDA")
    destaque_ipv: str = Field(..., alias="Destaque IPV", description="Destaque ou Melhorar em IPV")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    """Output schema for the prediction endpoint."""

    prediction: int = Field(..., description="0 = sem risco de defasagem, 1 = risco de defasagem")
    probability: float = Field(..., description="Probabilidade de risco de defasagem (0.0 - 1.0)")
    risk_level: str = Field(..., description="Nível de risco: Baixo, Médio ou Alto")
    message: str = Field(..., description="Descrição textual do resultado")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool = False
    version: str = "1.0.0"
