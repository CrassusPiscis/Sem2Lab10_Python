from pydantic import BaseModel, Field
from typing import List


class LoanPredictionInput(BaseModel):
    """схема валидации для одного клиента"""
    person_age: float = Field(..., description="Возраст", ge=18, le=100)
    person_gender: str = Field(..., description="Пол")
    person_education: str = Field(..., description="Образование")
    person_income: float = Field(..., description="Доход в год", ge=0)
    person_emp_exp: int = Field(..., description="Стаж", ge=0, le=60)
    person_home_ownership: str = Field(..., description="Жилье")
    loan_amnt: float = Field(..., description="Сумма кредита", ge=0)
    loan_intent: str = Field(..., description="Цель кредита")
    loan_int_rate: float = Field(..., description="Процентная ставка", ge=0)
    loan_percent_income: float = Field(..., description="Доля кредита от годового дохода", ge=0, le=1)
    cb_person_cred_hist_length: float = Field(..., description="Продолжительность кредитной истории в годах", ge=0)
    credit_score: int = Field(..., description="Кредитный рейтинг скоринга", ge=300, le=850)
    previous_loan_defaults_on_file: str = Field(..., description="Были ли дефолты ранее (Y/N)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "person_age": 24.0,
                "person_gender": "female",
                "person_education": "Bachelor",
                "person_income": 47204.0,
                "person_emp_exp": 1,
                "person_home_ownership": "RENT",
                "loan_amnt": 5000.0,
                "loan_intent": "PERSONAL",
                "loan_int_rate": 8.59,
                "loan_percent_income": 0.07,
                "cb_person_cred_hist_length": 3.0,
                "credit_score": 601,
                "previous_loan_defaults_on_file": "N"
            }
        }
    }


class PredictRequest(BaseModel):
    """схема для POST /predict. Принимает массив клиентов"""
    objects: List[LoanPredictionInput]


class PredictionOutput(BaseModel):
    """схема ответа для объекта (включает признаки и предсказание)"""
    features: LoanPredictionInput
    loan_status: int = Field(..., description="Статус одобрения: 1 - одобрено, 0 - отказ")
    loan_probability: float = Field(..., description="Вероятность одобрения от 0 до 1", ge=0, le=1)


class PredictResponse(BaseModel):
    """Схема ответа со списком результатов предсказаний"""
    predictions: List[PredictionOutput]