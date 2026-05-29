import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Загрузка датасета и очистка от аномалий"""
    logging.info(f"загрузка данных  ({filepath})")
    df = pd.read_csv(filepath)

    # фильтрация выбросов
    initial_count = len(df)
    df = df[(df['person_age'] < 100) & (df['person_emp_exp'] < 60)]
    logging.info(f"Удалено аномальных строк: {initial_count - len(df)}")
    return df


def build_pipeline(numeric_cols: list, categorical_cols: list) -> Pipeline:
    """создание пайплайна предобработки и модели."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    return pipeline


def main():
    # определение признаков
    target = 'loan_status'
    numeric_features = [
        'person_age', 'person_income', 'person_emp_exp',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length', 'credit_score'
    ]
    categorical_features = [
        'person_gender', 'person_education',
        'person_home_ownership', 'loan_intent',
        'previous_loan_defaults_on_file'
    ]

    # подготовка данных
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data(str(DATA_DIR / "loan_data.csv"))
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # обучение
    logging.info("инициализация и обучение пайплайна Random Forest...")
    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    # оценка работы
    proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    logging.info(f"Тестовый ROC-AUC: {roc_auc:.4f}")

    # сохранение артефактов
    model_path = MODELS_DIR / "mortgage_pipeline.pkl"
    joblib.dump(pipeline, model_path)
    logging.info("пайплайн Random Forest сохранен в '%s'", model_path)

    # генерация тестового json-файла
    sample_client = X_test.iloc[0].to_dict()
    with open(DATA_DIR / 'sample_single_client.json', 'w', encoding='utf-8') as f:
        json.dump(sample_client, f, ensure_ascii=False, indent=4)

    # генерация тестовых csv файлов для валидации API методов
    test_with_target = X_test.copy()
    test_with_target[target] = y_test
    test_with_target.head(20).to_csv(DATA_DIR / 'test_with_target.csv', index=False)
    X_test.head(20).to_csv(DATA_DIR / 'test_without_target.csv', index=False)
    logging.info("Тестовые артефакты для API успешно обновлены.")


if __name__ == "__main__":
    main()