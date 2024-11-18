import streamlit as st
import pandas as pd
import joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objects as go
from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch.distributions.studentT import StudentTOutput
import os
import sys
sys.path.append("lag-llama")
from lag_llama.gluon.estimator import LagLlamaEstimator
import lightgbm as lgb

# Установка фиксированного сида
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.serialization.add_safe_globals([StudentTOutput])

# Настройка Streamlit
st.set_page_config(page_title="Работа с моделью", page_icon=":material/show_chart:")
st.title("Работа с моделью")

st.write("Загрузите Excel файл с тестовыми данными для прогнозирования:")

# Загрузка файла
uploaded_file = st.file_uploader("Выберите файл", type=["xlsx"])

# Если файл загружен, сохраняем его в сессии
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file  # Сохраняем файл в сессии
elif 'uploaded_file' in st.session_state:
    uploaded_file = st.session_state.uploaded_file  # Загружаем файл из сессии

# Функция для загрузки данных
def load_data(df):
    current_values = df.iloc[:, 1].values.astype(float)
    return current_values

# Функция для получения предсказаний с использованием Lag-Llama
def get_lag_llama_predictions(data, prediction_length, device, context_length=32, use_rope_scaling=False, num_samples=100):
    df = pd.DataFrame({"item_id": ["series_1"] * len(data), "target": data.astype(np.float32)})
    dataset = PandasDataset.from_long_dataframe(df, target="target", item_id="item_id")

    # Определение пути к файлу lag-llama.ckpt
    ckpt_path = os.path.join("lag-llama", "lag-llama.ckpt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }

    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,
        batch_size=1,
        num_parallel_samples=num_samples,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, _ = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(forecast_it)
    return forecasts

# Обработка загруженных данных
if uploaded_file is not None:
    # Чтение данных из Excel
    test_df = pd.read_excel(uploaded_file)
    st.write("Данные успешно загружены.")

    # Проверка наличия необходимых колонок
    if test_df.shape[1] >= 2:
        current_values = load_data(test_df)
        # ---- Классификация режима работы с помощью LightGBM ----
        st.subheader("Классификация режима работы")
        
        # Создание нового MinMaxScaler для LightGBM
        scaler_class = joblib.load('scaler.pkl')  # Используем импортированный скейлер для анализа режима
        loaded_model = lgb.Booster(model_file='model.txt')

        MAX_LENGTH = 13162
        original_length = len(current_values)

        if original_length < MAX_LENGTH:
            current_values_padded = np.pad(current_values, (0, MAX_LENGTH - original_length), 'constant')
        else:
            current_values_padded = current_values[:MAX_LENGTH]

        current_values_reshaped = current_values_padded.reshape(1, -1)
        test_data_scaled = scaler_class.transform(current_values_reshaped)

        pred_prob = loaded_model.predict(test_data_scaled)[0]
        threshold = 0.5
        pred_label = "Нормальный режим" if pred_prob < threshold else "Аварийный режим"
        
        st.write(f"Режим: **{pred_label}**")
        st.write(f"Вероятность аварийного режима: {pred_prob:.2f}")


        # ---- Прогнозирование временного ряда с помощью Lag-Llama ----
        st.subheader("Прогнозирование временного ряда")
        
        # Создание нового стандартного скейлера для Lag-Llama
        scaler = StandardScaler()
        current_values_scaled = scaler.fit_transform(current_values.reshape(-1, 1)).flatten().astype(np.float32)

        prediction_length = 24  # Длина прогноза
        num_samples = 100
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Получение предсказаний
        forecasts = get_lag_llama_predictions(current_values_scaled, prediction_length, device, num_samples=num_samples)

        # Рескейлим предсказанные значения
        predicted_values = np.concatenate([forecast.samples[0] for forecast in forecasts])
        extended_series_rescaled = scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()

        # График прогноза временного ряда
        fig = go.Figure()

        last_index = len(current_values) - 1

        fig.add_trace(go.Scatter(
            x=np.arange(last_index + 1), 
            y=current_values, 
            mode='lines', 
            name='Истинные значения', 
            line=dict(color='blue')
        ))

        predicted_values = [current_values[last_index]] + list(extended_series_rescaled)
        forecast_index = np.arange(last_index, last_index + prediction_length + 1)

        fig.add_trace(go.Scatter(
            x=forecast_index, 
            y=predicted_values, 
            mode='lines', 
            name='Прогноз', 
            line=dict(color='green')
        ))

        fig.update_layout(
            title="Прогноз временного ряда",
            xaxis_title="Индекс",
            yaxis_title="Значение",
            showlegend=True
        )

        st.plotly_chart(fig)

    else:
        st.warning("В загруженных данных недостаточно колонок. Убедитесь, что файл содержит хотя бы два столбца.")
else:
    st.info("Пожалуйста, загрузите файл для прогнозирования.")
