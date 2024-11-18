import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import os

st.set_page_config(page_title="Исследовательский анализ данных", page_icon=":bar_chart:")
st.title("Исследовательский анализ данных (EDA)")

@st.cache_data
def load_data():
    folders = ['Аварийный режим', 'Нормальный режим']
    data = []
    labels = []
    for folder in folders:
        label = 1 if folder == 'Аварийный режим' else 0
        folder_path = os.path.join(os.getcwd(), folder)
        for file in os.listdir(folder_path):
            if file.endswith('.xlsx'):
                file_path = os.path.join(folder_path, file)
                df = pd.read_excel(file_path, header=0)
                current_values = df.iloc[:, 1].values.astype(float)
                data.append(current_values)
                labels.append(label)
    return data, labels

# Загрузка данных
data, labels = load_data()

max_length = max(len(series) for series in data) # Находим максимальную длину временного ряда
data_padded = np.array([np.pad(series, (0, max_length - len(series)), 'constant') for series in data]) # Дополняем каждый временной ряд нулями до максимальной длины
data_df = pd.DataFrame(data_padded, columns=[f'Value_{i+1}' for i in range(data_padded.shape[1])]) # Создаем DataFrame
data_df['Label'] = labels # Добавляем колонку с метками

# Функция для создания графиков с кэшированием
@st.cache_data
def create_mode_counts_chart():
    mode_counts = pd.Series(labels).value_counts()
    fig_mode_counts = go.Figure(data=[
        go.Bar(x=["Аварийный режим", "Нормальный режим"], y=mode_counts, marker_color=['red', 'blue'])
    ])
    fig_mode_counts.update_layout(yaxis_title="Количество")
    return fig_mode_counts

@st.cache_data
def create_distribution_chart():
    normal_series = np.concatenate([series for series, label in zip(data, labels) if label == 0])
    emergency_series = np.concatenate([series for series, label in zip(data, labels) if label == 1])
    fig_distribution = go.Figure()
    fig_distribution.add_trace(go.Histogram(x=normal_series, nbinsx=100, name="Нормальный режим", marker_color='blue', opacity=0.5))
    fig_distribution.add_trace(go.Histogram(x=emergency_series, nbinsx=100, name="Аварийный режим", marker_color='red', opacity=0.5))
    fig_distribution.update_layout(xaxis_title="Значение тока", yaxis_title="Частота", barmode='overlay')
    return fig_distribution

@st.cache_data
def create_kde_chart():
    normal_series = np.concatenate([series for series, label in zip(data, labels) if label == 0])
    emergency_series = np.concatenate([series for series, label in zip(data, labels) if label == 1])
    kde_normal = gaussian_kde(normal_series)
    kde_emergency = gaussian_kde(emergency_series)
    x_normal = np.linspace(min(normal_series), max(normal_series), 1000)
    x_emergency = np.linspace(min(emergency_series), max(emergency_series), 1000)
    fig_kde = go.Figure()
    fig_kde.add_trace(go.Scatter(x=x_normal, y=kde_normal(x_normal), mode='lines', fill='tozeroy', name="Нормальный режим", line=dict(color='blue')))
    fig_kde.add_trace(go.Scatter(x=x_emergency, y=kde_emergency(x_emergency), mode='lines', fill='tozeroy', name="Аварийный режим", line=dict(color='red')))
    fig_kde.update_layout(xaxis_title="Значение тока", yaxis_title="Плотность")
    return fig_kde

@st.cache_data
def create_mean_values_chart():
    normal_data = data_padded[np.array(labels) == 0]
    emergency_data = data_padded[np.array(labels) == 1]
    mean_normal = np.mean(normal_data, axis=0)
    mean_emergency = np.mean(emergency_data, axis=0)
    fig_mean_values = go.Figure()
    fig_mean_values.add_trace(go.Scatter(x=np.arange(len(mean_normal)), y=mean_normal, mode='lines', name="Среднее - Нормальный режим", line=dict(color='blue')))
    fig_mean_values.add_trace(go.Scatter(x=np.arange(len(mean_emergency)), y=mean_emergency, mode='lines', name="Среднее - Аварийный режим", line=dict(color='red')))
    fig_mean_values.update_layout(xaxis_title="Временные шаги", yaxis_title="Среднее значение тока")
    return fig_mean_values

@st.cache_data
def create_box_plot():
    normal_series = np.concatenate([series for series, label in zip(data, labels) if label == 0])
    emergency_series = np.concatenate([series for series, label in zip(data, labels) if label == 1])
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=normal_series, name="Нормальный режим", marker_color='blue'))
    fig_box.add_trace(go.Box(y=emergency_series, name="Аварийный режим", marker_color='red'))
    fig_box.update_layout(yaxis_title="Значение тока")
    return fig_box

@st.cache_data
def create_violin_plot():
    normal_series = np.concatenate([series for series, label in zip(data, labels) if label == 0])
    emergency_series = np.concatenate([series for series, label in zip(data, labels) if label == 1])
    fig_violin = go.Figure()
    fig_violin.add_trace(go.Violin(y=normal_series, name="Нормальный режим", line_color='blue', box_visible=True, meanline_visible=True))
    fig_violin.add_trace(go.Violin(y=emergency_series, name="Аварийный режим", line_color='red', box_visible=True, meanline_visible=True))
    fig_violin.update_layout(yaxis_title="Значение тока")
    return fig_violin

# Визуализация графиков
st.subheader("Количество наблюдений в каждом режиме")
st.plotly_chart(create_mode_counts_chart())
st.markdown("Диаграмма показывает распределение количества наблюдений по каждому режиму. Обнаруженный дисбаланс может повлиять на модель, так как режим с меньшим количеством данных может быть хуже предсказан.")

st.subheader("Распределение значений тока по режимам")
st.plotly_chart(create_distribution_chart())
st.markdown("Гистограмма распределения значений тока позволяет увидеть разницу в диапазоне значений для каждого режима. Широкое распределение может указывать на больший разброс значений тока в одном из режимов.")

st.subheader("Оценка плотности ядра (KDE) по режимам")
st.plotly_chart(create_kde_chart())
st.markdown("Оценка плотности ядра показывает вероятностное распределение значений тока по каждому режиму. Можно сделать вывод о пересечении распределений, что указывает на наличие схожих значений тока в обоих режимах при определенных диапазонах.")

st.subheader("Средние значения по временным шагам для каждого режима")
st.plotly_chart(create_mean_values_chart())
st.markdown("Диаграмма средних значений тока во времени позволяет выявить тренд в изменении значений тока. Например, снижение средних значений может свидетельствовать об уменьшении активности или нагрузки на оборудование.")

st.subheader("Диаграмма размаха значений тока по режимам")
st.plotly_chart(create_box_plot())
st.markdown("Диаграмма размаха показывает распределение значений тока и выбросы в каждом режиме. Наличие большого количества выбросов в одном из режимов может свидетельствовать о нестабильности в работе оборудования.")

st.subheader("Вайолин-плот значений тока по режимам")
st.plotly_chart(create_violin_plot())
st.markdown("Вайолин-плот показывает распределение значений тока и плотность наблюдений по каждому диапазону. Визуализация позволяет оценить симметрию и наличие пиков в распределении, что может помочь в анализе вариативности токов для разных режимов.")

