import streamlit as st
from pathlib import Path

# Задаем путь к текущему каталогу
dir_path = Path(__file__).parent

def run():
    page = st.navigation(
        [
            st.Page(dir_path / "main_page.py", icon="🏠", title="Главное меню"),
            st.Page(dir_path / "eda_page.py", icon="📊", title="Анализ данных"),
            st.Page(dir_path / "model_page.py", icon="🛠️", title="Работа с моделью"),
        ]
    )

    page.run()

if __name__ == "__main__":
    run()
