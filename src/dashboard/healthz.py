"""
Endpoint для проверки здоровья Dashboard приложения
"""
import streamlit as st
from datetime import datetime
import json

def health_check():
    """Простая проверка здоровья для Docker healthcheck"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "diagmod-dashboard",
        "version": "1.0.0"
    }

# Создаем простую страницу для healthcheck
if __name__ == "__main__":
    st.write(json.dumps(health_check(), indent=2))
