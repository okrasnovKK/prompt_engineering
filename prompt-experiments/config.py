import os

"""
Конфигурация для работы с GigaChat API
"""

GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "repetition_penalty": 1.0
}

#Необходимо предварительное создание файла .env
AUTH_TOKEN = os.getenv("GIGACHAT_TOKEN", "your_token_here")
