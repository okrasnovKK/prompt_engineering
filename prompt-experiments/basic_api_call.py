import requests
import json
from config import GIGACHAT_API_URL, AUTH_TOKEN

"""
Базовый запрос к GigaChat API
Демонстрирация:
- Авторизации
- Отправки запроса
- Обработки ответа
"""




def call_gigachat(prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:

    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "GigaChat",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 512
    }

    response = requests.post(GIGACHAT_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"


if __name__ == "__main__":
    # Тестовый запрос
    system = "Ты — полезный ассистент. Отвечай кратко и по делу."
    user = "Сколько дней в неделе?"

    answer = call_gigachat(user, system, temperature=0.3)
    print(f"Ответ: {answer}")
