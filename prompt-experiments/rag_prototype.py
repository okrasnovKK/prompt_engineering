import requests
from config import GIGACHAT_API_URL, AUTH_TOKEN


class SimpleRAG:
    """
    Простая RAG-система.
    """

    def __init__(self):
        """Инициализация: создаем пустой список документов"""
        self.documents = []  # здесь будем хранить наши документы

    def add_documents(self, documents):
        """
        Добавляет документы в базу знаний.
        documents: список строк (текстов документов)
        """
        self.documents = documents
        print(f"Добавлено {len(documents)} документов")

    def search(self, query):
        """
        Ищет документы, релевантные запросу.
        Для простоты ищем по ключевым словам (в реальном RAG используют эмбеддинги).

        query: вопрос пользователя
        возвращает: список найденных документов
        """
        #todo: разобрать использование эмбеддингов

        # Разбиваем вопрос на ключевые слова (убираем предлоги, оставляем значимые слова)
        stop_words = ['сколько', 'как', 'где', 'когда', 'что', 'это', 'ли', 'же', 'бы', 'то']
        keywords = [word.lower() for word in query.split() if word.lower() not in stop_words]

        print(f"Ищем по ключевым словам: {keywords}")

        # Ищем документы, где встречаются ключевые слова
        found_docs = []
        for doc in self.documents:
            doc_lower = doc.lower()
            # Считаем, сколько ключевых слов встретилось в документе
            score = sum(1 for kw in keywords if kw in doc_lower)
            if score > 0:
                found_docs.append((doc, score))

        # Сортируем по релевантности (чем больше совпадений, тем выше)
        found_docs.sort(key=lambda x: x[1], reverse=True)

        # Возвращаем только текст документов (без оценок)
        return [doc for doc, _ in found_docs[:3]]  # берем топ-3

    def ask(self, question):
        """
        Главный метод: отвечает на вопрос, используя базу знаний.

        question: вопрос пользователя
        возвращает: ответ модели
        """
        print(f"\nВопрос: {question}")

        # ШАГ 1: Ищем релевантные документы
        relevant_docs = self.search(question)

        if not relevant_docs:
            print("⚠️ Документы не найдены")
            return "Извините, у меня нет информации по этому вопросу."

        # ШАГ 2: Формируем контекст из найденных документов
        context = "\n\n---\n\n".join(relevant_docs)
        print(f"Найдено {len(relevant_docs)} документов")

        # ШАГ 3: Создаем промпт с контекстом
        system_prompt = """
        Ты — полезный ассистент. Отвечай на вопрос, используя ТОЛЬКО информацию из контекста ниже.
        Если ответа нет в контексте — скажи "У меня нет информации об этом".
        Не придумывай ничего от себя.
        """

        user_prompt = f"""
        КОНТЕКСТ (информация для ответа):
        {context}

        ВОПРОС: {question}

        ОТВЕТ:
        """

        # ШАГ 4: Отправляем запрос в LLM
        headers = {
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "GigaChat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,  # низкая температура = меньше фантазий
            "max_tokens": 512
        }

        try:
            response = requests.post(GIGACHAT_API_URL, headers=headers, json=payload)

            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                print(f"✅ Получен ответ")
                return answer
            else:
                return f"Ошибка API: {response.status_code}"

        except Exception as e:
            return f"Ошибка при запросе: {e}"
