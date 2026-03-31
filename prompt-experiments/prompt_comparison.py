import pandas as pd
import json
from datetime import datetime
from bert_score import score
from typing import List, Dict
from config import AUTH_TOKEN

# Импортируем функцию из предыдущего скрипта
from basic_api_call import call_gigachat

"""
Сравнение разных промптов на тестовом датасете
Демонстрирует:
- Батч-тестирование промптов
- Вычисление метрик (exact match, BERTScore)
- Логирование результатов
"""

# Тестовый датасет
TEST_DATASET = [
    {
        "question": "Какая гарантия на диваны из натуральной кожи?",
        "expected_answer": "5 лет на каркас и механизм трансформации, 2 года на обивку",
        "category": "warranty"
    },
    {
        "question": "Можно ли вернуть мебель, если не подошла по цвету?",
        "expected_answer": "Только в течение 7 дней, если товар не был в использовании и сохранена заводская упаковка",
        "category": "returns"
    },
    {
        "question": "Сколько времени занимает изготовление мебели на заказ?",
        "expected_answer": "от 4 до 6 недель в зависимости от сложности проекта",
        "category": "custom_order"
    },
    {
        "question": "Предоставляете ли вы рассрочку на мебель премиум-класса?",
        "expected_answer": "Да, рассрочка до 12 месяцев без процентов от партнеров-банков",
        "category": "payment"
    },
    {
        "question": "Как ухаживать за столом из массива дуба?",
        "expected_answer": "Использовать мягкую ткань без абразивов, избегать влажной уборки, раз в полгода наносить защитный воск",
        "category": "care"
    },
    {
        "question": "Есть ли выставочный зал для примерки мебели?",
        "expected_answer": "Да, шоурум в центре города работает ежедневно с 10 до 21 часа, запись не требуется",
        "category": "showroom"
    },
    {
        "question": "Какой срок доставки по Нижнему Новгороду?",
        "expected_answer": "1-3 рабочих дня при наличии на складе",
        "category": "delivery"
    },
    {
        "question": "Можно ли заказать образцы тканей перед покупкой?",
        "expected_answer": "Да, бесплатно до 5 образцов с доставкой курьером за 1-2 дня",
        "category": "samples"
    },
    {
        "question": "Сколько стоит подъем мебели на этаж без лифта?",
        "expected_answer": "Бесплатно для заказов от 150 000 рублей, до 150 000 — 500 рублей за этаж",
        "category": "delivery"
    },
    {
        "question": "Какие документы подтверждают качество и безопасность?",
        "expected_answer": "Сертификаты соответствия ТР ТС, гигиенические заключения, паспорта качества на материалы",
        "category": "certification"
    },
    {
        "question": "Можно ли изменить комплектацию после оформления заказа?",
        "expected_answer": "В течение 24 часов без штрафов, позже — только если производство еще не запущено",
        "category": "order_modification"
    },
    {
        "question": "Как вызвать мастера для сборки?",
        "expected_answer": "Сборка включена в стоимость доставки, мастер приезжает в день доставки в течение часа",
        "category": "assembly"
    }
]

# Промпты для тестирования
PROMPTS = [
    {
        "name": "zero_shot",
        "system_prompt": "Ты — полезный ассистент. Отвечай на вопросы."
    },
    {
        "name": "few_shot",
        "system_prompt": """
        Ты — ассистент службы поддержки.

        Примеры:
        Вопрос: Сколько времени занимает изготовление мебели на заказ?
        Ответ: от 4 до 6 недель в зависимости от сложности проекта.

        Вопрос: Сколько стоит подъем мебели на этаж без лифта?
        Ответ: Бесплатно для заказов от 150 000 рублей, до 150 000 — 500 рублей за этаж
        """
    },
    {
        "name": "chain_of_thought",
        "system_prompt": """
        Ты — ассистент. Отвечай пошагово.

        Сначала напиши "Рассуждение:", затем "Ответ:".
        """
    }
]


def evaluate_responses(results: List[Dict], expected_answers: Dict) -> Dict:
    """
    Оценивает качество ответов модели
    """
    model_answers = [r["answer"] for r in results]
    references = [expected_answers[r["question"]] for r in results]

    # Exact match
    exact_matches = sum(
        1 for a, e in zip(model_answers, references)
        if a.strip().lower() == e.strip().lower()
    )
    exact_match = exact_matches / len(references) if references else 0

    # BERTScore
    try:
        P, R, F1 = score(model_answers, references, lang="ru", verbose=False)
        bert_score = F1.mean().item()
    except Exception as e:
        print(f"BERTScore failed: {e}")
        bert_score = None

    return {
        "exact_match": exact_match,
        "bert_score": bert_score,
        "total_samples": len(references)
    }


def run_experiment():
    """
    Запускает эксперимент по сравнению промптов
    """
    results = []

    for prompt in PROMPTS:
        print(f"\n=== Тестируем промпт: {prompt['name']} ===")

        responses = []
        for test in TEST_DATASET:
            answer = call_gigachat(
                prompt=test["question"],
                system_prompt=prompt["system_prompt"],
                temperature=0.3
            )
            responses.append({
                "question": test["question"],
                "answer": answer
            })
            print(f"Q: {test['question']}")
            print(f"A: {answer}\n")

        expected = {t["question"]: t["expected_answer"] for t in TEST_DATASET}
        metrics = evaluate_responses(responses, expected)

        results.append({
            "prompt_name": prompt["name"],
            **metrics,
            "timestamp": datetime.now().isoformat()
        })

    # Сохраняем результаты
    df = pd.DataFrame(results)
    print("\n=== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ===")
    print(df[["prompt_name", "exact_match", "bert_score"]])

    # Сохраняем в CSV
    df.to_csv("results/experiment_results.csv", index=False)

    # Сохраняем в JSON
    with open("results/experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    run_experiment()
