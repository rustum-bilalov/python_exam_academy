1. Цели и предпосылки
1.1. Зачем идем в разработку продукта?

Бизнес-цель: прогнозирование количества контактов.
Почему станет лучше: Прогнозирование с помощью машинного обучения позволит оптимизировать распределение ресурсов и повысить точность прогнозирования.
Успех итерации: Увеличение точности прогнозов на n% сравнительно с текущим методом.
1.2. Бизнес-требования и ограничения

Бизнес-требования: прогнозирование KPI по контактам за 2023 год.
Бизнес-ограничения: Использование данных за последние три года для обучения модели.
Успешный пилот: Создание пайплайна прогнозирования с хорошей точностью.
1.3. Что входит в скоуп проекта/итерации, что не входит

Закрытие БТ: Разработка модели прогнозирования с использованием данных за последние три года.
Не входит: Интеграция модели в производственное окружение.
Результат: Прототип модели с демонстрацией ее эффективности.
1.4. Предпосылки решения

Прогнозирование временных рядов требует учета сезонности и трендов.
Использование модели Prophet обосновано ее способностью автоматически учитывать сезонные и праздничные паттерны.
2. Методология
2.1. Постановка задачи

Прогнозирование количества контактов.
2.3. Этапы решения

Этап 1 - Подготовка данных:
Выгрузка данных из Google BigQuery.
Предобработка данных и агрегация по дням.
Разделение данных на обучающую и тестовую выборки.
Этап 2 - Обучение базовой модели (бейзлайн):
Использование модели Prophet для обучения базовой модели.
Оценка качества прогнозов по выбранным метрикам.
Этап 3 - Обучение основной модели (MVP):
Тюнинг параметров модели Prophet для улучшения прогнозов.
Оценка результатов и сравнение с базовой моделью.
Этап 4 - Оценка результатов и анализ:
Сравнение качества прогнозов бейзлайна и MVP.
Этап 5 - Интеграция результатов:
Подготовка отчета о результатах для бизнеса.
Планирование интеграции модели.

3. Архитектура решения
3.1. Архитектура решения

Блок-схема:
Импорт данных из Google BigQuery в Jupyter Notebook.
Обработка данных и обучение моделей прогнозирования временных рядов.
Анализ результатов и подготовка отчетности.
3.2. Описание инфраструктуры и масштабируемости

Инфраструктура: Использование Jupyter Notebook для разработки моделей.
Плюсы выбора: Легкость в использовании, интерактивный анализ данных.
Минусы выбора: Ограниченные возможности масштабирования и интеграции.
3.3. Integration points

Описание взаимодействия между сервисами (методы API и др.): Интеграция с Google BigQuery для получения данных.
