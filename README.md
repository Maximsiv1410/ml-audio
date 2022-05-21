# ML-Audio
Данный проект представляет собой ВКР на тему __Анализ существующих архитектур нейронных сетей для распознавания звуков и сбор достаточного датасета.__
В качестве датасета используются аудиодорожки, полученные с видеорегистраторов автомобилей в период записи того или иного значимого события.
Датасет представляет собой множество аудизаписей .wav формата, размещенных в директориях, именованных по классу события.

## Технологический стек
Проект основан Python 3.6
- LibROSA - Пакет для обработки звука
- NumPy - Фундаментальная библиотека для различных вычислений.
- Pandas - Мощный инструмент анализа данных
- Tensorflow - Фреймворк машинного обучения

## Зависимости
Для установки всех требуемых пакетов:
```sh
pip install -r requirements.txt
```

## Начало
Необходимо внимательно ознакомиться со структурой файла конфигурации.
В нем представлены следующие секции:
- [Transform]
    | Параметр | Назначение |
    | ------ | ------ |
    | raw_dataset_path | Путь до директории датасета |
    | classes | Интересующие классы звуков |
    | target_sr | Целевая частота дискретизации |
    | target_duration | Целевая продолжительность аудио |
    | mels | Кол-во мел-фильтров |
    | result_path | Путь до директории для сохранения |
- [Fit]
    | Параметр | Назначение |
    | ------ | ------ |
    | cooked_dataset | Путь до директории обработанного датасета |
    | arch | Используемая архитектура нейросети |
    | epochs | Количество эпох для обучения |
    | batch_size | Размер пакета для прогонки |
- [Predict]
    | Параметр | Назначение |
    | ------ | ------ |
    | В | Процессе |

## Запуск
Доступно три действия:
1. transform
2. fit
3. predict

Запуск происходит следующим образом:
```sh
/bin/python cli.py --action <action> --config <config_path>
```
# Roadmap
В ближайшее время добавится:
1. Генерация отчетов
