# История Казахстана — QA-бот

Дообучение языковой модели для ответов на вопросы по истории Казахстана
(школьный учебник, 11 класс) на русском языке.

## Описание

Проект реализует полный конвейер: распознавание текста учебника, генерацию
обучающего набора вопрос–ответ, дообучение модели методом QLoRA и оценку
качества.

**Конвейер:**

1. **OCR** — распознавание страниц 3–23 файла `book.pdf` («История Казахстана,
   11 класс») с помощью vision-модели **Qwen2.5-VL-7B** (4-bit). Модель читает
   страницу с учётом контекста, поэтому римские цифры (века), даты, казахские и
   английские слова распознаются корректно → `history_text.txt`.
2. **Генерация датасета** — по распознанному тексту формируется набор пар
   «вопрос–ответ» (283 пары, генерация через `gpt-5-mini`) →
   `history_sft_dataset.json`.
3. **Дообучение (SFT / QLoRA)** — модель `Meta-Llama-3.1-8B-Instruct` (4-bit)
   дообучается с помощью Unsloth и TRL `SFTTrainer`.
4. **Оценка** — метрики ROUGE / BLEU / BERTScore на отложенной выборке и
   сравнение базовой и дообученной моделей.

## Файлы

| Файл | Описание |
| --- | --- |
| `history_finetuning.py` | Основной скрипт в percent-формате (`# %%`). |
| `history_finetuning.ipynb` | Ноутбук, полученный из скрипта, со всеми выводами. |
| `history_text.txt` | Текст учебника после OCR. |
| `history_sft_dataset.json` | Датасет пар «вопрос–ответ». |
| `requirements.txt` | Зависимости. |

`book.pdf` (исходный скан учебника) в репозиторий не входит и предоставляется
отдельно.

## Запуск (Google Colab, GPU T4)

1. Загрузите `book.pdf` и `history_finetuning.py` в сессию Colab.
2. Задайте секреты через Colab Secrets или локальный `.env`:
   `OPENAI_API_KEY`, `HF_TOKEN` (опционально `HF_USERNAME` для выгрузки на Hub).
3. Преобразуйте скрипт в ноутбук:
   ```bash
   pip install jupytext
   jupytext --to notebook history_finetuning.py
   ```
4. Откройте `history_finetuning.ipynb` → **Runtime → Run all**.
5. Сохраните ноутбук вместе с выводами ячеек.

## Публикация

После обучения артефакты публикуются на Hugging Face Hub (публично):
- Модель (LoRA-адаптер): `timkaiyr/kz-history-qa-llama3.1-8b`
- Датасет + OCR-текст (`history_text.txt`): `timkaiyr/kz-history-qa-dataset`

Кривая обучения строится в ноутбуке через matplotlib (`loss_curve.png`) и
показывает две линии — training loss и eval loss — на одних осях.

## Секреты

`OPENAI_API_KEY` и `HF_TOKEN` хранятся в локальном `.env` (исключён из git) или
в Colab Secrets. Не добавляйте их в систему контроля версий.

## Стек

Python · Unsloth · TRL · Transformers · PyTorch · Hugging Face Datasets ·
Qwen2.5-VL (OCR) · OpenAI API · `evaluate` (ROUGE / BLEU / BERTScore) · matplotlib.
