# FIM benchmark

Сравнение моделей код-комплита (fill-in-the-middle) на OpenVINO-агенте через
`POST /v1/completions`. Бенч гоняет фиксированный набор кейсов и оценивает,
содержит ли заполнение середины ключевой фрагмент, плюс замеряет время.

## Запуск

```bash
# сервер уже запущен вручную (порт 8887)
.venv\Scripts\python -m fim_bench --model Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq

# бенч сам стартует/останавливает сервер на каждую модель (через _bench_model.bat)
.venv\Scripts\python -m fim_bench --model Qwen3.5-2B-int4-sym --launch

# несколько моделей подряд + отчёт в файл
.venv\Scripts\python -m fim_bench --model A --model B --launch --report reports\fim.md
```

## Структура

| Файл | Назначение |
|---|---|
| `models.py` | реестр моделей и их **FIM-токенов** (Qwen / LFM / Mellum-стили); добавление моделей сюда |
| `cases.py` | кейсы `(prefix, suffix, key)`; добавление задач сюда |
| `client.py` | HTTP-клиент `/v1/completions` + ожидание готовности сервера |
| `runner.py` | прогон кейсов, результаты, форматирование отчёта (expected/actual) |
| `__main__.py` | CLI: `--model` (повторяемый), `--port`, `--launch`, `--server-arg`, `--report` |
| `_bench_model.bat` | запуск сервера модели с безопасными NPU-флагами (`<model> [extra args...]`) |
| `fim-bench-all.bat` | бенч всех моделей из дефолтного набора, отчёт в `reports\` |
| `prefill-compare.bat` | сравнение prefill-режимов (STATIC/DYN-DEFAULT/DYN-HFA/DYN-PYRAMID) для одной модели |
| `qwen25-full-compare.bat` | матрица: все Qwen2.5-Coder × все prefill-режимы |

## Важно: FIM-токены

У семейств разные управляющие токены, при неверном стиле модель выдаёт пустоту
или мусор:

| Семейство | Формат |
|---|---|
| Qwen2.5-Coder, Qwen3.5 | `<\|fim_prefix\|> … <\|fim_suffix\|> … <\|fim_middle\|>` |
| LFM2.x-Coder (LiquidAI) | `<\|fim_pre\|> … <\|fim_suf\|> … <\|fim_mid\|>` |
| Mellum (JetBrains) | `<fim_prefix> … <fim_suffix> … <fim_middle>` (без пайпов) |

## Серверные флаги

`_bench_model.bat` запускает агент с консервативной конфигурацией NPU:
`--npu_prefill_hint STATIC` (динамические attention-хинты на LFM2.5 падают —
см. `bug-report-openvino-38246-npuw-strided-view.md`), без prefix caching.

Тесты: `.venv\Scripts\python -m pytest fim_bench -q`

## Сводный отчёт

Каждый прогон `python -m fim_bench` дописывает результаты кейсов в общий лог
`reports\fim-results.jsonl` (метка прогона: `--label`, у prefill-compare это имя
режима). Сводная таблица по всем прогонам:

```bash
.venv\Scripts\python -m fim_bench.summary
.venv\Scripts\python -m fim_bench.summary --out reports\summary.md
```

Формат таблицы: строка = (модель, режим), колонки = кейсы (`PASS 12.3s` /
`fail 4.5s` / `ERR`), плюс итог passed и суммарное время.

Скрипты `fim-bench-all.bat` и `prefill-compare.bat` сами обновляют сводную
таблицу `reports\fim-summary.md` после прогона.
