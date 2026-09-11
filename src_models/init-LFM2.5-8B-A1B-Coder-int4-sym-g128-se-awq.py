import gc
import json
import os
import shutil
from pathlib import Path

from openvino import save_model
from openvino_tokenizers import convert_tokenizer
from optimum.intel.openvino import OVQuantizer, OVModelForCausalLM
from optimum.intel.openvino.configuration import OVConfig

# 1. Настройка путей
MODEL_ID = "./josephmayo/LFM2.5-8B-A1B-Coder"
SAVE_DIR = "../models/LFM2.5-8B-A1B-Coder-int4-sym-g128-se-awq"
TMP_FP16_DIR = "./tmp_ov_fp16"  # Папка для промежуточного базового экспорта

# Формируем конфигурацию квантования (NNCF)
quantization_config = {
    "bits": 4,
    "sym": True,  # --sym
    "group_size": 128,  # --group-size 128
    "ratio": 1.0,  # Квантовать всю модель
    "dataset": "wikitext2",  # --dataset wikitext2
    "tokenizer": MODEL_ID,  # Жестко прописываем путь для калибратора
    "awq": True,  # --awq
    "scale_estimation": True,  # --scale-estimation
    "backup_precision": "int8_sym",  # --backup-precision int8_sym
}

# Инициализируем OVConfig для квантования
ov_config = OVConfig(quantization_config=quantization_config)

print("--> Шаг 1: Чистый экспорт базовой модели с полным отключением авто-квантования...")
# Использование флага load_in_8bit=False и явного отключения сжатия
model = OVModelForCausalLM.from_pretrained(
    MODEL_ID,
    export=True,
    trust_remote_code=True,
    load_in_8bit=False,  # Явно запрещаем сжатие до 8-бит по умолчанию
    quantization_config={}  # Передаем пустой словарь вместо None, чтобы сбросить дефолты NNCF
)

print("--> Шаг 2: Временное сохранение чистой модели на диск...")
model.save_pretrained(TMP_FP16_DIR)

# Полностью очищаем память, освобождая дескрипторы для Windows
del model
gc.collect()

print("--> Шаг 3: Загрузка несжатой OpenVINO модели для AWQ квантования...")
# Загружаем уже готовый FP16 IR граф
model_clean = OVModelForCausalLM.from_pretrained(TMP_FP16_DIR, trust_remote_code=True)

print("--> Шаг 4: Инициализация квантователя...")
quantizer = OVQuantizer.from_pretrained(model_clean)

print("--> Шаг 5: Запуск квантования (Калибровка wikitext2 + AWQ)...")
try:
    quantizer.quantize(
        ov_config=ov_config,
        save_directory=SAVE_DIR,
        # file_name удален, так как он не поддерживается nncf.compress_weights
    )
    print(f"\n--> Успешно! Квантованная модель сохранена в: {SAVE_DIR}")

except PermissionError as e:
    # Перехват известного бага Windows с зависшими Temp файлами
    if "openvino_model.bin" in str(e):
        print("\n[!] Перехвачена ошибка Windows PermissionError (файл занят в Temp).")
        print("--> Проверяем физический результат...")

        del quantizer
        del model_clean
        gc.collect()

        final_bin = Path(SAVE_DIR) / "openvino_model.bin"
        if final_bin.exists() and final_bin.stat().st_size > 0:
            print(f"--> Все файлы модели успешно записаны в {SAVE_DIR}")
        else:
            print("[!] Ошибка: Итоговый файл весов не найден.")
    else:
        raise e

finally:
    # Очищаем нашу временную папку FP16
    if os.path.exists(TMP_FP16_DIR):
        try:
            shutil.rmtree(TMP_FP16_DIR)
        except Exception:
            print(f"[!] Не удалось удалить временную папку {TMP_FP16_DIR}, удалите её вручную после завершения работы.")

print("--> Восстановление chat_template для OpenVINO GenAI...")
orig_config_path = Path(MODEL_ID) / "tokenizer_config.json"
dest_config_path = Path(SAVE_DIR) / "tokenizer_config.json"

if orig_config_path.exists() and dest_config_path.exists():
    # Читаем оригинальный конфиг токенизатора
    with open(orig_config_path, "r", encoding="utf-8") as f:
        orig_data = json.load(f)

    # Читаем созданный Optimum конфиг токенизатора
    with open(dest_config_path, "r", encoding="utf-8") as f:
        dest_data = json.load(f)

    # Извлекаем шаблон чата. Проверяем стандартное поле и альтернативные варианты
    chat_template = orig_data.get("chat_template")

    if chat_template:
        dest_data["chat_template"] = chat_template
        # Записываем обратно исправленный файл в готовую модель
        with open(dest_config_path, "w", encoding="utf-8") as f:
            json.dump(dest_data, f, ensure_ascii=False, indent=2)
        print("--> Шаблон чата успешно скопирован в итоговую модель!")
    else:
        print("[!] Ошибка: В исходном tokenizer_config.json не найден chat_template.")
else:
    print("[!] Не удалось найти файлы конфигурации токенизатора для восстановления.")

# Путь к вашей готовой квантованной модели
SAVE_DIR = "../models/LFM2.5-8B-A1B-Coder-int4-sym-g128-se-awq"

print("--> Компиляция токенизатора в формат OpenVINO IR...")
try:
    # Загружаем JSON-токенизатор из папки и конвертируем его в OpenVINO структуру
    ov_tokenizer_model = convert_tokenizer(SAVE_DIR, with_detokenizer=True)

    # Сохраняем openvino_tokenizer.xml и openvino_tokenizer.bin прямо в папку модели
    save_model(ov_tokenizer_model, Path(SAVE_DIR) / "openvino_tokenizer.xml")
    print(f"--> Успешно! Файлы openvino_tokenizer.xml/.bin добавлены в {SAVE_DIR}")

except Exception as e:
    print(f"[!] Ошибка при конвертации токенизатора: {e}")
    print("Проверьте, установлена ли библиотека: pip install openvino-tokenizers")

# Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 128/128 • 0:07:32 • 0:00:00
# INFO:nncf:Statistics of the bitwidth distribution:
# +---------------------------+-----------------------------+----------------------------------------+
# | Weight compression mode   | % all parameters (layers)   | % ratio-defining parameters (layers)   |
# +===========================+=============================+========================================+
# | float                     | 0% (1 / 174)                | 0% (0 / 154)                           |
# +---------------------------+-----------------------------+----------------------------------------+
# | int8_sym, per-channel     | 3% (19 / 174)               | 0% (0 / 154)                           |
# +---------------------------+-----------------------------+----------------------------------------+
# | int4_sym, group size 128  | 97% (154 / 174)             | 100% (154 / 154)                       |
# +---------------------------+-----------------------------+----------------------------------------+
# Applying data-aware AWQ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 24/24 • 0:06:01 • 0:00:00
# Applying Scale Estimation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 173/173 • 0:32:38 • 0:00:00
# Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% • 0:00:20 • 0:00:00
