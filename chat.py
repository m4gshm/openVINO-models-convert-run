import shutil  # Для определения размера консоли

import openvino_genai as ov_genai


def main():
    model_name = "OpenVINO/Qwen3-8B-int4-cw-ov"
    model_path = "./models/" + model_name
    device = "AUTO"

    print(f"Загрузка модели {model_name} на {device}...")
    # Для CPU убираем cache_dir, чтобы не было RuntimeError
    pipe = ov_genai.LLMPipeline(model_path, device)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 1024
    config.do_sample = True
    config.temperature = 0.5

    history = ov_genai.ChatHistory()

    print("\nЧат запущен! (exit/clear/history)")

    while True:
        prompt = input("\nВы: ").strip()
        if not prompt: continue
        if prompt.lower() == 'exit': break

        if prompt.lower() == 'clear':
            history = ov_genai.ChatHistory()
            print("История очищена.")
            continue

        print("AI: ", end="", flush=True)

        # --- Настройки для контроля переноса ---
        cols, _ = shutil.get_terminal_size()  # Получаем ширину окна
        current_line_length = 4  # Учитываем начальный отступ "AI: "

        def streamer(subword):
            nonlocal current_line_length

            # Если в токене есть явный перенос строки \n
            if "\n" in subword:
                current_line_length = 0
            else:
                # Если текущее слово не влезает в строку, переносим заранее
                if current_line_length + len(subword) > cols - 1:
                    print("\n", end="", flush=True)
                    current_line_length = 0

            print(subword, end="", flush=True)
            current_line_length += len(subword)
            return ov_genai.StreamingStatus.RUNNING

        history.append({'role': 'user', 'content': prompt})
        pipe.generate(history, generation_config=config, streamer=streamer)
        print()

    del pipe
    print("Программа завершена.")


if __name__ == "__main__":
    main()
