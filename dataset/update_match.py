import os
import re
import json
import requests
import time
from datetime import datetime

"""
Обновление данных каждого матча из уже имеющегося списка матчей
Данный парсер делает запрос API OpenDota, который возвращает информацию о конкретном матче
Если запрашиваемый матч сохранён в базе OpenDota кратко (до 20Мб записи),
  то отправляется запрос на полное обновление данных этого матча сервисом
  Через некоторе время можно будет собирать полную информацию матча
"""

# --- НАСТРОЙКИ ---
FOLDER_PATH = r"D:\match_jsons_full"
MIN_SIZE_THRESHOLD = 20000
TARGET_VERSION = 22
API_URL_REQUEST = "https://api.opendota.com/api/request  "
FILENAME_PATTERN = re.compile(r"match_(\d+)\.txt")

# ЛИМИТЫ API (Безопасные значения для бесплатного тарифа)
# OpenDota часто дает ~1 запрос на парсинг в минуту.
# Ставим 65 секунд, чтобы точно не получить бан.
DELAY_BETWEEN_REQUESTS = 65
MAX_RETRIES_ON_429 = 3  # Сколько раз пробовать повторить при ошибке 429 перед пропуском

def get_match_id_from_filename(filename):
    match = FILENAME_PATTERN.match(filename)
    return match.group(1) if match else None

def check_file_status(filepath):
    """Быстрая проверка: возвращает True, если нужен парсинг"""
    filename = os.path.basename(filepath)
    match_id = get_match_id_from_filename(filename)
    if not match_id: return False, "Неверное имя"

    # 1. Проверка размера
    if os.path.getsize(filepath) < MIN_SIZE_THRESHOLD:
        return True, f"Малый размер ({os.path.getsize(filepath)} б)"

    # 2. Проверка версии
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        version = data.get('version')
        if version is None or int(version) < TARGET_VERSION:
            return True, f"Версия {version}"
        return False, "OK"
    except Exception:
        return True, "Ошибка чтения"

def request_parse_with_retry(match_id):
    """Отправляет запрос с обработкой ошибки 429, формируя URL вручную"""
    
    # Формируем URL
    # Добавляем ID в конец пути
    url = f"https://api.opendota.com/api/request/{match_id}"
    
    payload = {} 

    for attempt in range(MAX_RETRIES_ON_429):
        try:
            # Отправляем запрос на сформированный URL
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                resp_data = response.json()
                # Если матч уже в очереди, API может вернуть статус или сообщение об этом
                # Обычно успешный ответ содержит queue_position или similar
                queue_pos = resp_data.get('queue_position', 'в обработке')
                return True, f"Очередь: {queue_pos}"
            
            elif response.status_code == 429:
                # Лимит превышен. Ждем дольше и пробуем снова.
                wait_time = 60 * (attempt + 1)
                print(f"\n   ⚠️ ЛИМИТ (429). Ждем {wait_time} сек перед повтором...", end="", flush=True)
                time.sleep(wait_time)
                continue
            
            elif response.status_code == 400:
                # Матч уже в очереди или ID неверен
                return False, "Уже в очереди/Не найден (400)"
            
            elif response.status_code == 404:
                # Несуществующий матч, а не ошибку в URL
                return False, "Матч не найден (404)"
            
            else:
                return False, f"Ошибка {response.status_code}"
                
        except Exception as e:
            return False, f"Сеть: {e}"
            
    return False, "Превышено число попыток (429)"

def main():
    if not os.path.exists(FOLDER_PATH):
        print(f"❌ Папка не найдена: {FOLDER_PATH}")
        return

    print("🔍 ЭТАП 1: Сканирование файлов...")
    all_files = sorted([f for f in os.listdir(FOLDER_PATH) if f.endswith('.txt')])
    
    if not all_files:
        print("📂 Файлы не найдены.")
        return

    # Собираем список задач
    tasks = []
    print(f"Обрабатываю {len(all_files)} файлов...", end=" ", flush=True)
    
    for filename in all_files:
        filepath = os.path.join(FOLDER_PATH, filename)
        needs_parse, reason = check_file_status(filepath)
        if needs_parse:
            match_id = get_match_id_from_filename(filename)
            if match_id:
                tasks.append({'id': match_id, 'file': filename, 'reason': reason})
    
    print(f"Готово.\n🎯 Найдено {len(tasks)} матчей для парсинга из {len(all_files)}.")

    if not tasks:
        print("✅ Все матчи уже имеют полную версию!")
        return

    # Сохраняем список на случай сбоя, чтобы не терять прогресс
    pending_file = "pending_matches.txt"
    with open(pending_file, 'w', encoding='utf-8') as f:
        for t in tasks:
            f.write(f"{t['id']} ({t['reason']})\n")
    # ... (предыдущий код без изменений)

    print(f"💾 Список ожидающих сохранен в {pending_file}")

    SKIP_COUNT = 0
    if len(tasks) > SKIP_COUNT:
        print(f"⏭️  Пропускаем первые {SKIP_COUNT} файлов из списка...")
        tasks = tasks[SKIP_COUNT:]
        print(f"   Осталось обработать: {len(tasks)} файлов.")
    else:
        print(f"⚠️  В списке всего {len(tasks)} файлов, пропуск не требуется или список пуст.")
        tasks = []
    # --- КОНЕЦ ИЗМЕНЕНИЙ ---

    if not tasks:
        print("✅ Нечего отправлять после пропуска.")
        return

    print(f"\n🚀 ЭТАП 2: Отправка запросов (Режим: 1 запрос в {DELAY_BETWEEN_REQUESTS} сек)...")
    print("⚠️  Это займет время. Не закрывайте окно.\n")

    success_count = 0
    fail_count = 0
    
    start_time = time.time()

    for i, task in enumerate(tasks, 1):
        # ... (остальной код без изменений)
        match_id = task['id']
        filename = task['file']
        
        print(f"[{i}/{len(tasks)}] Матч {match_id} ({filename[:20]}...) - {task['reason']}")
        
        success, msg = request_parse_with_retry(match_id)
        
        if success:
            print(f"   ✅ Успех: {msg}")
            success_count += 1
        else:
            print(f"   ❌ Провал: {msg}")
            fail_count += 1
        
        # Если это не последний элемент, делаем паузу
        if i < len(tasks):
            next_action_time = time.time() + DELAY_BETWEEN_REQUESTS
            print(f"   ⏳ Следующий запрос через {DELAY_BETWEEN_REQUESTS} сек (в {datetime.fromtimestamp(next_action_time).strftime('%H:%M:%S')})...")
            time.sleep(DELAY_BETWEEN_REQUESTS)

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "="*50)
    print("🏁 РАБОТА ЗАВЕРШЕНА")
    print("="*50)
    print(f"Успешно отправлено: {success_count}")
    print(f"Ошибок: {fail_count}")
    print(f"Затраченное время: {int(hours)}ч {int(minutes)}м {int(seconds)}с")
    print(f"Средняя скорость: ~1 запрос в минуту (лимит API)")
    print("="*50)
    print("💡 Совет: Запустите этот скрипт ночью или когда компьютер не нужен,")
    print("так как обработка большого списка может занять много часов.")

if __name__ == "__main__":
    main()
