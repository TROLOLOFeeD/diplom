#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Построение датасета, по локально имеющимся логам каждого матча:
  Симулирует построение инвентаря по каждой записи покупки из лога с учётом составной сборки некоторых предметов
  Фильтрует в построении и записях покупки расходуемые предметы, отдельно обрабатываются поглощаемые предметы Aghanim's Blessing и Aghanim's Shard
  Запись датасета формируется по симулированному состоянию матча в момент любой покупки любым игроком
  Учёт прогресса в processed_matches.txt, сохранение прогресса
"""

import json
import csv
import os
from bisect import bisect_right
from pathlib import Path

# =========================
# CONFIG
# =========================

OUTPUT_CSV = "new_dataset.csv"         # Выходной датасет
PROCESSED_FILE = "processed_matches.txt"  # Трекер обработанных

MATCHES_DIR = r"D:\match_jsons_full"               # Папка с локальными JSON-логами матчей
ITEMS_JSON = "items.json"              # Путь к базе предметов

MAX_MATCHES_PER_RUN = None  # Лимит матчей за запуск (None = без лимита)

# =========================
# XP → LEVEL TABLE
# =========================

XP_THRESHOLDS = [
    0, 230, 600, 1080, 1680, 2300, 2940, 3600, 4280, 5080,
    5900, 6740, 7600, 8480, 9380, 10300, 11240, 12200, 13180,
    14180, 15200, 16240, 17300, 18380, 19480, 20600, 21740,
    22900, 24080, 25280
]

def xp_to_level(xp: int) -> int:
    idx = bisect_right(XP_THRESHOLDS, xp) - 1
    return max(1, idx + 1)

# =========================
# ITEMS DB HELPERS
# =========================

_ITEMS_DB = None
_ITEM_CACHE = {}

def load_items_db(path: str) -> dict:
    """Загружает items.json в глобальный кэш"""
    global _ITEMS_DB
    if _ITEMS_DB is None:
        print(f"Loading items database from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            _ITEMS_DB = json.load(f)
        print(f"Loaded {len(_ITEMS_DB)} items")
    return _ITEMS_DB

def _get_item_data(item_key: str) -> dict:
    """Кэшированный доступ к данным предмета"""
    if item_key not in _ITEM_CACHE:
        _ITEM_CACHE[item_key] = _ITEMS_DB.get(item_key, {}) if _ITEMS_DB else {}
    return _ITEM_CACHE[item_key]

def _is_consumable(item_key: str) -> bool:
    """
    Определяет, является ли предмет расходником.
    Проверяет: behavior-флаги, поле consumable, и фоллбэк-список.
    """
    data = _get_item_data(item_key)
    behavior = data.get("behavior", "")
    
    if "DOTA_ABILITY_BEHAVIOR_CONSUMABLE" in behavior:
        return True
    if data.get("consumable", False):
        return True
    
    consumables = {
        "tango", "tango_single", "clarity", "enchanted_mango",
        "flask", "smoke_of_deceit", "ward_observer", "ward_sentry",
        "dust", "faerie_fire", "blight_stone", "iron_branch", "branches",
        "observer_ward", "sentry_ward", "tpscroll", "bottle", "blood_grenade", "infused_raindrop"
    }
    return item_key in consumables

def _is_aghanim_item(item_key: str) -> bool:
    """
    Проверяет, является ли предмет Aghanim's Scepter или Shard.
    Эти предметы не должны попадать в слоты инвентаря.
    """
    aghanim_items = {
        "aghanims_shard",      # Aghanim's Shard
        "ultimate_scepter"      # Aghanim's Scepter
        "ultimate_scepter_2"     # Aghanim's Scepter Blessing
    }
    return item_key in aghanim_items

def _get_components(item_key: str) -> list:
    """Возвращает список компонентов для крафта предмета"""
    return _get_item_data(item_key).get("components") or []

def reconstruct_inventory(purchase_log, t: int):
    """
    Реконструирует инвентарь игрока на момент времени t.
    Игнорирует расходники и Aghanim's Scepter/Shard.
    """
    inventory = []
    
    for e in purchase_log:
        if e["time"] > t:
            continue
        key = e.get("key", "") or e.get("item_id", "")
        if not key:
            continue
            
        if "recipe" in key.lower():
            continue
        if _is_consumable(key):
            continue
        # Пропускаем Aghanim's Scepter и Shard - они учитываются через флаги
        if _is_aghanim_item(key):
            continue
            
        components = _get_components(key)
        for comp in components:
            if comp in inventory:
                inventory.remove(comp)
                
        inventory.append(key)
    
    slots = inventory[-6:] if len(inventory) > 6 else inventory
    slots += [0] * (6 - len(slots))
    return slots

def has_item_in_log(purchase_log, t: int, item_key: str) -> int:
    for e in purchase_log:
        if e["time"] > t:
            continue
        key = e.get("key", "") or e.get("item_id", "")
        if key == item_key:
            return 1
    return 0

# =========================
# FILE I/O HELPERS
# =========================

def load_processed_matches():
    if not os.path.exists(PROCESSED_FILE):
        return set()
    with open(PROCESSED_FILE, encoding="utf-8") as f:
        return set(line.strip() for line in f)

def mark_match_processed(match_id):
    with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
        f.write(f"{match_id}\n")

def load_local_match(match_id: str, matches_dir: str):
    """
    Загружает данные матча из локального JSON-файла.
    """
    candidates = [
        Path(matches_dir) / f"{match_id}.json",
        Path(matches_dir) / f"match_{match_id}.json",
        Path(matches_dir) / f"{match_id}.txt",
        Path(matches_dir) / f"match_{match_id}.txt",   # ← добавлено
    ]
    
    for match_path in candidates:
        if match_path.exists():
            try:
                with open(match_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                print(f"JSON decode error for {match_path}: {e}")
                return None
            except Exception as e:
                print(f"Error loading {match_path}: {e}")
                return None
    
    print(f"Match file not found for ID {match_id} in {matches_dir}")
    return None

def is_parsed(match_json):
    return match_json is not None and match_json.get("version") is not None

def get_xp_at_time(xp_t, times, t):
    if not xp_t or not times:
        return 0
    idx = bisect_right(times, t) - 1
    if idx < 0:
        return 0
    return xp_t[idx]

# =========================
# MAIN PIPELINE
# =========================

def main():
    # 1. Загружаем базу предметов
    load_items_db(ITEMS_JSON)
    
    # 2. Загружаем трекер обработанных матчей
    processed = load_processed_matches()
    print(f"Already processed: {len(processed)} matches")
    
    # 3. Сканируем MATCHES_DIR на наличие файлов (вместо чтения INPUT_CSV)
    match_dir = Path(MATCHES_DIR)
    candidate_ids = {f.stem for f in match_dir.glob("*.json")} | {f.stem for f in match_dir.glob("*.txt")}
    # Нормализуем имена (убираем префикс match_ если есть)
    clean_ids = {stem[6:] if stem.startswith("match_") else stem for stem in candidate_ids}
    # Оставляем только ещё не обработанные и сортируем для детерминированности
    match_ids = sorted(mid for mid in clean_ids if mid.isdigit() and mid not in processed)
    print(f"Found {len(match_ids)} unprocessed matches in {MATCHES_DIR}")
    
    write_header = not os.path.exists(OUTPUT_CSV)
    processed_this_run = 0
    
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        
        if write_header:
            header = [
                "match_id", "game_time", "radiant_win",
                "target_player_slot", "event_type", "event_item_id"
            ]
            for slot in [0,1,2,3,4,128,129,130,131,132]:
                header += [
                    f"p{slot}_hero_id",
                    f"p{slot}_level",
                    f"p{slot}_item_0", f"p{slot}_item_1", f"p{slot}_item_2",
                    f"p{slot}_item_3", f"p{slot}_item_4", f"p{slot}_item_5",
                    f"p{slot}_aghanims_scepter",
                    f"p{slot}_aghanims_shard"
                ]
            writer.writerow(header)
        
        for match_id in match_ids:
            if MAX_MATCHES_PER_RUN is not None and processed_this_run >= MAX_MATCHES_PER_RUN:
                print(f"Reached MAX_MATCHES_PER_RUN = {MAX_MATCHES_PER_RUN}, stopping.")
                break
                
            print(f"Processing match {match_id}")
            match = load_local_match(match_id, MATCHES_DIR)
            if match is None:
                continue
                
            if not is_parsed(match):
                print(f"Match {match_id} is not parsed, skipping")
                continue
                
            radiant_win = 1 if match["radiant_win"] else 0
            players = {p["player_slot"]: p for p in match["players"]}
            
            purchase_events = []
            for p in match["players"]:
                for e in p.get("purchase_log", []):
                    purchase_events.append({
                        "player_slot": p["player_slot"],
                        "time": e["time"],
                        "item_key": e.get("key", "") or e.get("item_id", "")
                    })
            
            for event in purchase_events:
                t = event["time"]
                target_slot = event["player_slot"]
                event_item = event["item_key"]
                
                # ПРОПУСКАЕМ события покупки расходников
                if _is_consumable(event_item):
                    continue
                
                row_base = [
                    match_id, t, radiant_win,
                    target_slot, "none", event_item
                ]
                
                state_features = []
                
                for slot in [0,1,2,3,4,128,129,130,131,132]:
                    if slot not in players:
                        state_features += [0] * 10
                        continue
                        
                    p = players[slot]
                    hero_id = p["hero_id"]
                    
                    xp = get_xp_at_time(
                        p.get("xp_t", []),
                        p.get("times", []),
                        t
                    )
                    level = xp_to_level(xp)
                    
                    items = reconstruct_inventory(
                        p.get("purchase_log", []), t - 1
                    )
                    
                    # Проверяем наличие Aghanim's Scepter и Shard
                    has_scepter = (
                        has_item_in_log(p.get("purchase_log", []), t, "ultimate_scepter") or
                        has_item_in_log(p.get("purchase_log", []), t, "ultimate_scepter_2")
                    )
                    has_shard = has_item_in_log(
                        p.get("purchase_log", []), t, "aghanims_shard"
                    )
                    
                    state_features += [
                        hero_id, level,
                        *items,
                        has_scepter,
                        has_shard
                    ]
                
                writer.writerow(row_base + state_features)
            
            mark_match_processed(match_id)
            processed_this_run += 1
            print(f"Finished match {match_id} ({processed_this_run} this run)")
    
    print("✅ DONE")

if __name__ == "__main__":
    main()
