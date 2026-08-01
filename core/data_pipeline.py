import zmq
import json
import time
import numpy as np
from CV.state_manager import GameStateManager
from model.onnx_inference import ONNXInferencer
from model.model import DotaDataProcessor

class DataPipeline:
    def __init__(self, heroes_json: str = "dataset/heroes.json", items_json: str = "dataset/items.json"):
        self.state_manager = GameStateManager()
        self.inferencer = ONNXInferencer()
        self.processor = DotaDataProcessor(heroes_json, items_json)
        
        # Настройка ZeroMQ PUB сокета
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        # Привязываем к порту. UI будет подключаться к tcp://localhost:5555
        self.socket.bind("tcp://*:5555")
        print("[Pipeline] ZeroMQ PUB socket bound to tcp://*:5555")
    
    def encode_player_state(self, player_state: dict) -> np.ndarray:
        """
        Преобразует состояние игрока в вектор признаков для ONNX модели.
        Возвращает массив формы (1, input_dim) где input_dim зависит от архитектуры модели.
        """
        inventory = player_state.get("inventory", [])
        hero_name = player_state.get("hero_name", "unknown")
        level = player_state.get("level", 1)
        
        # Создаем фиктивные признаки для демонстрации
        # В реальности здесь должна быть логика маппинга предметов и героев на ID
        item_ids = []
        for item in inventory[:6]:  # Берем максимум 6 предметов
            # Пытаемся найти предмет в словаре, иначе используем 0
            item_idx = self.processor.item_to_idx.get(item, 0)
            item_ids.append(item_idx)
        
        # Дополняем до 6 предметов нулями
        while len(item_ids) < 6:
            item_ids.append(0)
        
        # Получаем ID героя (если имя не найдено, используем 0)
        # В реальности нужен маппинг hero_name -> hero_id
        hero_id = 0  # Заглушка
        
        # Формируем простой вектор признаков [hero_id, level, item_0, ..., item_5]
        # Это упрощенная версия - реальная модель может требовать более сложную структуру
        features = np.array([hero_id, level] + item_ids, dtype=np.float32)
        
        # Добавляем размерность batch: (batch_size=1, features)
        return features.reshape(1, -1)

    def _calculate_global_reliability(self) -> float:
        """
        Считает геометрическое среднее достоверности всех 10 игроков.
        """
        states = self.state_manager.get_all_states()
        reliabilities = [state["reliability"] for state in states]
        
        # Защита от полного обнуления (если один игрок 0.0, геом. среднее будет 0.0).
        # Добавляем минимальный эпсилон, чтобы UI не "ломался", но оставался в желтой зоне.
        reliabilities = [max(r, 1e-4) for r in reliabilities]
        
        # Формула геометрического среднего: (Π x_i) ^ (1/n)
        global_rel = float(np.prod(reliabilities) ** (1.0 / len(reliabilities)))
        return round(global_rel, 4)

    def process_update(self, player_id: int, raw_inventory: list, hero: str, level: int):
        """Вызывается из CV модуля при обнаружении изменений"""
        # 1. Обновляем состояние и метрику достоверности
        self.state_manager.update_player(player_id, raw_inventory, hero, level)
        
        # 2. Подготовка признаков для модели
        player_state = self.state_manager.players[player_id].to_dict()
        features = self.encode_player_state(player_state)
        
        # 3. ONNX Инференс
        predictions = self.inferencer.predict(features)
        
        # 4. Формируем пакет данных для UI
        global_reliability = self._calculate_global_reliability()
        
        update_packet = {
            "type": "GLOBAL_UPDATE", # Изменили тип для ясности
            "global_reliability": global_reliability,
            "timestamp": time.time(),
            "players": self.state_manager.get_all_states(),
            "top_3_items": self._get_top_items(predictions)
        }
        
        # 5. Публикация в шину
        self.socket.send_string(json.dumps(update_packet))

    def _get_top_items(self, predictions: np.ndarray, top_k: int = 3) -> list:
        # Простая логика получения топ-K предметов по вероятности
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        return [f"item_id_{idx}" for idx in top_indices] # Замените на реальные имена предметов

    def run(self):
        """Бесконечный цикл ожидания (если нужно слушать и REQ/REP, иначе просто держит сокет открытым)"""
        try:
            while True:
                zmq.proxy(self.socket, self.socket) # Заглушка, держит поток живым
        except KeyboardInterrupt:
            print("[Pipeline] Shutting down...")
            self.socket.close()
            self.context.term()
