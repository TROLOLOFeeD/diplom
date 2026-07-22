import zmq
import json
import numpy as np
from dataset.state_manager import GameStateManager
from model.onnx_inference import ONNXInferencer
from dataset.builder import DotaDataProcessor # Ваш препроцессинг

class DataPipeline:
    def __init__(self):
        self.state_manager = GameStateManager()
        self.inferencer = ONNXInferencer()
        self.processor = DotaDataProcessor()
        
        # Настройка ZeroMQ PUB сокета
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        # Привязываем к порту. UI будет подключаться к tcp://localhost:5555
        self.socket.bind("tcp://*:5555")
        print("[Pipeline] ZeroMQ PUB socket bound to tcp://*:5555")

    def process_update(self, player_id: int, raw_inventory: list, hero: str, level: int):
        """Вызывается из CV модуля при обнаружении изменений"""
        # 1. Обновляем состояние и метрику достоверности
        self.state_manager.update_player(player_id, raw_inventory, hero, level)
        
        # 2. Подготовка признаков для модели
        player_state = self.state_manager.players[player_id]
        features = self.processor.encode_state(player_state) # Ваш метод векторизации
        
        # 3. ONNX Инференс
        predictions = self.inferencer.predict(features)
        
        # 4. Формируем пакет данных для UI
        update_packet = {
            "type": "PLAYER_UPDATE",
            "player_id": player_id,
            "hero": hero,
            "level": level,
            "inventory": raw_inventory,
            "reliability": player_state.get_reliability(),
            "top_3_items": self._get_top_items(predictions)
        }
        
        # 5. Публикация в шину (сериализация в JSON)
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
