import time
from typing import List, Dict, Any

class PlayerState:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.inventory: List[str] = []
        self.last_update_time: float = time.time()
        self.hero_name: str = "unknown"
        self.level: int = 1

    def update_data(self, new_inventory: List[str], hero_name: str = None, level: int = None):
        """Обновляет данные и сбрасывает таймер достоверности до 1.0 (100%)"""
        self.inventory = new_inventory
        if hero_name:
            self.hero_name = hero_name
        if level:
            self.level = level
        self.last_update_time = time.time()

    def get_reliability(self) -> float:
        """
        Рассчитывает текущую достоверность данных.
        Период полураспада = 180 секунд.
        """
        elapsed_time = time.time() - self.last_update_time
        # Формула: 1.0 * (0.5 ** (прошедшее_время / 180))
        reliability = 1.0 * (0.5 ** (elapsed_time / 180.0))
        return max(0.0, min(1.0, reliability)) # Ограничиваем диапазон [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "hero_name": self.hero_name,
            "level": self.level,
            "inventory": self.inventory,
            "reliability": round(self.get_reliability(), 3),
            "last_update": self.last_update_time
        }

class GameStateManager:
    def __init__(self):
        # Словарь состояний для 10 игроков (ID 0-9)
        self.players: Dict[int, PlayerState] = {i: PlayerState(i) for i in range(10)}

    def update_player(self, player_id: int, inventory: List[str], hero_name: str = None, level: int = None):
        if player_id in self.players:
            self.players[player_id].update_data(inventory, hero_name, level)

    def get_all_states(self) -> List[Dict[str, Any]]:
        return [player.to_dict() for player in self.players.values()]
