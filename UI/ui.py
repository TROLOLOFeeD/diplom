import sys
import json
import zmq
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

class ZMQSubscriberThread(QThread):
    data_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.is_running = True

    def run(self):
        while self.is_running:
            try:
                message = self.socket.recv_string()
                data = json.loads(message)
                self.data_received.emit(data)
            except zmq.ZMQError:
                if self.is_running:
                    pass # Игнорируем тихие ошибки при закрытии
                break

    def stop(self):
        self.is_running = False
        self.socket.close()
        self.context.term()

class DotaOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        self.zmq_thread = ZMQSubscriberThread()
        self.zmq_thread.data_received.connect(self.update_overlay)
        self.zmq_thread.start()

    def init_ui(self):
        # Настройки прозрачного окна поверх всех (Click-through)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput 
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Геометрия оверлея (подстройте под ваше разрешение)
        self.setGeometry(50, 50, 350, 500)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # 1. Глобальный индикатор достоверности (Верхняя часть)
        self.global_status_label = QLabel("Ожидание данных...")
        self.global_status_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.global_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.global_status_label)

        # Разделительная линия
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #555555;")
        main_layout.addWidget(line)

        # 2. Контейнер для данных игроков (Средняя часть)
        self.players_layout = QVBoxLayout()
        self.players_layout.setSpacing(5)
        
        self.player_labels = {}
        for i in range(10):
            p_label = QLabel(f"Игрок {i+1}: Нет данных")
            p_label.setFont(QFont("Segoe UI", 10))
            p_label.setStyleSheet("color: #CCCCCC;")
            self.players_layout.addWidget(p_label)
            self.player_labels[i] = p_label
            
        main_layout.addLayout(self.players_layout)

        # 3. Глобальная рекомендация инференса (Нижняя часть)
        self.inference_box = QFrame()
        self.inference_box.setStyleSheet("background-color: rgba(0, 0, 0, 150); border-radius: 8px;")
        inf_layout = QVBoxLayout(self.inference_box)
        
        self.inf_title = QLabel("🤖 РЕКОМЕНДАЦИЯ СИСТЕМЫ:")
        self.inf_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.inf_title.setStyleSheet("color: #FFFFFF;")
        
        self.inf_prediction = QLabel("Загрузка модели...")
        self.inf_prediction.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.inf_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inf_prediction.setWordWrap(True)
        
        inf_layout.addWidget(self.inf_title)
        inf_layout.addWidget(self.inf_prediction)
        main_layout.addWidget(self.inference_box)

    def update_overlay(self, data: dict):
        if data.get("type") != "GLOBAL_UPDATE":
            return
            
        global_rel = data.get("global_reliability", 0.0)
        
        # --- ЛОГИКА ЦВЕТОВЫХ СЛОЕВ ---
        if global_rel < 0.5:
            # 🔹 ЖЁЛТЫЙ слой: от 0 до 0.5
            color_hex = "#FFCC00"
            status_text = "⚠ Оценка проводится по устаревшим данным"
            alpha_bg = "rgba(255, 204, 0, 40)" # Легкий желтый фон
        elif global_rel < 0.7:
            # 🔹 СЕРЫЙ слой: от 0.5 до 0.7
            color_hex = "#A0A0A4"
            status_text = "⚡ Данные частично устарели"
            alpha_bg = "rgba(160, 160, 164, 40)" # Легкий серый фон
        else:
            # 🔹 ЗЕЛЁНЫЙ слой: от 0.7 до 1.0
            color_hex = "#2ECC71"
            status_text = "✅ Оценка актуальна"
            alpha_bg = "rgba(46, 204, 113, 40)" # Легкий зеленый фон

        # Применяем цвет к глобальному статусу
        self.global_status_label.setText(f"{status_text}\n(Достоверность: {global_rel:.1%})")
        self.global_status_label.setStyleSheet(f"color: {color_hex}; background-color: {alpha_bg}; border-radius: 5px; padding: 5px;")

        # Применяем цвет шрифта к выводу инференса
        self.inf_prediction.setStyleSheet(f"color: {color_hex};")
        self.inf_prediction.setText(", ".join(data.get("top_3_items", ["Нет рекомендаций"])))

        # Обновляем список игроков (опционально, для полноты картины)
        players_data = data.get("players", [])
        for p_data in players_data:
            pid = p_data["player_id"]
            if pid in self.player_labels:
                hero = p_data["hero_name"]
                level = p_data["level"]
                rel = p_data["reliability"]
                
                # Цвет имени игрока тоже зависит от его личной достоверности
                p_color = color_hex if rel > 0.7 else "#A0A0A4"
                self.player_labels[pid].setText(
                    f"Игрок {pid+1}: {hero} (Ур. {level}) <span style='color:{p_color}'>[ {rel:.0%} ]</span>"
                )
                # Включаем поддержку HTML в QLabel
                self.player_labels[pid].setTextFormat(Qt.TextFormat.RichText)

    def closeEvent(self, event):
        self.zmq_thread.stop()
        self.zmq_thread.wait()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = DotaOverlay()
    overlay.show()
    sys.exit(app.exec())
