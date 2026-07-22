# UI/ui.py
import sys
import json
import zmq
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QProgressBar
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

class ZMQSubscriberThread(QThread):
    # Сигнал для передачи данных в главный поток GUI
    data_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:5555")
        # Подписываемся на все сообщения (пустая строка)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.is_running = True

    def run(self):
        while self.is_running:
            try:
                # Ждем сообщение (блокирующий вызов, но в отдельном потоке, так что GUI не виснет)
                message = self.socket.recv_string()
                data = json.loads(message)
                self.data_received.emit(data)
            except zmq.ZMQError as e:
                if self.is_running:
                    print(f"ZMQ Error: {e}")
                break

    def stop(self):
        self.is_running = False
        self.socket.close()
        self.context.term()

class DotaOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        # Настройка ZeroMQ потока
        self.zmq_thread = ZMQSubscriberThread()
        self.zmq_thread.data_received.connect(self.update_overlay)
        self.zmq_thread.start()

    def init_ui(self):
        # Настройки прозрачного окна поверх всех (как в оригинале)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput # Пропускает клики сквозь окно!
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Геометрия (должна совпадать с окном Dota 2, настраивается через calibrate.py)
        self.setGeometry(100, 100, 400, 600)
        
        # Основной лейаут
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(self.layout)
        
        # Словарь для хранения виджетов каждого игрока
        self.player_widgets = {}
        
        # Инициализация пустых виджетов для 10 игроков
        for i in range(10):
            player_container = QWidget()
            player_layout = QVBoxLayout(player_container)
            
            # Имя героя и уровень
            self.hero_label = QLabel(f"Player {i+1}: Ожидание данных...")
            self.hero_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            self.hero_label.setStyleSheet("color: white;")
            
            # Прогресс-бар достоверности
            self.reliability_bar = QProgressBar()
            self.reliability_bar.setRange(0, 100)
            self.reliability_bar.setValue(0)
            self.reliability_bar.setTextVisible(True)
            self.reliability_bar.setFormat("Достоверность: %p%")
            # Стилизация прогресс-бара
            self.reliability_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid grey;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #333333;
                }
                QProgressBar::chunk {
                    background-color: #00FF00; /* Зеленый по умолчанию */
                }
            """)
            
            # Рекомендации модели
            self.prediction_label = QLabel("Рекомендация: -")
            self.prediction_label.setFont(QFont("Arial", 10))
            self.prediction_label.setStyleSheet("color: #FFFF00;")
            
            player_layout.addWidget(self.hero_label)
            player_layout.addWidget(self.reliability_bar)
            player_layout.addWidget(self.prediction_label)
            
            self.layout.addWidget(player_container)
            self.player_widgets[i] = {
                "hero": self.hero_label,
                "reliability": self.reliability_bar,
                "prediction": self.prediction_label
            }

    def update_overlay(self, data: dict):
        """Вызывается в главном потоке при получении данных из ZMQ"""
        if data.get("type") != "PLAYER_UPDATE":
            return
            
        player_id = data["player_id"]
        if player_id not in self.player_widgets:
            return
            
        widgets = self.player_widgets[player_id]
        reliability = data["reliability"] * 100 # Переводим в проценты (0-100)
        
        # Обновление текста героя
        widgets["hero"].setText(f"Игрок {player_id+1}: {data['hero']} (Ур. {data['level']})")
        
        # Обновление прогресс-бара достоверности и его цвета
        widgets["reliability"].setValue(int(reliability))
        if reliability > 70:
            color = "#00FF00" # Зеленый (высокая достоверность)
        elif reliability > 30:
            color = "#FFA500" # Оранжевый (средняя)
        else:
            color = "#FF0000" # Красный (низкая, данные устарели)
            
        widgets["reliability"].setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid grey;
                border-radius: 3px;
                text-align: center;
                background-color: #333333;
                color: white;
            }}
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)
        
        # Обновление рекомендаций
        top_items = ", ".join(data.get("top_3_items", ["Нет данных"]))
        widgets["prediction"].setText(f"Рекомендация: {top_items}")

    def closeEvent(self, event):
        self.zmq_thread.stop()
        self.zmq_thread.wait()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = DotaOverlay()
    overlay.show()
    sys.exit(app.exec())
