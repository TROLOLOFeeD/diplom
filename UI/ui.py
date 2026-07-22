import sys
import cv2
import numpy as np
import win32gui
import win32con
import win32ui
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint
from PyQt6.QtGui import QFont
import pickle
from pathlib import Path


class Dota2Overlay(QWidget):
    def __init__(self, template_path: str, model_path: str = None):
        super().__init__()
        
        self.template_path = Path(template_path)
        self.template = cv2.imread(str(self.template_path), cv2.IMREAD_GRAYSCALE)
        
        if self.template is None:
            raise ValueError(f"Не удалось загрузить шаблон: {template_path}")
        
        self.h_template, self.w_template = self.template.shape
        
        # Загрузка модели (опционально)
        self.inference_engine = None
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        # Настройка окна оверлея
        self.setup_overlay()
        
        # Label для отображения вероятности
        self.probability_label = QLabel(self)
        self.probability_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 24px;
                font-weight: bold;
                background-color: transparent;
                padding: 5px;
            }
        """)
        self.probability_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.probability_label.hide()
        
        # Таймер для обновления
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(100)  # Обновление каждые 100ms
        
        # Поиск окна Dota 2
        self.dota_window = None
        self.find_dota_window()
    
    def setup_overlay(self):
        """Настройка прозрачного click-through окна"""
        # Убираем рамки и делаем окно поверх всех
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        
        # Делаем окно прозрачным для мыши
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen)
        
        # Прозрачный фон
        self.setStyleSheet("background-color: transparent;")
        
        # Полноэкранное окно
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.show()
    
    def find_dota_window(self):
        """Поиск окна Dota 2"""
        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if "Dota 2" in window_text:
                    self.dota_window = hwnd
                    return True
            return False
        
        try:
            win32gui.EnumWindows(callback, None)
            if self.dota_window:
                print(f"Найдено окно Dota 2: {self.dota_window}")
            else:
                print("Окно Dota 2 не найдено")
        except Exception as e:
            print(f"Ошибка поиска окна: {e}")
    
    def get_dota_window_rect(self):
        """Получение координат окна Dota 2"""
        if not self.dota_window:
            return None
        
        try:
            rect = win32gui.GetWindowRect(self.dota_window)
            return QRect(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
        except:
            return None
    
    def capture_dota_screen(self):
        """Захват экрана Dota 2"""
        if not self.dota_window:
            return None
        
        try:
            hwnd = self.dota_window
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            
            # Получаем размеры окна
            rect = win32gui.GetWindowRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            
            # Создаем bitmap
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bitmap)
            save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)
            
            # Конвертируем в numpy array
            bmpinfo = bitmap.GetInfo()
            bmpstr = bitmap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype='uint8')
            img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
            
            # Освобождаем ресурсы
            win32gui.DeleteObject(bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            
            # Конвертируем в BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            return img_gray
        except Exception as e:
            print(f"Ошибка захвата экрана: {e}")
            return None
    
    def find_template_roi(self, screen_gray):
        """Поиск шаблона на экране и возврат ROI"""
        if screen_gray is None:
            return None
        
        # Template matching
        result = cv2.matchTemplate(screen_gray, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Если нашли шаблон с достаточной уверенностью
        if max_val > 0.8:
            # Возвращаем координаты найденного шаблона
            x, y = max_loc
            return QRect(x, y, self.w_template, self.h_template)
        
        return None
    
    def load_model(self, model_path: str):
        """Загрузка обученной модели"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Импортируем inference engine из model.py
            sys.path.append(str(Path(__file__).parent / 'model'))
            from model import DotaInferenceEngine
            
            self.inference_engine = DotaInferenceEngine(model_data)
            print("Модель успешно загружена")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.inference_engine = None
    
    def predict_delta_probability(self, roi_rect: QRect):
        """Получение дельта-вероятности из модели"""
        if self.inference_engine is None:
            # Демонстрационный режим (заглушка)
            return "+6.7%"
        
        try:
            # Здесь должна быть логика получения текущего состояния игры
            # и вызова counterfactual_predict из модели
            # Для примера возвращаем фиксированное значение
            delta = self.inference_engine.counterfactual_predict(
                hero_id=0,  # Пример: первый герой
                item_id=None,  # Анализ текущей ситуации
                position=0
            )
            
            sign = "+" if delta >= 0 else ""
            return f"{sign}{delta:.1f}%"
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return "+0.0%"
    
    def update_overlay(self):
        """Обновление позиции и контента оверлея"""
        # Проверяем существование окна Dota 2
        if not self.dota_window or not win32gui.IsWindow(self.dota_window):
            self.find_dota_window()
            return
        
        # Получаем позицию окна Dota 2
        dota_rect = self.get_dota_window_rect()
        if not dota_rect:
            return
        
        # Захватываем экран Dota 2
        screen_gray = self.capture_dota_screen()
        if screen_gray is None:
            return
        
        # Ищем шаблон
        roi = self.find_template_roi(screen_gray)
        if roi:
            # Позиционируем оверлей относительно найденного ROI
            # Например, справа от шаблона
            overlay_x = dota_rect.x() + roi.x() + self.w_template + 10
            overlay_y = dota_rect.y() + roi.y()
            
            # Получаем предсказание
            delta_prob = self.predict_delta_probability(roi)
            
            # Обновляем label
            self.probability_label.setText(delta_prob)
            self.probability_label.move(overlay_x, overlay_y)
            self.probability_label.show()
            
            # Обновляем позицию всего оверлея (если нужно)
            self.raise_()
        else:
            self.probability_label.hide()


def main():
    app = QApplication(sys.argv)
    
    # Путь к шаблону (белый квадрат в левом верхнем углу)
    template_path = "изображение_2026-05-28_015236290.png"
    
    # Путь к модели (опционально)
    model_path = "model/saved_model.pkl"  # или путь к вашей модели
    
    overlay = Dota2Overlay(template_path, model_path)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
