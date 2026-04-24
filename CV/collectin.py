import cv2
import numpy as np
import mss
import win32gui
import win32process
import psutil
import threading
import time
import queue
from pathlib import Path
from pynput import mouse
from typing import List, Tuple, Callable, Optional

class Dota2StateCapture:
    def __init__(self, 
                 hero_regions: List[Tuple[int, int, int, int]],
                 item_regions: List[Tuple[int, int, int, int]],
                 inventory_template: str = "inventory_grid.png",  # 🔥 Путь к вашей сетке
                 window_name: str = "Dota 2",
                 save_dir: str = "dota2_dataset",
                 on_items_captured: Optional[Callable[[int, List[np.ndarray]], None]] = None):
        
        self.hero_regions = hero_regions
        self.item_regions = item_regions
        self.window_name = window_name
        self.on_items_captured = on_items_captured
        
        self.inventory_template_path = inventory_template
        self.inventory_offset = (0, 0)  # (dx, dy) смещение относительно хардкода
        self._inv_edges = None
        self._prepare_inventory_template()
        
        # Настройки сохранения
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._save_queue = queue.Queue()
        self._stop_event = threading.Event()
        
        self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._save_thread.start()
        
        self.window_rect = None
        self._find_dota_window()
        
        self._mouse_listener = None
        self._last_click_time = 0
        self._debounce = 0.3
        self._running = False
        self._lock = threading.Lock()

    def _prepare_inventory_template(self):
        """Конвертирует шаблон в контуры для устойчивого поиска"""
        if not Path(self.inventory_template_path).exists():
            print(f"[!] Шаблон инвентаря не найден: {self.inventory_template_path}")
            return
        img = cv2.imread(self.inventory_template_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("[!] Ошибка чтения шаблона инвентаря")
            return
        
        self._inv_edges = cv2.Canny(img, 50, 150)
        print(f"[OK] Шаблон инвентаря загружен. Размер: {self._inv_edges.shape}")

    def _calibrate_inventory(self) -> bool:
        """Ищет сетку инвентаря по контурам. Вызывается при каждом клике."""
        if self._inv_edges is None or not self.window_rect:
            return False

        anchor_x, anchor_y = self.item_regions[0][0], self.item_regions[0][1]
        margin = 80  # Запас поиска ±80px
        h_t, w_t = self._inv_edges.shape

        # Область поиска относительно окна
        roi_x = max(0, anchor_x - margin)
        roi_y = max(0, anchor_y - margin)
        roi_w = w_t + 2 * margin
        roi_h = h_t + 2 * margin

        abs_left = self.window_rect[0] + roi_x
        abs_top  = self.window_rect[1] + roi_y

        with mss.mss() as sct:
            roi = np.array(sct.grab({"top": abs_top, "left": abs_left, "width": roi_w, "height": roi_h}))
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)

        # Контурный поиск
        roi_edges = cv2.Canny(roi_gray, 50, 150)
        res = cv2.matchTemplate(roi_edges, self._inv_edges, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.1:  # Порог уверенности для контуров
            found_x = roi_x + max_loc[0]
            found_y = roi_y + max_loc[1]
            self.inventory_offset = (found_x - anchor_x, found_y - anchor_y)
            print(f"📐 Калибровка: смещение {self.inventory_offset} (conf={max_val:.2f})")
            return True
        
        print(f"[!] Инвентарь не найден (conf={max_val:.2f})")
        return False

    def _find_dota_window(self) -> bool:
        def callback(hwnd, lParam):
            try:
                if not win32gui.IsWindowVisible(hwnd): return True
                title = win32gui.GetWindowText(hwnd)
                if self.window_name.lower() in title.lower():
                    try:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        if psutil.Process(pid).name().lower() == "dota2.exe":
                            self.window_rect = win32gui.GetWindowRect(hwnd)
                            return False
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, Exception):
                        pass
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(callback, None)
        except Exception as e:
            print(f"[!] EnumWindows прерван: {e}")
            hwnd = win32gui.FindWindow(None, self.window_name)
            if hwnd and win32gui.IsWindowVisible(hwnd):
                self.window_rect = win32gui.GetWindowRect(hwnd)

        if self.window_rect:
            print(f"[OK] Dota 2 найдено. Координаты: {self.window_rect}")
            return True
        print("[!] Окно не найдено. Запустите от имени Администратора.")
        return False

    def _capture_safe(self, regions: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        if not self.window_rect: return []
        
        # 🔥 Применяем динамическое смещение
        dx = self.inventory_offset + 5
        dy = self.inventory_offset + 7
        min_x = min(r[0] + dx for r in regions)
        min_y = min(r[1] + dy for r in regions)
        max_x = max(r[0] + r[2] + dx for r in regions)
        max_y = max(r[1] + r[3] + dy for r in regions)
        
        abs_left = min_x + self.window_rect[0]
        abs_top  = min_y + self.window_rect[1]
        w, h = max_x - min_x, max_y - min_y
        
        with mss.mss() as sct:
            frame = cv2.cvtColor(np.array(sct.grab({"top": abs_top, "left": abs_left, "width": w, "height": h})), cv2.COLOR_BGRA2BGR)
            
        crops = []
        for rx, ry, rw, rh in regions:
            # Корректируем координаты с учётом смещения
            adj_x, adj_y = rx + dx, ry + dy
            x1, y1 = max(0, adj_x - min_x), max(0, adj_y - min_y)
            x2, y2 = min(w, adj_x - min_x + rw), min(h, adj_y - min_y + rh)
            crops.append(frame[y1:y2, x1:x2])
        return crops

    def _queue_save(self, dtype: str, crops: List[np.ndarray], hero_idx: int = -1):
        ts = int(time.time() * 1000)
        self._save_queue.put((dtype, crops, hero_idx, ts))

    def _save_worker(self):
        while not self._stop_event.is_set():
            try:
                dtype, crops, hero_idx, ts = self._save_queue.get(timeout=1.0)
                self._save_images(dtype, crops, hero_idx, ts)
                self._save_queue.task_done()
            except queue.Empty:
                continue

    def _save_images(self, dtype: str, crops: List[np.ndarray], hero_idx: int, ts: int):
        if dtype == "heroes":
            folder = self.save_dir / "heroes" / f"{ts}"
        else:
            folder = self.save_dir / "items" / f"hero_{hero_idx}" / f"{ts}"
        folder.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(crops):
            path = folder / f"{i:02d}.png"
            cv2.imwrite(str(path), img)
        print(f"💾 Сохранено {len(crops)} кадров в: {folder}")

    def capture_heroes_once(self) -> List[np.ndarray]:
        heroes = self._capture_safe(self.hero_regions)
        if heroes:
            self._queue_save("heroes", heroes)
        return heroes

    def _on_click(self, x: int, y: int, button, pressed: bool):
        if not pressed or not self._running or not self.window_rect:
            return True
            
        with self._lock:
            now = time.time()
            if now - self._last_click_time < self._debounce:
                return True
            self._last_click_time = now

        rel_x, rel_y = x - self.window_rect[0], y - self.window_rect[1]
        
        for hero_idx, (hx, hy, hw, hh) in enumerate(self.hero_regions):
            if hx <= rel_x <= hx + hw and hy <= rel_y <= hy + hh:
                print(f"🖱 Клик по герою #{hero_idx}. Рекалибровка...")
                
                # 🔥 1. Рекалибруем позицию инвентаря
                if not self._calibrate_inventory():
                    print("[⚠] Пропуск захвата: инвентарь не найден")
                    return True
                
                # 🔥 2. Снимаем предметы с учётом найденного смещения
                print("📦 Захват предметов...")
                items = self._capture_safe(self.item_regions)
                self._queue_save("items", items, hero_idx=hero_idx)
                if self.on_items_captured:
                    self.on_items_captured(hero_idx, items)
                break
        return True

    def start(self):
        if not self.window_rect:
            print("[!] Окно не найдено.")
            return
        self._running = True
        self._mouse_listener = mouse.Listener(on_click=self._on_click)
        self._mouse_listener.start()
        print("✅ Слушатель запущен. Кликайте по героям.")

    def stop(self):
        self._running = False
        if self._mouse_listener:
            self._mouse_listener.stop()
        self._stop_event.set()
        print("🛑 Остановка.")
        
if __name__ == "__main__":
    HERO_REGIONS = [(545 + i*62, 5, 62, 35) for i in range(5)] + [(1064 + i*62, 5, 62, 35) for i in range(5)]
    ITEM_REGIONS = [(1140 + i*65, 942, 65, 45) for i in range(3)] + [(1140 + i*65, 992, 65, 45) for i in range(3)] + [(1140 + i*65, 1042, 65, 30) for i in range(3)]

    def on_items_detected(hero_idx, item_crops):
        pass

    # 🔥 Укажите путь к сохранённому изображению сетки
    capturer = Dota2StateCapture(
        hero_regions=HERO_REGIONS,
        item_regions=ITEM_REGIONS,
        inventory_template="inventory_grid1.png",  # Ваша картинка
        save_dir="dataset",
        on_items_captured=on_items_detected
    )

    print("📸 Снимаю состав команд...")
    capturer.capture_heroes_once()
    capturer.start()

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        capturer.stop()
