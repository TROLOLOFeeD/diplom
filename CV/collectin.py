import os
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
from PIL import Image
import imagehash

# =========================
# 1. DHASH ENGINE
# =========================
HASH_SIZE = 8  # 8x8 = 64 бита

class DHashEngine:
    @staticmethod
    def compute_hash(image, hash_size=HASH_SIZE, crop_pixels=0):  # 🔥 ДОБАВЛЕН crop_pixels
        """
        Вычисляет dHash. Поддерживает путь к файлу или numpy массив (OpenCV).
        crop_pixels — сколько пикселей обрезать сверху и снизу.
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            if crop_pixels > 0:
                w, h = image.size
                if h > 2 * crop_pixels:
                    image = image.crop((0, crop_pixels, w, h - crop_pixels))
                else:
                    print(f"⚠️ Изображение слишком маленькое ({w}x{h}) для обрезки на {crop_pixels}px")
        elif isinstance(image, np.ndarray):
            # OpenCV использует BGR, PIL ожидает RGB
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if crop_pixels > 0:
                h, w = image.size[1], image.size[0]
                if h > 2 * crop_pixels:
                    image = image.crop((0, crop_pixels, w, h - crop_pixels))
                else:
                    print(f"⚠️ Изображение слишком маленькое ({w}x{h}) для обрезки на {crop_pixels}px")
        
        return imagehash.dhash(image, hash_size)
    
    @staticmethod
    def hamming_distance(hash1, hash2):
        return hash1 - hash2
    
    @staticmethod
    def similarity(hash1, hash2, max_bits=HASH_SIZE*HASH_SIZE):
        distance = hash1 - hash2
        return (1 - distance / max_bits) * 100


# =========================
# 2. DATASET & SEARCH ENGINE
# =========================
class MetricDataset:
    def __init__(self, hash_file="icon_hashes.txt"):
        self.hash_file = hash_file
        self.filepaths = []
        self.labels = []
        self.class_types = {}
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.hashes = {}
        
        self._load_hashes()

    def _load_hashes(self):
        if not os.path.exists(self.hash_file):
            print(f"[!] Файл хешей не найден: {self.hash_file}")
            return
        
        print(f"📂 Загрузка эталонных хешей из {self.hash_file}...")
        with open(self.hash_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    filepath, hash_str, label_str, class_type = parts[0], parts[1], parts[2], parts[3]
                    label = int(label_str)
                    
                    self.filepaths.append(filepath)
                    self.labels.append(label)
                    self.class_types[label] = class_type
                    self.hashes[filepath] = imagehash.hex_to_hash(hash_str)
                    
                    if label not in self.class_to_idx:
                        filename = os.path.basename(filepath)
                        classname = filename.rsplit('_', 1)[0] if '_' in filename else filename.rsplit('.', 1)[0]
                        self.class_to_idx[label] = classname
                        self.idx_to_class[label] = classname
        
        print(f"✅ Загружено {len(self.hashes)} эталонных хешей ({len(self.class_to_idx)} уникальных классов)")


class DHashSearchEngine:
    def __init__(self, dataset: MetricDataset):
        self.dataset = dataset
        self.hash_engine = DHashEngine()
        
        self.hero_indices = [i for i, label in enumerate(self.dataset.labels) if self.dataset.class_types.get(label) == "hero"]
        self.item_indices = [i for i, label in enumerate(self.dataset.labels) if self.dataset.class_types.get(label) == "item"]
        print(f"🚀 Индексы предвычислены: {len(self.hero_indices)} hero, {len(self.item_indices)} item")

    def _search_core(self, query_image, indices, top_k=1, min_similarity=75.0, query_crop_pixels=0):  # 🔥 ДОБАВЛЕН query_crop_pixels
        query_hash = self.hash_engine.compute_hash(query_image, crop_pixels=query_crop_pixels)
        similarities = []
        
        for idx in indices:
            filepath = self.dataset.filepaths[idx]
            db_hash = self.dataset.hashes[filepath]
            
            max_bits_diff = int(64 * (1 - min_similarity / 100))
            distance = self.hash_engine.hamming_distance(query_hash, db_hash)
            
            if distance > max_bits_diff:
                continue
            
            similarity = self.hash_engine.similarity(query_hash, db_hash)
            if similarity >= min_similarity:
                similarities.append((idx, distance, similarity))
        
        similarities.sort(key=lambda x: x[1])
        
        results = []
        for idx, distance, similarity in similarities[:top_k]:
            label = self.dataset.labels[idx]
            results.append({
                "class_name": self.dataset.idx_to_class.get(label, "unknown"),
                "similarity_percent": similarity,
                "hamming_distance": distance
            })
        return results

    def find_similar_heroes(self, query_image, top_k=1, min_similarity=75.0):
        return self._search_core(query_image, self.hero_indices, top_k, min_similarity, query_crop_pixels=0)

    def find_similar_items(self, query_image, top_k=1, min_similarity=75.0, query_crop_pixels=0):  # 🔥 ДОБАВЛЕН query_crop_pixels
        return self._search_core(query_image, self.item_indices, top_k, min_similarity, query_crop_pixels)


# =========================
# 3. DOTA2 CAPTURER
# =========================
class Dota2StateCapture:
    def __init__(self, 
                 hero_regions: List[Tuple[int, int, int, int]],
                 item_regions: List[Tuple[int, int, int, int]],
                 inventory_template: str = "inventory_grid1.png",
                 window_name: str = "Dota 2",
                 save_dir: str = "dataset",
                 search_engine: Optional[DHashSearchEngine] = None,
                 classification_threshold: float = 75.0,
                 capture_delay_ms: float = 15.0,
                 item_crop_pixels: int = 5,  # 🔥 НОВОЕ: сколько обрезать у items
                 on_items_captured: Optional[Callable[[int, List[str]], None]] = None):
        
        self.hero_regions = hero_regions
        self.item_regions = item_regions
        self.window_name = window_name
        self.on_items_captured = on_items_captured
        self.item_crop_pixels = item_crop_pixels  # 🔥 НОВОЕ
        
        self.inventory_template_path = inventory_template
        self.inventory_offset = (0, 0)
        self._inv_edges = None
        self._prepare_inventory_template()
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._save_queue = queue.Queue()
        self._stop_event = threading.Event()
        
        self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._save_thread.start()
        
        self.search_engine = search_engine
        self.classification_threshold = classification_threshold
        self.capture_delay_ms = capture_delay_ms
        
        self.window_rect = None
        self._find_dota_window()
        
        self._mouse_listener = None
        self._last_click_time = 0
        self._debounce = 0.3
        self._running = False
        self._lock = threading.Lock()
        
        # 🔥 НОВОЕ: Определяем, какие индексы items нужно обрезать
        self._item_crop_mask = self._compute_item_crop_mask()

    def _compute_item_crop_mask(self) -> List[int]:
        """
        Определяет, сколько пикселей обрезать для каждого item.
        Ряды с высотой 45 обрезаем на item_crop_pixels, ряд с высотой 30 — нет.
        """
        mask = []
        for rx, ry, rw, rh in self.item_regions:
            if rh == 45:  # Первые два ряда
                mask.append(self.item_crop_pixels)
            elif rh == 30:  # Последний ряд (уже обрезан)
                mask.append(0)
            else:
                mask.append(0)  # По умолчанию не обрезаем
        return mask

    def _prepare_inventory_template(self):
        if not Path(self.inventory_template_path).exists():
            print(f"[!] Шаблон инвентаря не найден: {self.inventory_template_path}")
            return
    
        # Загружаем с альфа-каналом
        img = cv2.imread(self.inventory_template_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("[!] Ошибка чтения шаблона инвентаря")
            return
    
        # Если есть альфа-канал (4 канала), создаем маску
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            # Маска: непрозрачные пиксели = 255, прозрачные = 0
            _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        
            # Обрезаем шаблон до контура непустой области
            coords = cv2.findNonZero(mask)
            x, y, w, h = cv2.boundingRect(coords)
        
            # Обрезаем шаблон и маску
            self._inv_template = img[y:y+h, x:x+w, :3]  # RGB без альфы
            self._inv_mask = mask[y:y+h, x:x+w]
        
            # Для edge detection используем только видимую часть
            gray = cv2.cvtColor(self._inv_template, cv2.COLOR_BGR2GRAY)
            self._inv_edges = cv2.Canny(gray, 50, 150)
        
            print(f"[OK] Шаблон обрезан: {w}x{h}, маска: {cv2.countNonZero(self._inv_mask)} пикселей")
        else:
            # Если нет альфа-канала, создаем маску по порогу яркости
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, self._inv_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            self._inv_edges = cv2.Canny(gray, 50, 150)
            self._inv_template = img

    def _calibrate_inventory(self) -> bool:
        if self._inv_edges is None or not self.window_rect:
            return False

        anchor_x, anchor_y = self.item_regions[0][0], self.item_regions[0][1]
        margin = 80
        h_t, w_t = self._inv_edges.shape

        roi_x = max(0, anchor_x - margin)
        roi_y = max(0, anchor_y - margin)
        roi_w = w_t + 2 * margin
        roi_h = h_t + 2 * margin

        abs_left = self.window_rect[0] + roi_x
        abs_top  = self.window_rect[1] + roi_y

        with mss.mss() as sct:
            roi = np.array(sct.grab({"top": abs_top, "left": abs_left, "width": roi_w, "height": roi_h}))
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)

        roi_edges = cv2.Canny(roi_gray, 50, 150)
    
        # 🔥 ВАЖНО: Используем маску если она есть
        if hasattr(self, '_inv_mask') and self._inv_mask is not None:
            res = cv2.matchTemplate(roi_edges, self._inv_edges, cv2.TM_CCORR_NORMED, mask=self._inv_mask)
        else:
            res = cv2.matchTemplate(roi_edges, self._inv_edges, cv2.TM_CCOEFF_NORMED)
    
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.3:  # 🔥 Повышаем порог (было 0.1)
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
                    except Exception:
                        pass
            except Exception:
                pass
            return True

        try:
            win32gui.EnumWindows(callback, None)
        except Exception:
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
        
        if self.capture_delay_ms > 0:
            time.sleep(self.capture_delay_ms / 1000.0)
        
        dx = self.inventory_offset[0] if isinstance(self.inventory_offset, tuple) else self.inventory_offset + 5
        dy = self.inventory_offset[1] if isinstance(self.inventory_offset, tuple) else self.inventory_offset + 7
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
            adj_x, adj_y = rx + dx, ry + dy
            x1, y1 = max(0, adj_x - min_x), max(0, adj_y - min_y)
            x2, y2 = min(w, adj_x - min_x + rw), min(h, adj_y - min_y + rh)
            crops.append(frame[y1:y2, x1:x2])
            # В конце _capture_safe, перед return crops:
            debug_frame = frame.copy()
            for rx, ry, rw, rh in regions:
                adj_x, adj_y = rx + dx, ry + dy
                x1, y1 = max(0, adj_x - min_x), max(0, adj_y - min_y)
                cv2.rectangle(debug_frame, (x1, y1), (x1 + rw, y1 + rh), (0, 0, 255), 2)

            cv2.imshow("DEBUG_CROPS", debug_frame)
            cv2.waitKey(0)  # Нажмите любую клавишу для продолжения
            cv2.destroyAllWindows()
        return crops

    def _queue_save(self, dtype: str, crops: List[np.ndarray], hero_idx: int = -1, timestamp: int = None):
        ts = timestamp if timestamp else int(time.time() * 1000)
        self._save_queue.put((dtype, crops, hero_idx, ts))

    def _save_worker(self):
        while not self._stop_event.is_set():
            try:
                item = self._save_queue.get(timeout=1.0)
                if item is None:
                    break
                dtype, crops, hero_idx, ts = item
                self._process_and_save(dtype, crops, hero_idx, ts)
                self._save_queue.task_done()
            except queue.Empty:
                continue

    def _classify_batch(self, crops: List[np.ndarray], search_func, crop_pixels_list: List[int] = None) -> List[str]:
        """Классифицирует пакет изображений через search_engine"""
        if not self.search_engine:
            return [f"unknown_{i:02d}" for i in range(len(crops))]
        
        classnames = []
        for i, crop in enumerate(crops):
            # 🔥 Получаем crop_pixels для текущего изображения
            crop_px = crop_pixels_list[i] if crop_pixels_list and i < len(crop_pixels_list) else 0
            
            # 🔥 Передаём crop_pixels в search_func
            matches = search_func(query_image=crop, top_k=1, min_similarity=self.classification_threshold, query_crop_pixels=crop_px)
            
            if matches:
                best = matches[0]
                name = best['class_name']
                conf = best['similarity_percent']
                classnames.append(f"{name}_{i:02d}")
                print(f"  [{i}] Распознано: {name} ({conf:.1f}%, crop={crop_px}px)")
            else:
                classnames.append(f"unknown_{i:02d}")
                print(f"  [{i}] Не распознано (ниже порога {self.classification_threshold}%, crop={crop_px}px)")
        return classnames

    def _process_and_save(self, dtype: str, crops: List[np.ndarray], hero_idx: int, ts: int):
        """Фоновая обработка: классификация + сохранение"""
        if dtype == "heroes":
            folder = self.save_dir / "heroes" / f"{ts}"
            search_func = self.search_engine.find_similar_heroes if self.search_engine else None
            crop_pixels_list = [0] * len(crops)  # Heroes не обрезаются
        else:
            folder = self.save_dir / "items" / f"hero_{hero_idx}" / f"{ts}"
            search_func = self.search_engine.find_similar_items if self.search_engine else None
            # 🔥 Используем предвычисленную маску обрезки
            crop_pixels_list = self._item_crop_mask
        
        folder.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 Классификация {len(crops)} изображений ({dtype})...")
        classnames = self._classify_batch(crops, search_func, crop_pixels_list)
        
        for i, (img, classname) in enumerate(zip(crops, classnames)):
            path = folder / f"{classname}.png"
            cv2.imwrite(str(path), img)
        
        print(f"💾 Сохранено {len(crops)} кадров в: {folder}")
        
        if self.on_items_captured and dtype == "items":
            self.on_items_captured(hero_idx, classnames)

    def capture_heroes_once(self) -> List[np.ndarray]:
        """Сбор героев при инициализации"""
        print("📸 Сбор героев...")
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
                print(f"\n🖱 Клик по герою #{hero_idx}. Рекалибровка...")
                
                if not self._calibrate_inventory():
                    print("[⚠] Пропуск захвата: инвентарь не найден")
                    return True
                
                print("📦 Захват предметов...")
                items = self._capture_safe(self.item_regions)
                self._queue_save("items", items, hero_idx=hero_idx)
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
        self._save_queue.put(None)
        print("🛑 Остановка.")


# =========================
# 4. MAIN
# =========================
if __name__ == "__main__":
    print("="*80)
    print("1. ИНИЦИАЛИЗАЦИЯ DHASH SEARCH ENGINE")
    print("="*80)
    
    # 🔥 ВАЖНО: используем тот же файл кэша, что и в первом скрипте
    dataset = MetricDataset(hash_file="icon_hashes_crop5.txt")
    if dataset.hashes:
        search_engine = DHashSearchEngine(dataset)
    else:
        print("[!] ВНИМАНИЕ: Эталонные хеши не загружены. Классификация будет пропущена")
        search_engine = None

    HERO_REGIONS = [(545 + i*62, 5, 62, 35) for i in range(5)] + [(1064 + i*62, 5, 62, 35) for i in range(5)]
    ITEM_REGIONS = [(1143 + i*65, 945, 65, 45) for i in range(3)] + \
                   [(1143 + i*65, 995, 65, 45) for i in range(3)] + \
                   [(1143 + i*65, 1045, 65, 30) for i in range(3)]

    def on_items_detected(hero_idx, item_classnames):
        print(f"\n🎯 Итог для героя #{hero_idx}:")
        for i, name in enumerate(item_classnames):
            print(f"   Слот {i}: {name}")
        print()

    print("\n" + "="*80)
    print("2. ЗАПУСК CAPTURER")
    print("="*80)
    
    capturer = Dota2StateCapture(
        hero_regions=HERO_REGIONS,
        item_regions=ITEM_REGIONS,
        inventory_template="inventory_grid1.png",
        save_dir="dataset",
        search_engine=search_engine,
        classification_threshold=75.0,
        capture_delay_ms=15.0,
        item_crop_pixels=5,
        on_items_captured=on_items_detected
    )

    capturer.capture_heroes_once()
    
    print("\n✅ Запуск слушателя. Кликайте по героям для захвата предметов.")
    print("   Формат сохранения: classname_00.png")
    print("   Нажмите Ctrl+C для остановки.\n")
    
    capturer.start()

    try:
        while True: 
            time.sleep(1)
    except KeyboardInterrupt:
        capturer.stop()
