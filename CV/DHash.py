import os
import numpy as np
from PIL import Image
from collections import defaultdict
import imagehash

# =========================
# CONFIG
# =========================
HASH_SIZE = 8  # Размер хеша (8x8 = 64 бита)
DATASET_PATHS = [
    ("/heroes_dataset", "hero"),
    ("/items_dataset", "item"),
]
ITEM_CROP_PIXELS = 5  # Сколько пикселей обрезать у items сверху и снизу

# =========================
# DHASH КЛАСС
# =========================
class DHashEngine:
    """
    Разностное хеширование (dHash)
    Быстрое сравнение изображений по perceptual hash
    """
    
    @staticmethod
    def compute_hash(image, hash_size=HASH_SIZE, crop_pixels=0):
        """
        Вычисляет dHash для изображения.
        crop_pixels — сколько пикселей обрезать сверху и снизу (для items).
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Обрезка изображения сверху и снизу
        if crop_pixels and crop_pixels > 0:
            w, h = image.size
            if h > 2 * crop_pixels:
                image = image.crop((0, crop_pixels, w, h - crop_pixels))
            else:
                print(f"⚠️ Изображение слишком маленькое ({w}x{h}) для обрезки на {crop_pixels}px")
        
        return imagehash.dhash(image, hash_size)
    
    @staticmethod
    def hash_to_hex(img_hash):
        """Конвертирует хеш в hex строку для хранения"""
        return str(img_hash)
    
    @staticmethod
    def hex_to_hash(hex_str):
        """Конвертирует hex строку обратно в хеш"""
        return imagehash.hex_to_hash(hex_str)
    
    @staticmethod
    def hamming_distance(hash1, hash2):
        """Вычисляет расстояние Хэмминга между двумя хешами"""
        return hash1 - hash2
    
    @staticmethod
    def similarity(hash1, hash2, max_bits=HASH_SIZE*HASH_SIZE):
        """Вычисляет схожесть в процентах (0-100%)"""
        distance = hash1 - hash2
        return (1 - distance / max_bits) * 100


# =========================
# DATASET
# =========================
class MetricDataset:
    def __init__(self, root_dirs, min_images_per_class=1):
        if isinstance(root_dirs, str):
            root_dirs = [(root_dirs, "hero" if "hero" in root_dirs.lower() else "item")]
        elif isinstance(root_dirs, list) and len(root_dirs) > 0 and isinstance(root_dirs[0], str):
            root_dirs = [(path, "hero" if "hero" in path.lower() else "item") for path in root_dirs]
        
        self.root_dirs = [r[0] if isinstance(r, tuple) else r for r in root_dirs]
        self.root_types = [r[1] if isinstance(r, tuple) else ("hero" if "hero" in r.lower() else "item") for r in root_dirs]
        
        self.filepaths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.class_counts = {}
        self.class_types = {}
        self.class_source_dir = {}
        self.hashes = {}
        
        global_class_idx = 0

        for root_dir, class_type in zip(self.root_dirs, self.root_types):
            if not os.path.exists(root_dir):
                print(f"⚠️ Пропущена директория: {root_dir}")
                continue

            class_images = defaultdict(list)
            for fname in os.listdir(root_dir):
                if not fname.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                parts = fname.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                cls_name, _ = parts
                class_images[cls_name].append(os.path.join(root_dir, fname))

            valid_classes = [cls for cls, paths in class_images.items() if len(paths) >= min_images_per_class]
            
            for cls in valid_classes:
                if cls not in self.class_to_idx:
                    self.class_to_idx[cls] = global_class_idx
                    self.idx_to_class[global_class_idx] = cls
                    self.class_types[global_class_idx] = class_type
                    self.class_source_dir[global_class_idx] = root_dir
                    global_class_idx += 1
                
                label = self.class_to_idx[cls]
                self.class_counts[label] = len(class_images[cls])
                
                for p in class_images[cls]:
                    self.filepaths.append(p)
                    self.labels.append(label)

        if not self.filepaths:
            raise ValueError("Не загружено ни одного изображения.")

        hero_count = sum(1 for t in self.class_types.values() if t == "hero")
        item_count = sum(1 for t in self.class_types.values() if t == "item")
        
        print(f"✅ Загружено {len(self.class_to_idx)} уникальных классов, {len(self.filepaths)} изображений")
        print(f"   Героев: {hero_count}, Предметов: {item_count}")

    def compute_all_hashes(self, hash_engine, item_crop_pixels=0):  # добавлен параметр
        print("🔐 Вычисляем dHash для всех изображений...")
        if item_crop_pixels > 0:
            print(f"   ✂️ Items будут обрезаны на {item_crop_pixels}px сверху и снизу")
        
        for idx, filepath in enumerate(self.filepaths):
            if (idx + 1) % 100 == 0:
                print(f"  Обработано {idx + 1}/{len(self.filepaths)}")
            
            # Определяем тип класса и применяем обрезку только для items
            label = self.labels[idx]
            class_type = self.class_types.get(label, "hero")
            crop = item_crop_pixels if class_type == "item" else 0
            
            img_hash = hash_engine.compute_hash(filepath, crop_pixels=crop)
            self.hashes[filepath] = img_hash
        print(f"✅ Готово! Вычислено {len(self.hashes)} хешей")

    def get_class_info(self, idx):
        return {
            "idx": idx,
            "name": self.idx_to_class.get(idx, "unknown"),
            "type": self.class_types.get(idx, "unknown"),
            "source_dir": self.class_source_dir.get(idx, "unknown"),
            "count": self.class_counts.get(idx, 0)
        }

    def __len__(self):
        return len(self.filepaths)


# =========================
# DHASH-BASED SEARCH ENGINE
# =========================
class DHashSearchEngine:
    """
    Поисковый движок на основе dHash с разделенными функциями поиска
    """
    def __init__(self, dataset, hash_engine, item_crop_pixels=0):
        self.dataset = dataset
        self.hash_engine = hash_engine
        self.item_crop_pixels = item_crop_pixels
        
        if not dataset.hashes:
            raise ValueError("Сначала вызовите dataset.compute_all_hashes()")
        
        self.hero_indices = [
            i for i, label in enumerate(self.dataset.labels) 
            if self.dataset.class_types.get(label) == "hero"
        ]
        self.item_indices = [
            i for i, label in enumerate(self.dataset.labels) 
            if self.dataset.class_types.get(label) == "item"
        ]
        print(f"🚀 Индексы предвычислены: {len(self.hero_indices)} hero, {len(self.item_indices)} item")

    def _search_core(self, query_image, indices, top_k=10, threshold_bits=None, query_crop_pixels=0):
        """Внутренний метод для выполнения поиска по заданным индексам"""
        # Применяем обрезку к запросу, если нужно
        query_hash = self.hash_engine.compute_hash(query_image, crop_pixels=query_crop_pixels)
        similarities = []
        
        for idx in indices:
            filepath = self.dataset.filepaths[idx]
            db_hash = self.dataset.hashes[filepath]
            distance = self.hash_engine.hamming_distance(query_hash, db_hash)
            
            if threshold_bits is not None and distance > threshold_bits:
                continue
            
            similarity = self.hash_engine.similarity(query_hash, db_hash)
            similarities.append((idx, distance, similarity))
        
        similarities.sort(key=lambda x: x[1])
        
        results = []
        for idx, distance, similarity in similarities[:top_k]:
            label = self.dataset.labels[idx]
            class_info = self.dataset.get_class_info(label)
            results.append({
                "filepath": self.dataset.filepaths[idx],
                "hamming_distance": distance,
                "similarity_percent": similarity,
                "class_name": class_info["name"],
                "class_type": class_info["type"],
                "label": label
            })
        
        return results

    def find_similar_heroes(self, query_image, top_k=10, threshold_bits=None):
        """Поиск похожих изображений ТОЛЬКО среди классов типа 'hero'"""
        # Для героев обрезка не нужна
        return self._search_core(query_image, self.hero_indices, top_k, threshold_bits, query_crop_pixels=0)

    def find_similar_items(self, query_image, top_k=10, threshold_bits=None):
        """Поиск похожих изображений ТОЛЬКО среди классов типа 'item'"""
        # Для предметов используем ту же обрезку, что и в датасете
        return self._search_core(
            query_image, self.item_indices, top_k, threshold_bits, 
            query_crop_pixels=self.item_crop_pixels
        )

    def find_similar_all(self, query_image, top_k=10, threshold_bits=None, query_crop_pixels=0):
        """Поиск по всему датасету"""
        all_indices = list(range(len(self.dataset.filepaths)))
        return self._search_core(query_image, all_indices, top_k, threshold_bits, query_crop_pixels)
    
    def find_duplicates(self, threshold_bits=5):
        """Поиск дубликатов в датасете (хеши уже посчитаны с учётом обрезки)"""
        print(f"🔍 Поиск дубликатов (порог: {threshold_bits} бит)...")
        duplicates = []
        processed = set()
        
        for i, fp1 in enumerate(self.dataset.filepaths):
            if i in processed:
                continue
            
            hash1 = self.dataset.hashes[fp1]
            group = [fp1]
            
            for j, fp2 in enumerate(self.dataset.filepaths[i+1:], start=i+1):
                if j in processed:
                    continue
                
                hash2 = self.dataset.hashes[fp2]
                distance = self.hash_engine.hamming_distance(hash1, hash2)
                
                if distance <= threshold_bits:
                    group.append(fp2)
                    processed.add(j)
            
            if len(group) > 1:
                duplicates.append(group)
                processed.add(i)
        
        print(f"✅ Найдено {len(duplicates)} групп дубликатов")
        return duplicates


# =========================
# UTILITIES
# =========================
def print_search_results(results):
    """Красивый вывод результатов поиска"""
    if not results:
        print("Ничего не найдено.")
        return
        
    print("\n" + "="*100)
    print(f"{'Ранг':<5} {'Схожесть':<12} {'Dist':<6} {'Тип':<8} {'Класс':<20} {'Путь'}")
    print("="*100)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['similarity_percent']:<12.2f}% {r['hamming_distance']:<6} "
              f"{r['class_type']:<8} {r['class_name']:<20} {r['filepath']}")
    print("="*100 + "\n")


def save_hashes_to_file(dataset, filepath="hashes.txt", item_crop_pixels=0): 
    with open(filepath, 'w') as f:
        # Пишем метаданные первой строкой, чтобы знать, с какой обрезкой считались хеши
        f.write(f"# crop_pixels={item_crop_pixels}\n")
        for img_path, img_hash in dataset.hashes.items():
            label = dataset.filepaths.index(img_path)  # Оптимизация: лучше сделать маппинг заранее
            label = dataset.labels[dataset.filepaths.index(img_path)]
            class_type = dataset.class_types.get(label, "unknown")
            f.write(f"{img_path}|{str(img_hash)}|{label}|{class_type}\n")
    print(f"💾 Хеши сохранены в {filepath}")


def load_hashes_from_file(dataset, filepath="hashes.txt", expected_crop_pixels=0):
    if not os.path.exists(filepath):
        return False
    
    print(f"📂 Загрузка хешей из {filepath}...")
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        # Проверяем, совпадает ли обрезка с ожидаемой
        if first_line.startswith("# crop_pixels="):
            file_crop = int(first_line.split("=")[1])
            if file_crop != expected_crop_pixels:
                print(f"⚠️ Кэш был посчитан с crop={file_crop}, а нужно {expected_crop_pixels}. Пересчитываю...")
                return False
        
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                img_path, hash_str = parts[0], parts[1]
                if img_path in dataset.filepaths:
                    dataset.hashes[img_path] = imagehash.hex_to_hash(hash_str)
    
    print(f"✅ Загружено {len(dataset.hashes)} хешей")
    return True


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # 1. Создаем датасет
    dataset = MetricDataset(
        root_dirs=DATASET_PATHS,
        min_images_per_class=1
    )
    
    # 2. Создаем движок хеширования
    hash_engine = DHashEngine()
    
    # Имя файла кэша теперь зависит от параметра обрезки,
    # чтобы хеши с разной обрезкой не смешивались
    cache_filename = f"icon_hashes_crop{ITEM_CROP_PIXELS}.txt"
    
    # 3. Вычисляем хеши (или загружаем из файла)
    if not load_hashes_from_file(dataset, cache_filename, expected_crop_pixels=ITEM_CROP_PIXELS):
        dataset.compute_all_hashes(hash_engine, item_crop_pixels=ITEM_CROP_PIXELS)
        save_hashes_to_file(dataset, cache_filename, item_crop_pixels=ITEM_CROP_PIXELS)
    
    # 4. Создаем поисковый движок
    search_engine = DHashSearchEngine(dataset, hash_engine, item_crop_pixels=ITEM_CROP_PIXELS)
    
    # 5. Пример поиска
    print("\n" + "="*80)
    print("ПРИМЕР РАЗДЕЛЬНОГО ПОИСКА")
    print("="*80)
    
    test_image_path = dataset.filepaths[0]
    print(f"\n🔍 Запрос: {test_image_path}")
    
    print("\n🛡️ 1. Поиск похожих ГЕРОЕВ:")
    hero_results = search_engine.find_similar_heroes(
        query_image=test_image_path,
        top_k=5
    )
    print_search_results(hero_results)
    
    print("\n⚔️ 2. Поиск похожих ПРЕДМЕТОВ:")
    item_results = search_engine.find_similar_items(
        query_image=test_image_path,
        top_k=5
    )
    print_search_results(item_results)
    
    print("\n" + "="*80)
    print("ПОИСК ДУБЛИКАТОВ (ОБЩИЙ)")
    print("="*80)
    duplicates = search_engine.find_duplicates(threshold_bits=3)
    for i, group in enumerate(duplicates[:3], 1):
        print(f"\nГруппа {i} ({len(group)} файлов):")
        for fp in group:
            print(f"  - {fp}")
    
    print("\n✅ Готово!")
