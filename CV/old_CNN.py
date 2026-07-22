import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from collections import defaultdict
from PIL import Image
import random

# =========================
# CONFIG
# =========================
EMBED_DIM = 256
BACKBONE = "resnet18"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 Укажите список датасетов для совместного обучения:
DATASET_PATHS = [
    ("/home/asd/heroes_dataset", "hero"),  # 256x144
    ("/home/asd/items_dataset", "item"),   # 88x64
]

# =========================
# DATASET (поддержка нескольких папок + аугментация одиночных классов)
# =========================
class MetricDataset(Dataset):
    def __init__(self, root_dirs, transform=None, min_images_per_class=1, augment_single=True):
        """
        root_dirs: list[tuple[str, str]] — список кортежей (директория, тип_класса)
                   или list[str] — тогда тип определяется по имени папки
        augment_single: если True, применять более агрессивную аугментацию к классам с 1 изображением
        """
        # Нормализуем вход: если переданы строки, пытаемся определить тип по имени папки
        if isinstance(root_dirs, str):
            root_dirs = [(root_dirs, "hero" if "hero" in root_dirs.lower() else "item")]
        elif isinstance(root_dirs, list) and len(root_dirs) > 0 and isinstance(root_dirs[0], str):
            root_dirs = [(path, "hero" if "hero" in path.lower() else "item") for path in root_dirs]
        
        self.root_dirs = [r[0] if isinstance(r, tuple) else r for r in root_dirs]
        self.root_types = [r[1] if isinstance(r, tuple) else ("hero" if "hero" in r.lower() else "item") for r in root_dirs]
        
        self.transform = transform
        self.augment_single = augment_single
        self.filepaths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}  # 🔥 Обратный маппинг
        self.class_counts = {}
        self.class_types = {}  # 🔥 Тип класса (hero/item)
        self.class_source_dir = {}  # 🔥 Источник класса

        # Глобальный счётчик индексов классов
        global_class_idx = 0

        for root_dir, class_type in zip(self.root_dirs, self.root_types):
            if not os.path.exists(root_dir):
                print(f"⚠️ Пропущена директория: {root_dir}")
                continue

            class_images = defaultdict(list)
            for fname in os.listdir(root_dir):
                if not fname.endswith('.png'):
                    continue
                parts = fname.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                cls_name, _ = parts
                class_images[cls_name].append(os.path.join(root_dir, fname))

            # Фильтрация по min_images_per_class
            valid_classes = [cls for cls, paths in class_images.items() if len(paths) >= min_images_per_class]
            
            for cls in valid_classes:
                if cls not in self.class_to_idx:
                    self.class_to_idx[cls] = global_class_idx
                    self.idx_to_class[global_class_idx] = cls  # 🔥
                    self.class_types[global_class_idx] = class_type  # 🔥
                    self.class_source_dir[global_class_idx] = root_dir  # 🔥
                    global_class_idx += 1
                
                label = self.class_to_idx[cls]
                self.class_counts[label] = len(class_images[cls])
                
                for p in class_images[cls]:
                    self.filepaths.append(p)
                    self.labels.append(label)

        if not self.filepaths:
            raise ValueError("Не загружено ни одного изображения. Проверьте пути и формат имён файлов.")

        # 🔥 Статистика по типам
        hero_count = sum(1 for t in self.class_types.values() if t == "hero")
        item_count = sum(1 for t in self.class_types.values() if t == "item")
        
        print(f"✅ Загружено {len(self.class_to_idx)} уникальных классов, {len(self.filepaths)} изображений")
        print(f"   Героев: {hero_count}, Предметов: {item_count}")
        print(f"   Классов с 1 изображением: {sum(1 for c in self.class_counts.values() if c == 1)}")

    def get_class_info(self, idx):
        """🔥 Получить информацию о классе по индексу"""
        return {
            "idx": idx,
            "name": self.idx_to_class.get(idx, "unknown"),
            "type": self.class_types.get(idx, "unknown"),
            "source_dir": self.class_source_dir.get(idx, "unknown"),
            "count": self.class_counts.get(idx, 0)
        }

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # 🔥 Если класс имеет только 1 изображение и включена аугментация — применяем более сильные трансформы
        if self.augment_single and self.class_counts.get(label, 0) == 1:
            # Создаём копию трансформов с усиленной аугментацией
            strong_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return strong_transform(img), label
        
        if self.transform:
            img = self.transform(img)
        return img, label


# =========================
# PK BATCH SAMPLER (с поддержкой sampling with replacement)
# =========================
class PKBatchSampler(Sampler):
    def __init__(self, labels, p_classes, k_samples, shuffle=True, allow_replacement=True):
        """
        allow_replacement: если True, для классов с < k_samples изображений 
                          допускаем повторный сэмплинг одного изображения
        """
        self.labels = np.array(labels)
        self.p_classes = p_classes
        self.k_samples = k_samples
        self.shuffle = shuffle
        self.allow_replacement = allow_replacement

        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label].append(idx)

        # Классы, из которых МОЖНО взять хотя бы 1 пример
        self.valid_classes = list(self.class_indices.keys())
        if len(self.valid_classes) < p_classes:
            raise ValueError(f"Не хватает классов для PK-сэмплинга. Требуется {p_classes}, доступно {len(self.valid_classes)}")

        self.num_batches = len(self.valid_classes) // self.p_classes

    def __iter__(self):
        valid = self.valid_classes.copy()
        if self.shuffle:
            random.shuffle(valid)

        for i in range(self.num_batches):
            batch = []
            start = i * self.p_classes
            selected = valid[start:start + self.p_classes]

            for cls in selected:
                idxs = self.class_indices[cls].copy()
                
                # 🔥 Если изображений меньше чем нужно — сэмплим с возвратом
                if len(idxs) < self.k_samples and self.allow_replacement:
                    chosen = random.choices(idxs, k=self.k_samples)
                else:
                    if self.shuffle:
                        random.shuffle(idxs)
                    chosen = idxs[:self.k_samples]
                
                batch.extend(chosen)

            if self.shuffle:
                random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# =========================
# LOSS, MODEL, TRANSFORMS — без изменений (кроме добавления Normalize в base transform)
# =========================
class SupervisedContrastiveLoss(nn.Module):
    """
    Лучше работает с singleton-классами:
    - Positive: аугментации того же изображения + другие изображения того же класса
    - Negative: все остальные
    """
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        """
        features: (N, D) — L2-normalized эмбеддинги
        labels: (N,) — метки классов
        """
        device = features.device
        
        # Создаём маску позитивных пар
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        # Исключаем сам объект
        eye = torch.eye(mask.size(0), device=device)
        mask = mask * (1 - eye)
        
        # Косинусная схожесть всех пар
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Logits: вычитаем максимум для стабильности
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        # Softmax для негативов
        exp_logits = torch.exp(logits) * (1 - eye)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean log-likelihood для позитивов
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        loss = -mean_log_prob_pos.mean()
        return loss


class EmbeddingNet(nn.Module):
    def __init__(self, backbone_name="resnet18", embed_dim=256):
        super().__init__()
        if backbone_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError("Unsupported backbone")
        self.backbone = backbone
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # ← Добавили
            nn.Linear(512, embed_dim),
            nn.Dropout(0.1),  # ← Добавили
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding_head(x)
        return F.normalize(x, p=2, dim=1)
    
    def predict(self, image_input, transform, device="cpu", return_class_name=False, idx_to_class=None):
        """
        image_input: путь к файлу (str) или PIL.Image
        transform: ваши трансформы (train_transform)
        return_class_name: если True, вернет (вектор, название_класса)
        idx_to_class: словарь {0: "hero_name", 1: "item_name", ...}
        """
        self.eval()  # Режим инференса (отключает Dropout и BatchNorm)
        
        with torch.no_grad():  # Не считать градиенты для экономии памяти
            # 1. Загрузка изображения
            if isinstance(image_input, str):
                img = Image.open(image_input).convert('RGB')
            else:
                img = image_input

            # 2. Подготовка тензора
            tensor = transform(img).unsqueeze(0).to(device)

            # 3. Получение эмбеддинга
            embedding = self(tensor)

            # 4. Если нужна классификация (для MultiTask модели)
            if return_class_name and hasattr(self, 'classifier'):
                logits = self.classifier(embedding)
                pred_idx = torch.argmax(logits, dim=1).item()
                class_name = idx_to_class.get(pred_idx, f"Unknown_{pred_idx}")
                return embedding.squeeze(0).cpu(), class_name

            # 5. Возврат просто вектора
            return embedding.squeeze(0).cpu()


# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.4, 0.9), ratio=(0.75, 1.33)),  # агрессивнее
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.3),  # сильнее
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),  # добавили
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # добавили
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))], p=0.5),
    transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),  # добавили
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # добавили!
])


# =========================
# TRAIN / INFERENCE — без изменений
# =========================
def train(model, dataloader, optimizer, loss_fn, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            embeddings = model(images)
            loss = loss_fn(embeddings, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")


@torch.no_grad()
def compute_embedding(model, image_tensor):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    return model(image_tensor).squeeze(0)


def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


@torch.no_grad()
def find_similar(model, query_image, dataset, transform, device="cpu", top_k=10, filter_type=None):
    """
    🔥 Поиск похожих изображений с фильтрацией по типу
    
    Args:
        model: обученная модель
        query_image: путь к изображению или PIL.Image
        dataset: MetricDataset
        transform: трансформы
        device: устройство
        top_k: количество результатов
        filter_type: None | "hero" | "item" — фильтровать по типу
    """
    model.eval()
    
    # 1. Получаем эмбеддинг запроса
    if isinstance(query_image, str):
        query_img = Image.open(query_image).convert('RGB')
    else:
        query_img = query_image
    
    query_tensor = transform(query_img).unsqueeze(0).to(device)
    query_embedding = model(query_tensor).squeeze(0)
    
    # 2. Вычисляем эмбеддинги всех изображений (можно оптимизировать через DataLoader)
    similarities = []
    for idx in range(len(dataset)):
        # 🔥 Фильтрация по типу
        if filter_type is not None:
            label = dataset.labels[idx]
            if dataset.class_types.get(label) != filter_type:
                continue
        
        img = Image.open(dataset.filepaths[idx]).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        embedding = model(img_tensor).squeeze(0)
        
        sim = F.cosine_similarity(query_embedding.unsqueeze(0), embedding.unsqueeze(0)).item()
        similarities.append((idx, sim))
    
    # 3. Сортируем по схожести
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Возвращаем результаты с метаданными
    results = []
    for idx, sim in similarities[:top_k]:
        label = dataset.labels[idx]
        class_info = dataset.get_class_info(label)
        results.append({
            "filepath": dataset.filepaths[idx],
            "similarity": sim,
            "class_name": class_info["name"],
            "class_type": class_info["type"],
            "label": label
        })
    
    return results


def print_search_results(results):
    """🔥 Красивый вывод результатов поиска"""
    print("\n" + "="*80)
    print(f"{'Ранг':<5} {'Схожесть':<10} {'Тип':<8} {'Класс':<20} {'Путь'}")
    print("="*80)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['similarity']:<10.4f} {r['class_type']:<8} {r['class_name']:<20} {r['filepath']}")
    print("="*80 + "\n")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # 1. Датасет: поддержка нескольких папок + аугментация одиночных классов
    dataset = MetricDataset(
        root_dirs=DATASET_PATHS,
        transform=train_transform,
        min_images_per_class=1,        # 🔥 Разрешаем классы с 1 изображением
        augment_single=True            # 🔥 Включаем усиленную аугментацию для них
    )

    # 2. PK-сэмплинг: allow_replacement=True для работы с одиночными классами
    P, K = 40, 2
    batch_sampler = PKBatchSampler(dataset.labels, p_classes=P, k_samples=K, shuffle=True, allow_replacement=True)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)

    # 3. Модель
    model = EmbeddingNet(BACKBONE, EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-3)
    loss_fn = SupervisedContrastiveLoss(temperature=0.1)

    # 4. Обучение
    train(model, dataloader, optimizer, loss_fn, epochs=100)
