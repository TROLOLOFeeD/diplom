import json
import gc
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# 1. Data Processing with JSON Integration
# ============================================================================

class DotaDataProcessor:
    """Обработка датасета Dota 2 с использованием heroes.json и items.json."""
    
    def __init__(self, heroes_json_path: str, items_json_path: str):
        # Загрузка JSON данных
        with open(heroes_json_path, 'r') as f:
            self.heroes_data = json.load(f)
        
        with open(items_json_path, 'r') as f:
            self.items_data = json.load(f)
        
        # Создаем маппинги
        self.hero_id_to_features = {}
        self.item_name_to_features = {}
        self.item_costs = {}
        
        self._process_heroes_json()
        self._process_items_json()
        
        # Словари для энкодеров
        self.hero_to_idx = {}
        self.item_to_idx = {}
        
    def _process_heroes_json(self):
        """Извлекает признаки героев из JSON с защитой от None."""
        for hero_id_str, hero_data in self.heroes_data.items():
            hero_id = int(hero_id_str)
            
            # primary_attr
            primary_attr = hero_data.get('primary_attr') or 'str'
            attr_onehot = [0, 0, 0]
            if primary_attr == 'str':
                attr_onehot = [1, 0, 0]
            elif primary_attr == 'agi':
                attr_onehot = [0, 1, 0]
            elif primary_attr == 'int':
                attr_onehot = [0, 0, 1]
            
            # attack_type
            attack_type = hero_data.get('attack_type') or 'Melee'
            attack_onehot = [1, 0] if attack_type == 'Melee' else [0, 1]
            
            # roles (может быть None)
            roles = hero_data.get('roles') or []
            role_list = ['Carry', 'Support', 'Nuker', 'Disabler', 'Escape', 'Initiator', 'Durable', 'Pusher']
            roles_onehot = [1 if role in roles else 0 for role in role_list]
            
            features = []
            features.extend(attr_onehot)      # 3
            features.extend(attack_onehot)    # 2
            features.extend(roles_onehot)     # 8
            
            # Числовые статы с защитой от None
            def safe_get(d, key, default=0):
                v = d.get(key)
                return v if isinstance(v, (int, float)) else default
            
            features.append(safe_get(hero_data, 'base_str') / 50.0)
            features.append(safe_get(hero_data, 'base_agi') / 50.0)
            features.append(safe_get(hero_data, 'base_int') / 50.0)
            features.append(safe_get(hero_data, 'str_gain') / 5.0)
            features.append(safe_get(hero_data, 'agi_gain') / 5.0)
            features.append(safe_get(hero_data, 'int_gain') / 5.0)
            features.append(safe_get(hero_data, 'attack_range') / 1000.0)
            features.append(safe_get(hero_data, 'move_speed') / 400.0)
            features.append(safe_get(hero_data, 'base_armor') / 10.0)
            features.append(safe_get(hero_data, 'base_health') / 1000.0)
            features.append(safe_get(hero_data, 'base_mana') / 500.0)
            
            # 🔑 Защита длины
            expected_len = 24
            if len(features) < expected_len:
                features.extend([0.0] * (expected_len - len(features)))
            elif len(features) > expected_len:
                features = features[:expected_len]
                
            self.hero_id_to_features[hero_id] = np.array(features, dtype=np.float32)
    
    def _process_items_json(self):
        """Извлекает признаки предметов из JSON с защитой от None/false."""
        for item_name, item_data in self.items_data.items():
            cost = item_data.get('cost') or 0
            self.item_costs[item_name] = cost
            
            features = []
            
            # Стоимость (нормализованная)
            features.append(cost / 10000.0)
            
            # Количество компонентов (защита от None/false/отсутствия)
            components = item_data.get('components') or []
            features.append(len(components) / 5.0)
            
            # Cooldown (может быть None, false или 0)
            cd = item_data.get('cd') or 0
            if not isinstance(cd, (int, float)):
                cd = 0
            features.append(cd / 200.0)
            
            # Количество атрибутов
            attrib = item_data.get('attrib') or []
            features.append(len(attrib) / 20.0)
            
            # Behavior (one-hot для основных типов)
            behavior = item_data.get('behavior') or []
            behavior_types = ['No Target', 'Unit Target', 'Point Target', 'Instant Cast', 'Channelled']
            behavior_onehot = [1 if b in behavior else 0 for b in behavior_types]
            features.extend(behavior_onehot)
            
            # Has active ability
            abilities = item_data.get('abilities') or []
            has_active = 1 if any(
                isinstance(a, dict) and a.get('type') == 'active' 
                for a in abilities
            ) else 0
            features.append(has_active)
            
            # Is recipe
            is_recipe = 1 if item_name.startswith('recipe_') else 0
            features.append(is_recipe)
            
            # 🔑 Защита: если по какой-то причине features короче ожидаемого
            expected_len = 11
            if len(features) < expected_len:
                features.extend([0.0] * (expected_len - len(features)))
            elif len(features) > expected_len:
                features = features[:expected_len]
                
            self.item_name_to_features[item_name] = np.array(features, dtype=np.float32)
    
    def calculate_net_worth(self, row: pd.Series, player_prefix: str) -> int:
        """Рассчитывает net_worth игрока на основе его предметов."""
        total_cost = 0
        for i in range(6):
            item_col = f"{player_prefix}_item_{i}"
            item_name = row.get(item_col, '0')
            if item_name and item_name != '0':
                total_cost += self.item_costs.get(item_name, 0)
        
        if row.get(f"{player_prefix}_aghanims_scepter", 0) == 1:
            total_cost += 4200
        if row.get(f"{player_prefix}_aghanims_shard", 0) == 1:
            total_cost += 1400
        
        return total_cost
    
    def build_vocabularies_from_csv(self, csv_path: str, chunksize: int = 150_000):
        """Строит словари, не загружая весь CSV в память."""
        heroes = set()
        items = set(['0'])
        player_prefixes = [f"p{i}" for i in range(5)] + [f"p{128+i}" for i in range(5)]
        
        # Читаем только нужные колонки для скорости
        cols_to_read = [
            *[f"{p}_hero_id" for p in player_prefixes],
            *[f"{p}_item_{s}" for p in player_prefixes for s in range(6)]
        ]
        
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=cols_to_read):
            for prefix in player_prefixes:
                heroes.update(chunk[f"{prefix}_hero_id"].dropna().unique())
                for s in range(6):
                    items.update(chunk[f"{prefix}_item_{s}"].dropna().unique())
                    
        self.hero_to_idx = {hero: idx + 1 for idx, hero in enumerate(sorted(heroes))}
        self.item_to_idx = {item: idx for idx, item in enumerate(sorted(items))}
        print(f"✅ Vocabulary built: {len(self.hero_to_idx)} heroes, {len(self.item_to_idx)} items")
    
    def encode_heroes(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Кодирует героев для всех игроков."""
        batch_size = len(df)
        radiant_heroes = np.zeros((batch_size, 5), dtype=np.int64)
        dire_heroes = np.zeros((batch_size, 5), dtype=np.int64)
        
        for i in range(5):
            col = f"p{i}_hero_id"
            radiant_heroes[:, i] = df[col].map(self.hero_to_idx).fillna(0).astype(np.int64)
            
            col = f"p{128+i}_hero_id"
            dire_heroes[:, i] = df[col].map(self.hero_to_idx).fillna(0).astype(np.int64)
        
        return radiant_heroes, dire_heroes
    
    def extract_hero_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Извлекает JSON-признаки героев для всех игроков."""
        batch_size = len(df)
        num_features = 24  # Размер вектора признаков героя
        
        radiant_hero_feats = np.zeros((batch_size, 5, num_features), dtype=np.float32)
        dire_hero_feats = np.zeros((batch_size, 5, num_features), dtype=np.float32)
        
        for i in range(5):
            # Radiant
            col = f"p{i}_hero_id"
            for idx, hero_id in enumerate(df[col].values):
                if hero_id in self.hero_id_to_features:
                    radiant_hero_feats[idx, i, :] = self.hero_id_to_features[hero_id]
            
            # Dire
            col = f"p{128+i}_hero_id"
            for idx, hero_id in enumerate(df[col].values):
                if hero_id in self.hero_id_to_features:
                    dire_hero_feats[idx, i, :] = self.hero_id_to_features[hero_id]
        
        return radiant_hero_feats, dire_hero_feats
    
    def encode_inventories(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Кодирует инвентари для всех игроков."""
        batch_size = len(df)
        radiant_items = np.zeros((batch_size, 5, 6), dtype=np.int64)
        dire_items = np.zeros((batch_size, 5, 6), dtype=np.int64)
        
        for i in range(5):
            for slot in range(6):
                col = f"p{i}_item_{slot}"
                radiant_items[:, i, slot] = df[col].map(self.item_to_idx).fillna(0).astype(np.int64)
            
            for slot in range(6):
                col = f"p{128+i}_item_{slot}"
                dire_items[:, i, slot] = df[col].map(self.item_to_idx).fillna(0).astype(np.int64)
        
        return radiant_items, dire_items
    
    def extract_item_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Извлекает JSON-признаки предметов для всех игроков."""
        batch_size = len(df)
        num_features = 11  # Размер вектора признаков предмета
        
        radiant_item_feats = np.zeros((batch_size, 5, 6, num_features), dtype=np.float32)
        dire_item_feats = np.zeros((batch_size, 5, 6, num_features), dtype=np.float32)
        
        for i in range(5):
            # Radiant
            for slot in range(6):
                col = f"p{i}_item_{slot}"
                for idx, item_name in enumerate(df[col].values):
                    if item_name in self.item_name_to_features:
                        radiant_item_feats[idx, i, slot, :] = self.item_name_to_features[item_name]
            
            # Dire
            for slot in range(6):
                col = f"p{128+i}_item_{slot}"
                for idx, item_name in enumerate(df[col].values):
                    if item_name in self.item_name_to_features:
                        dire_item_feats[idx, i, slot, :] = self.item_name_to_features[item_name]
        
        return radiant_item_feats, dire_item_feats
    
    def extract_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Извлекает числовые признаки: время, уровни, net_worth, aghanim flags."""
        batch_size = len(df)
        features = []
        
        features.append(df['game_time'].values.reshape(-1, 1) / 5400.0)
        
        for i in range(5):
            prefix = f"p{i}"
            features.append(df[f"{prefix}_level"].values.reshape(-1, 1) / 30.0)
            features.append(df.apply(lambda row: self.calculate_net_worth(row, prefix), axis=1).values.reshape(-1, 1))
            features.append(df[f"{prefix}_aghanims_scepter"].values.reshape(-1, 1))
            features.append(df[f"{prefix}_aghanims_shard"].values.reshape(-1, 1))
            
            prefix = f"p{128+i}"
            features.append(df[f"{prefix}_level"].values.reshape(-1, 1) / 30.0)
            features.append(df.apply(lambda row: self.calculate_net_worth(row, prefix), axis=1).values.reshape(-1, 1))
            features.append(df[f"{prefix}_aghanims_scepter"].values.reshape(-1, 1))
            features.append(df[f"{prefix}_aghanims_shard"].values.reshape(-1, 1))
        
        return np.hstack(features)
    
    def process_chunk(self, df_chunk: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Обрабатывает один чанк, возвращает тензоры с экономией памяти."""
        radiant_heroes, dire_heroes = self.encode_heroes(df_chunk)
        radiant_hero_feats, dire_hero_feats = self.extract_hero_features(df_chunk)
        radiant_items, dire_items = self.encode_inventories(df_chunk)
        radiant_item_feats, dire_item_feats = self.extract_item_features(df_chunk)
        numerical_features = self.extract_numerical_features(df_chunk)
        print(
            "Numerical:",
            numerical_features.min(),
            numerical_features.max(),
            np.isnan(numerical_features).sum(),
            np.isinf(numerical_features).sum()
        )
        
        # Нормализация net_worth внутри чанка
        net_worth_cols = [i for i in range(1, numerical_features.shape[1], 4)]
        for col_idx in net_worth_cols:
            nw = numerical_features[:, col_idx]
            if nw.max() > 0:
                if np.any(nw < -1):
                    print("❌ Invalid net worth")
                    print("min =", nw.min())
                    print("max =", nw.max())
                    raise RuntimeError("Negative net worth")
                nw_log = np.log1p(nw)
                mean, std = nw_log.mean(), nw_log.std() + 1e-8
                numerical_features[:, col_idx] = (nw_log - mean) / std
                
        # 🔑 Ключевая экономия: int16/int32 вместо int64, float32 для стабильности
        return {
            'radiant_heroes': torch.tensor(radiant_heroes, dtype=torch.int32),
            'dire_heroes': torch.tensor(dire_heroes, dtype=torch.int32),
            'radiant_hero_feats': torch.tensor(radiant_hero_feats, dtype=torch.float32),
            'dire_hero_feats': torch.tensor(dire_hero_feats, dtype=torch.float32),
            'radiant_items': torch.tensor(radiant_items, dtype=torch.long),
            'dire_items': torch.tensor(dire_items, dtype=torch.long),
            'radiant_item_feats': torch.tensor(radiant_item_feats, dtype=torch.float32),
            'dire_item_feats': torch.tensor(dire_item_feats, dtype=torch.float32),
            'numerical_features': torch.tensor(numerical_features, dtype=torch.float32),
            'labels': torch.tensor(df_chunk['radiant_win'].values, dtype=torch.float32)
        }


# ============================================================================
# 2. TabTransformer with JSON Features
# ============================================================================

class TabTransformer(nn.Module):
    """TabTransformer с интеграцией JSON-признаков героев и предметов."""
    
    def __init__(
        self,
        num_heroes: int,
        num_items: int,
        hero_feat_dim: int = 24,
        item_feat_dim: int = 11,
        hero_emb_dim: int = 32,
        item_emb_dim: int = 32,
        dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Learnable embeddings
        self.hero_emb = nn.Embedding(num_heroes + 1, hero_emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(num_items + 1, item_emb_dim, padding_idx=0)
        
        # Feature encoders для JSON-признаков
        self.hero_feat_encoder = nn.Sequential(
            nn.Linear(hero_feat_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        self.item_feat_encoder = nn.Sequential(
            nn.Linear(item_feat_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        
        # Project embeddings to model dimension
        self.hero_proj = nn.Linear(hero_emb_dim, dim)
        self.item_proj = nn.Linear(item_emb_dim, dim)
        self.num_proj = nn.Linear(4, dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output head
        self.fc_out = nn.Linear(dim * 3 + 41, 1)
        
    def encode_team(
        self,
        heroes: torch.Tensor,  # [B, 5]
        hero_feats: torch.Tensor,  # [B, 5, hero_feat_dim]
        items: torch.Tensor,   # [B, 5, 6]
        item_feats: torch.Tensor,  # [B, 5, 6, item_feat_dim]
        player_numerical: torch.Tensor  # [B, 5, 4]
    ) -> torch.Tensor:
        """Кодирует одну команду с использованием JSON-признаков."""
        B = heroes.shape[0]
        
        # Hero embeddings + JSON features
        hero_learned = self.hero_proj(self.hero_emb(heroes))  # [B, 5, dim]
        assert torch.isfinite(hero_learned).all(), "hero_learned contains NaN/Inf"

        hero_json = self.hero_feat_encoder(hero_feats)  # [B, 5, dim]
        assert torch.isfinite(hero_json).all(), "hero_json contains NaN/Inf"
        hero_repr = hero_learned + hero_json  # [B, 5, dim]
        
        # Item embeddings + JSON features
        # Суммируем по слотам
        item_learned = self.item_emb(items).sum(dim=2)  # [B, 5, item_emb_dim]
        item_learned = self.item_proj(item_learned)  # [B, 5, dim]
        assert torch.isfinite(item_learned).all(), "item_learned contains NaN/Inf"
        
        item_json = self.item_feat_encoder(item_feats).sum(dim=2)  # [B, 5, dim]
        assert torch.isfinite(item_json).all(), "item_json contains NaN/Inf"
        item_repr = item_learned + item_json  # [B, 5, dim]
        
        # Numerical features
        num_repr = self.num_proj(player_numerical)  # [B, 5, dim]
        
        # Комбинируем
        team_repr = (hero_repr + item_repr + num_repr) / 3.0  # [B, 5, dim]
        assert torch.isfinite(team_repr).all(), "team_repr contains NaN/Inf"
        
        return team_repr
    
    def forward(
        self,
        radiant_heroes: torch.Tensor,
        dire_heroes: torch.Tensor,
        radiant_hero_feats: torch.Tensor,
        dire_hero_feats: torch.Tensor,
        radiant_items: torch.Tensor,
        dire_items: torch.Tensor,
        radiant_item_feats: torch.Tensor,
        dire_item_feats: torch.Tensor,
        numerical_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass с JSON-признаками."""
        B = radiant_heroes.shape[0]

        game_time = numerical_features[:, :1]      # (B, 1)
        player_features = numerical_features[:, 1:] # (B, 40)

        num_split = player_features.view(B, 10, 4)

        radiant_num = num_split[:, :5, :]
        dire_num = num_split[:, 5:, :]
        
        radiant_repr = self.encode_team(
            radiant_heroes, radiant_hero_feats,
            radiant_items, radiant_item_feats,
            radiant_num
        )
        dire_repr = self.encode_team(
            dire_heroes, dire_hero_feats,
            dire_items, dire_item_feats,
            dire_num
        )
        
        all_repr = torch.cat([radiant_repr, dire_repr], dim=1)
        transformed = self.transformer(all_repr)
        
        pooled = transformed.mean(dim=1)
        radiant_pooled = radiant_repr.mean(dim=1)
        dire_pooled = dire_repr.mean(dim=1)
        
        final_repr = torch.cat([
            pooled,
            radiant_pooled,
            dire_pooled,
            numerical_features
        ], dim=1)
        
        logits = self.fc_out(final_repr).squeeze(-1)
        
        return logits


# ============================================================================
# 3. Inference Engine (Updated)
# ============================================================================

class DotaInferenceEngine:
    """Движок для инференса с JSON-признаками."""
    
    def __init__(
        self,
        model: TabTransformer,
        data_processor: DotaDataProcessor,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.processor = data_processor
        self.device = device
    
    def prepare_state(
        self,
        game_time: int,
        radiant_heroes: List[int],
        dire_heroes: List[int],
        radiant_items: List[List[str]],
        dire_items: List[List[str]],
        radiant_levels: List[int],
        dire_levels: List[int],
        radiant_agh_scepter: List[int],
        radiant_agh_shard: List[int],
        dire_agh_scepter: List[int],
        dire_agh_shard: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Подготавливает состояние матча для инференса."""
        
        data = {'game_time': [game_time]}
        
        for i in range(5):
            prefix = f"p{i}"
            data[f"{prefix}_hero_id"] = [radiant_heroes[i]]
            data[f"{prefix}_level"] = [radiant_levels[i]]
            data[f"{prefix}_aghanims_scepter"] = [radiant_agh_scepter[i]]
            data[f"{prefix}_aghanims_shard"] = [radiant_agh_shard[i]]
            for slot in range(6):
                data[f"{prefix}_item_{slot}"] = [radiant_items[i][slot]]
        
        for i in range(5):
            prefix = f"p{128+i}"
            data[f"{prefix}_hero_id"] = [dire_heroes[i]]
            data[f"{prefix}_level"] = [dire_levels[i]]
            data[f"{prefix}_aghanims_scepter"] = [dire_agh_scepter[i]]
            data[f"{prefix}_aghanims_shard"] = [dire_agh_shard[i]]
            for slot in range(6):
                data[f"{prefix}_item_{slot}"] = [dire_items[i][slot]]
        
        df = pd.DataFrame(data)
        processed = self.processor.process_dataset(df)
        
        return {k: v.to(self.device) for k, v in processed.items()}
    
    def predict_win_probability(self, state: Dict[str, torch.Tensor]) -> float:
        """Прямой инференс."""
        with torch.no_grad():
            logits = self.model(
                state['radiant_heroes'],
                state['dire_heroes'],
                state['radiant_hero_feats'],
                state['dire_hero_feats'],
                state['radiant_items'],
                state['dire_items'],
                state['radiant_item_feats'],
                state['dire_item_feats'],
                state['numerical_features']
            )
            prob = torch.sigmoid(logits).item()
        return prob
    
    def counterfactual_predict(
        self,
        state: Dict[str, torch.Tensor],
        target_player_slot: int,
        event_item_name: str
    ) -> Tuple[float, float]:
        """Контрфактуальный инференс."""
        original_prob = self.predict_win_probability(state)
        
        cf_state = {k: v.clone() for k, v in state.items()}
        
        if target_player_slot < 5:
            team_idx = target_player_slot
            items_tensor = cf_state['radiant_items']
            item_feats_tensor = cf_state['radiant_item_feats']
        else:
            team_idx = target_player_slot - 128
            items_tensor = cf_state['dire_items']
            item_feats_tensor = cf_state['dire_item_feats']
        
        player_items = items_tensor[0, team_idx]
        empty_slot = (player_items == 0).nonzero(as_tuple=True)[0]
        
        if len(empty_slot) > 0:
            slot_to_replace = empty_slot[0].item()
        else:
            slot_to_replace = 5
        
        item_idx = self.processor.item_to_idx.get(event_item_name, 0)
        items_tensor[0, team_idx, slot_to_replace] = item_idx
        
        # Обновляем JSON-признаки предмета
        if event_item_name in self.processor.item_name_to_features:
            item_feats_tensor[0, team_idx, slot_to_replace, :] = torch.tensor(
                self.processor.item_name_to_features[event_item_name],
                dtype=torch.float32
            )
        
        cf_prob = self.predict_win_probability(cf_state)
        
        return original_prob, cf_prob


# ============================================================================
# 4. Training Pipeline (Updated)
# ============================================================================

class DotaDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
        self.size = data['labels'].shape[0]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'radiant_heroes': self.data['radiant_heroes'][idx],
            'dire_heroes': self.data['dire_heroes'][idx],
            'radiant_hero_feats': self.data['radiant_hero_feats'][idx],
            'dire_hero_feats': self.data['dire_hero_feats'][idx],
            'radiant_items': self.data['radiant_items'][idx],
            'dire_items': self.data['dire_items'][idx],
            'radiant_item_feats': self.data['radiant_item_feats'][idx],
            'dire_item_feats': self.data['dire_item_feats'][idx],
            'numerical_features': self.data['numerical_features'][idx],
            'label': self.data['labels'][idx]
        }


def train_model_chunked(
    model: TabTransformer,
    processor: DotaDataProcessor,
    csv_path: str,
    epochs: int = 5,
    chunksize: int = 150_000,
    batch_size: int = 256,
    lr: float = 1e-4,
    device: str = 'cpu'
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        chunks_processed = 0
        
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            if chunk.empty:
                continue
                
            # 1. Превращаем чанк в тензоры
            data = processor.process_chunk(chunk)
            for key, value in data.items():
                if value.dtype.is_floating_point:
                    if torch.isnan(value).any():
                        print(f"❌ NaN found in {key}")
                        raise RuntimeError(f"NaN in {key}")

                    if torch.isinf(value).any():
                        print(f"❌ Inf found in {key}")
                        raise RuntimeError(f"Inf in {key}")

            chunk_dataset = DotaDataset(data)
            chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            
            # 2. Обучение на чанке
            chunk_loss = 0.0
            chunk_samples = 0
            
            for batch in chunk_loader:
                optimizer.zero_grad()
                logits = model(
                    batch['radiant_heroes'].to(device),
                    batch['dire_heroes'].to(device),
                    batch['radiant_hero_feats'].to(device),
                    batch['dire_hero_feats'].to(device),
                    batch['radiant_items'].to(device),
                    batch['dire_items'].to(device),
                    batch['radiant_item_feats'].to(device),
                    batch['dire_item_feats'].to(device),
                    batch['numerical_features'].to(device)
                )
                if torch.isnan(logits).any():
                    print("❌ NaN in logits")
                    print("Logits min:", logits.min().item())
                    print("Logits max:", logits.max().item())
                    raise RuntimeError("NaN logits")

                if torch.isinf(logits).any():
                    raise RuntimeError("Inf logits")
                
                loss = criterion(logits, batch['label'].to(device))
                if torch.isnan(loss):
                    print("❌ Loss is NaN")
                    print("labels unique:", torch.unique(batch["label"]))
                    print("logits min:", logits.min().item())
                    print("logits max:", logits.max().item())
                    raise RuntimeError("Loss NaN")

                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print("NaN before backward:", name)
                        raise RuntimeError

                with torch.autograd.detect_anomaly():
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"❌ NaN gradient: {name}")
                            raise RuntimeError("NaN gradient")

                optimizer.step()
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"❌ NaN weights: {name}")
                        raise RuntimeError("NaN weights")
                
                chunk_loss += loss.item() * batch['label'].shape[0]
                chunk_samples += batch['label'].shape[0]
                
            epoch_loss += chunk_loss
            total_samples += chunk_samples
            chunks_processed += 1
            
            # 🔑 3. Жёсткая очистка памяти после каждого чанка
            del data, chunk_dataset, chunk_loader
            if device == 'cuda': torch.cuda.empty_cache()
            gc.collect()
            
            if chunks_processed % 3 == 0:
                print(f"  📦 Epoch {epoch+1} | Chunks processed: {chunks_processed} | Samples: {total_samples}")
                
        avg_loss = epoch_loss / total_samples if total_samples > 0 else 0
        print(f"✅ Epoch {epoch+1}/{epochs} finished. Avg Loss: {avg_loss:.4f}")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, f"checkpoint_epoch_{epoch+1}.pt")

        print(f"💾 Saved checkpoint_epoch_{epoch+1}.pt")


# ============================================================================
# 5. Model Saving and Loading Utilities
# ============================================================================

import pickle
from pathlib import Path

def save_final_model(
    model: TabTransformer,
    processor: DotaDataProcessor,
    save_dir: str = "./saved_models",
    model_name: str = "dota_model"
):
    """
    Сохраняет финальную модель и все необходимые метаданные.
    
    Args:
        model: Обученная модель TabTransformer
        processor: DotaDataProcessor с построенными словарями
        save_dir: Директория для сохранения
        model_name: Базовое имя для файлов
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Сохраняем веса модели
    model_file = save_path / f"{model_name}_weights.pt"
    torch.save(model.state_dict(), model_file)
    print(f"✅ Model weights saved to {model_file}")
    
    # 2. Сохраняем метаданные
    metadata_file = save_path / f"{model_name}_metadata.pkl"
    metadata = {
        # Словари для энкодеров
        "hero_to_idx": processor.hero_to_idx,
        "item_to_idx": processor.item_to_idx,
        
        # Маппинги признаков из JSON
        "hero_id_to_features": processor.hero_id_to_features,
        "item_name_to_features": processor.item_name_to_features,
        "item_costs": processor.item_costs,
        
        # Конфигурация модели
        "model_config": {
            "num_heroes": len(processor.hero_to_idx),
            "num_items": len(processor.item_to_idx),
            "hero_feat_dim": 24,
            "item_feat_dim": 11,
            "hero_emb_dim": 32,
            "item_emb_dim": 32,
            "dim": 64,
            "depth": 4,
            "heads": 4,
            "dropout": 0.1
        },
        
        # Информация о данных
        "data_info": {
            "hero_features_count": 24,
            "item_features_count": 11,
            "numerical_features_count": 41  # 1 (time) + 10 players * 4 features
        }
    }
    
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    print(f"✅ Metadata saved to {metadata_file}")
    
    # 3. Сохраняем информацию о версии
    info_file = save_path / f"{model_name}_info.txt"
    with open(info_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Saved at: {pd.Timestamp.now()}\n")
        f.write(f"Heroes in vocabulary: {len(processor.hero_to_idx)}\n")
        f.write(f"Items in vocabulary: {len(processor.item_to_idx)}\n")
        f.write(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    print(f"✅ Info saved to {info_file}")
    
    return {
        "model_file": str(model_file),
        "metadata_file": str(metadata_file),
        "info_file": str(info_file)
    }


def load_model_for_inference(
    save_dir: str = "./saved_models",
    model_name: str = "dota_model",
    device: str = "cpu"
) -> Tuple[TabTransformer, DotaDataProcessor]:
    """
    Загружает модель и метаданные для инференса.
    
    Args:
        save_dir: Директория с сохраненной моделью
        model_name: Базовое имя модели
        device: Устройство для загрузки ('cpu' или 'cuda')
    
    Returns:
        Tuple[model, processor]: Загруженная модель и процессор данных
    """
    save_path = Path(save_dir)
    
    # 1. Загружаем метаданные
    metadata_file = save_path / f"{model_name}_metadata.pkl"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    
    print(f"✅ Loaded metadata from {metadata_file}")
    
    # 2. Создаем процессор и восстанавливаем словари
    # Создаем пустые JSON файлы для инициализации (не используются, но нужны для конструктора)
    temp_heroes_json = save_path / "temp_heroes.json"
    temp_items_json = save_path / "temp_items.json"
    
    # Создаем минимальные JSON файлы
    with open(temp_heroes_json, "w") as f:
        json.dump({}, f)
    with open(temp_items_json, "w") as f:
        json.dump({}, f)
    
    processor = DotaDataProcessor(str(temp_heroes_json), str(temp_items_json))
    
    # Восстанавливаем словари из метаданных
    processor.hero_to_idx = metadata["hero_to_idx"]
    processor.item_to_idx = metadata["item_to_idx"]
    processor.hero_id_to_features = metadata["hero_id_to_features"]
    processor.item_name_to_features = metadata["item_name_to_features"]
    processor.item_costs = metadata["item_costs"]
    
    # Удаляем временные файлы
    temp_heroes_json.unlink()
    temp_items_json.unlink()
    
    # 3. Создаем модель с сохраненной конфигурацией
    model_config = metadata["model_config"]
    model = TabTransformer(
        num_heroes=model_config["num_heroes"],
        num_items=model_config["num_items"],
        hero_feat_dim=model_config["hero_feat_dim"],
        item_feat_dim=model_config["item_feat_dim"],
        hero_emb_dim=model_config["hero_emb_dim"],
        item_emb_dim=model_config["item_emb_dim"],
        dim=model_config["dim"],
        depth=model_config["depth"],
        heads=model_config["heads"],
        dropout=model_config["dropout"]
    )
    
    # 4. Загружаем веса
    model_file = save_path / f"{model_name}_weights.pt"
    if not model_file.exists():
        raise FileNotFoundError(f"Model weights file not found: {model_file}")
    
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded model weights from {model_file}")
    print(f"✅ Model ready for inference on {device}")
    
    return model, processor


# ============================================================================
# 6. Updated Training Pipeline with Final Model Saving
# ============================================================================

def train_model_chunked_with_save(
    model: TabTransformer,
    processor: DotaDataProcessor,
    csv_path: str,
    epochs: int = 5,
    chunksize: int = 150_000,
    batch_size: int = 256,
    lr: float = 1e-4,
    device: str = 'cpu',
    save_final: bool = True,
    save_dir: str = "./saved_models"
):
    """
    Обучает модель с сохранением финальной версии.
    
    Дополнительные параметры:
        save_final: Сохранять ли финальную модель
        save_dir: Директория для сохранения
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_loss = float('inf')
    
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        chunks_processed = 0
        
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            if chunk.empty:
                continue
                
            # 1. Превращаем чанк в тензоры
            data = processor.process_chunk(chunk)
            for key, value in data.items():
                if value.dtype.is_floating_point:
                    if torch.isnan(value).any():
                        print(f"❌ NaN found in {key}")
                        raise RuntimeError(f"NaN in {key}")

                    if torch.isinf(value).any():
                        print(f"❌ Inf found in {key}")
                        raise RuntimeError(f"Inf in {key}")

            chunk_dataset = DotaDataset(data)
            chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            
            # 2. Обучение на чанке
            chunk_loss = 0.0
            chunk_samples = 0
            
            for batch in chunk_loader:
                optimizer.zero_grad()
                logits = model(
                    batch['radiant_heroes'].to(device),
                    batch['dire_heroes'].to(device),
                    batch['radiant_hero_feats'].to(device),
                    batch['dire_hero_feats'].to(device),
                    batch['radiant_items'].to(device),
                    batch['dire_items'].to(device),
                    batch['radiant_item_feats'].to(device),
                    batch['dire_item_feats'].to(device),
                    batch['numerical_features'].to(device)
                )
                if torch.isnan(logits).any():
                    print("❌ NaN in logits")
                    print("Logits min:", logits.min().item())
                    print("Logits max:", logits.max().item())
                    raise RuntimeError("NaN logits")

                if torch.isinf(logits).any():
                    raise RuntimeError("Inf logits")
                
                loss = criterion(logits, batch['label'].to(device))
                if torch.isnan(loss):
                    print("❌ Loss is NaN")
                    print("labels unique:", torch.unique(batch["label"]))
                    print("logits min:", logits.min().item())
                    print("logits max:", logits.max().item())
                    raise RuntimeError("Loss NaN")

                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print("NaN before backward:", name)
                        raise RuntimeError

                with torch.autograd.detect_anomaly():
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"❌ NaN gradient: {name}")
                            raise RuntimeError("NaN gradient")

                optimizer.step()
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"❌ NaN weights: {name}")
                        raise RuntimeError("NaN weights")
                
                chunk_loss += loss.item() * batch['label'].shape[0]
                chunk_samples += batch['label'].shape[0]
                
            epoch_loss += chunk_loss
            total_samples += chunk_samples
            chunks_processed += 1
            
            # 🔑 3. Жёсткая очистка памяти после каждого чанка
            del data, chunk_dataset, chunk_loader
            if device == 'cuda': torch.cuda.empty_cache()
            gc.collect()
            
            if chunks_processed % 3 == 0:
                print(f"  📦 Epoch {epoch+1} | Chunks processed: {chunks_processed} | Samples: {total_samples}")
                
        avg_loss = epoch_loss / total_samples if total_samples > 0 else 0
        print(f"✅ Epoch {epoch+1}/{epochs} finished. Avg Loss: {avg_loss:.4f}")
        
        # Сохраняем чекпоинт эпохи
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, f"checkpoint_epoch_{epoch+1}.pt")
        print(f"💾 Saved checkpoint_epoch_{epoch+1}.pt")
        
        # Отслеживаем лучшую модель
        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_final:
                print(f"🏆 New best model! Loss: {best_loss:.4f}")
    
    # 🔑 Сохраняем финальную модель после всех эпох
    if save_final:
        print("\n" + "="*60)
        print("💾 Saving final model...")
        print("="*60)
        saved_files = save_final_model(
            model=model,
            processor=processor,
            save_dir=save_dir,
            model_name="dota_tabtransformer_final"
        )
        print("\n✅ Training complete! Model saved successfully.")
        print(f"📁 Files saved in: {save_dir}")
        for key, path in saved_files.items():
            print(f"   - {key}: {path}")


# ============================================================================
# 7. Example Usage with Model Loading
# ============================================================================

if __name__ == "__main__":
    csv_path = "/home/asd/diplom/dotaapi/new_dataset.csv"
    print(f"📊 Доступно RAM: {os.popen('free -m').readlines()[1].split()[6]} MB")
    
    # Режим: обучение или загрузка готовой модели
    MODE = "train"  # "train" или "inference"
    
    if MODE == "train":
        # 1. Сначала сканируем CSV для словарей
        print("1️⃣ Building vocabulary...")
        processor = DotaDataProcessor(
            "/home/asd/diplom/dotaapi/heroes (1).json",
            "/home/asd/diplom/dotaapi/items.json"
        )
        processor.build_vocabularies_from_csv(csv_path, chunksize=150_000)
        
        # 2. Создаём модель
        print("2️⃣ Initializing model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = TabTransformer(
            num_heroes=len(processor.hero_to_idx),
            num_items=len(processor.item_to_idx),
            hero_feat_dim=24, item_feat_dim=11,
            hero_emb_dim=32, item_emb_dim=32,
            dim=64, depth=4, heads=4, dropout=0.1
        )
        
        # 3. Обучение с сохранением финальной модели
        print("3️⃣ Training with chunks...")
        train_model_chunked_with_save(
            model=model,
            processor=processor,
            csv_path=csv_path,
            epochs=5,
            chunksize=150_000,
            batch_size=256,
            lr=1e-3,
            device=device,
            save_final=True,
            save_dir="./saved_models"
        )
    
    elif MODE == "inference":
        # Загрузка готовой модели
        print("🔄 Loading saved model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, processor = load_model_for_inference(
            save_dir="./saved_models",
            model_name="dota_tabtransformer_final",
            device=device
        )
    
    # 4. Инференс (работает в обоих режимах)
    print("\n" + "="*60)
    print("🎮 Running inference...")
    print("="*60)
    
    inference_engine = DotaInferenceEngine(model, processor, device)
    
    # Пример состояния матча
    state = inference_engine.prepare_state(
        game_time=677,
        radiant_heroes=[40, 35, 111, 104, 6],
        dire_heroes=[86, 114, 58, 63, 30],
        radiant_items=[
            ['magic_wand', 'spirit_vessel', 'guardian_greaves', '0', '0', '0'],
            ['wraith_band', 'boots_of_elves', 'power_treads', 'maelstrom', 'lesser_crit', 'demon_edge'],
            ['bracer', 'boots', 'void_stone', 'phylactery', 'glimmer_cape', 'gem'],
            ['quelling_blade', 'bracer', 'magic_wand', 'phase_boots', 'blade_mail', 'blink'],
            ['wraith_band', 'power_treads', 'falcon_blade', 'dragon_lance', 'boots_of_elves', 'manta']
        ],
        dire_items=[
            ['phylactery', 'boots', 'kaya', 'ultimate_scepter', '0', '0'],
            ['wraith_band', 'boots', 'diffusal_blade', 'maelstrom', 'great_famango', 'hyperstone'],
            ['circlet', 'ancient_janggo', 'arcane_boots', '0', '0', '0'],
            ['falcon_blade', 'magic_wand', 'dragon_lance', 'great_famango', 'boots_of_elves', 'power_treads'],
            ['magic_wand', 'arcane_boots', 'force_staff', '0', '0', '0']
        ],
        radiant_levels=[17, 17, 18, 15, 16],
        dire_levels=[15, 15, 10, 16, 12],
        radiant_agh_scepter=[0, 0, 0, 0, 0],
        radiant_agh_shard=[0, 0, 0, 0, 1],
        dire_agh_scepter=[1, 0, 0, 0, 0],
        dire_agh_shard=[0, 0, 0, 0, 1]
    )
    
    # Прямой инференс
    prob = inference_engine.predict_win_probability(state)
    print(f"\n📊 Current Radiant win probability: {prob:.4f}")
    
    # Контрфактуальный инференс
    original_prob, cf_prob = inference_engine.counterfactual_predict(
        state,
        target_player_slot=0,
        event_item_name='shadow_amulet'
    )
    
    delta = cf_prob - original_prob
    print(f"\n🔮 Counterfactual: What if p0 (Oracle) buys 'shadow_amulet'?")
    print(f"   Original probability: {original_prob:.4f}")
    print(f"   Counterfactual probability: {cf_prob:.4f}")
    print(f"   Delta: {delta:+.4f}")
