import torch
import model.model as model_module # Импортируйте ваш класс модели

def export_model():
    # 1. Инициализация модели (замените input_dim на ваш реальный размер признаков)
    input_dim = 256  # Пример: размерность вектора признаков
    num_classes = 200 # Пример: количество предметов в Dota 2
    
    model = model_module.heroes_items_model(input_dim=input_dim, num_classes=num_classes)
    
    # 2. Загрузка весов (укажите путь к вашему .pth файлу)
    model.load_state_dict(torch.load("model/best_model.pth", map_location="cpu"))
    model.eval() # Важно: переводим в режим оценки

    # 3. Создание фиктивного входа (dummy input)
    # Форма должна совпадать с реальным входом: [batch_size, features]
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    # 4. Экспорт в ONNX
    onnx_file_path = "model/dota_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=17, # Современный opset
        do_constant_folding=True,
        input_names=['input_features'],
        output_names=['item_probabilities'],
        dynamic_axes={
            'input_features': {0: 'batch_size'},
            'item_probabilities': {0: 'batch_size'}
        }
    )
    print(f"Модель успешно экспортирована в {onnx_file_path}")

if __name__ == "__main__":
    export_model()
