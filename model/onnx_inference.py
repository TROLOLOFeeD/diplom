import onnxruntime as ort
import numpy as np

class ONNXInferencer:
    def __init__(self, model_path: str = "model/dota_model.onnx"):
        # Явно указываем CPUExecutionProvider для работы на процессоре
        try:
            self.session = ort.InferenceSession(
                model_path, 
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            print(f"[ONNX] Model loaded from {model_path}")
        except Exception as e:
            print(f"[ONNX] Warning: Could not load model from {model_path}: {e}")
            print("[ONNX] Running in mock mode - predictions will be random")
            self.session = None
            self.input_name = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        features: numpy array формы (batch_size, input_dim)
        Возвращает массив вероятностей.
        """
        if self.session is None:
            # Mock prediction для тестирования без модели
            batch_size = features.shape[0]
            num_classes = 200  # Примерное количество предметов
            return np.random.rand(batch_size, num_classes).astype(np.float32)
        
        # Убеждаемся, что тип данных float32
        features = features.astype(np.float32)
        
        # Запуск инференса
        result = self.session.run(None, {self.input_name: features})
        return result[0] # Возвращаем первый (и единственный) выходной тензор
