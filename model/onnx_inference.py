import onnxruntime as ort
import numpy as np

class ONNXInferencer:
    def __init__(self, model_path: str = "model/dota_model.onnx"):
        # Явно указываем CPUExecutionProvider для работы на процессоре
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        features: numpy array формы (batch_size, input_dim)
        Возвращает массив вероятностей.
        """
        # Убеждаемся, что тип данных float32
        features = features.astype(np.float32)
        
        # Запуск инференса
        result = self.session.run(None, {self.input_name: features})
        return result[0] # Возвращаем первый (и единственный) выходной тензор
