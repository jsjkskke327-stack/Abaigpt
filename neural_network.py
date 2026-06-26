import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Инициализируем генератор случайных чисел
        np.random.seed(1)
        # Веса для 3 входных нейронов и 1 выходного (значения от -1 до 1)
        self.weights = 2 * np.random.random((3, 1)) - 1

    # Функция активации (Сигмоида) для нормализации значений от 0 до 1
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Производная сигмоиды для расчета гр��диента
    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    # Прямое распространение (скалярное произведение входов на веса)
    def predict(self, inputs):
        activation = np.dot(inputs, self.weights)
        return self._sigmoid(activation)

    # Обучение методом обратного распространения ошибки
    def train(self, train_inputs, train_outputs, epochs):
        for epoch in range(epochs):
            # Массив для накопления корректировок весов в текущей эпохе
            adjustments = np.zeros_like(self.weights)

            for i in range(len(train_inputs)):
                input_row = train_inputs[i]
                target = train_outputs[i]

                # 1. Получаем предсказание
                output = self.predict(input_row)

                # 2. Считаем ошибку
                error = target - output

                # 3. Считаем локальный градиент (ошибка * производную)
                delta = error * self._sigmoid_derivative(output)

                # 4. Накапливаем изменения для каждого веса
                adjustments += input_row.reshape(-1, 1) * delta

            # Обновляем веса после прохода по всей выборке
            self.weights += adjustments

if __name__ == "__main__":
    network = NeuralNetwork()

    # Обучающая выборка (4 примера, по 3 входа в каждом)
    train_inputs = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0]
    ])

    # Правильные ответы (целевые значения)
    train_outputs = np.array([1.0, 1.0, 0.0, 0.0])

    # Запуск обучения (20 000 эпох)
    network.train(train_inputs, train_outputs, 20000)

    print("Веса после обучения:")
    for i, weight in enumerate(network.weights.flatten()):
        print(f"Вес {i}: {weight:.4f}")

    # Тест на новых данных
    new_data = np.array([1.0, 0.0, 1.0])
    prediction = network.predict(new_data)

    print(f"\nПрогноз для теста: {prediction:.4f} (Ожидалось ближе к 1)")
