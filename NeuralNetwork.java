import java.util.Random;

public class NeuralNetwork {
    private double[] weights;

    // Конструктор: инициализируем 3 случайных веса от -1 до 1
    public NeuralNetwork() {
        weights = new double[3];
        Random rand = new Random(1); // Фиксированный сид для воспроизводимости
        for (int i = 0; i < weights.length; i++) {
            weights[i] = 2 * rand.nextDouble() - 1;
        }
    }

    // Функция активации (Сигмоида)
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Производная сигмоиды для расчета градиента
    private double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    // Прямое распространение (скалярное произведение входов на веса)
    public double predict(double[] inputs) {
        double activation = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            activation += inputs[i] * weights[i];
        }
        return sigmoid(activation);
    }

    // Обучение методом обратного распространения ошибки
    public void train(double[][] trainInputs, double[] trainOutputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Массив для накопления корректировок весов в текущей эпохе
            double[] adjustments = new double[3];

            for (int i = 0; i < trainInputs.length; i++) {
                double[] inputRow = trainInputs[i];
                double target = trainOutputs[i];

                // 1. Получаем предсказание
                double output = predict(inputRow);

                // 2. Считаем ошибку
                double error = target - output;

                // 3. Считаем локальный градиент (ошибка * производную)
                double delta = error * sigmoidDerivative(output);

                // 4. Накапливаем изменения для каждого веса
                for (int j = 0; j < weights.length; j++) {
                    adjustments[j] += inputRow[j] * delta;
                }
            }

            // Обновляем веса после прохода ��о всей выборке
            for (int j = 0; j < weights.length; j++) {
                weights[j] += adjustments[j];
            }
        }
    }

    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork();

        // Обучающая выборка (4 примера, по 3 входа в каждом)
        double[][] trainInputs = {
            {1.0, 0.0, 0.0},
            {1.0, 1.0, 1.0},
            {0.0, 1.0, 1.0},
            {0.0, 0.0, 1.0}
        };

        // Правильные ответы (целевые значения)
        double[] trainOutputs = {1.0, 1.0, 0.0, 0.0};

        // Запуск обучения (20 000 эпох)
        network.train(trainInputs, trainOutputs, 20000);

        System.out.println("Веса после обучения:");
        for (int i = 0; i < network.weights.length; i++) {
            System.out.printf("Вес %d: %.4f%n", i, network.weights[i]);
        }

        // Тест на новых данных
        double[] newData = {1.0, 0.0, 1.0};
        double prediction = network.predict(newData);

        System.out.printf("%nПрогноз для теста: %.4f (Ожидалось ближе к 1)%n", prediction);
    }
}
