
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_inputs, n_outputs):
        # Inicializa pesos e bias aleatórios
        self.w = np.random.uniform(-1, 1, size=(n_inputs, n_outputs))
        self.bias = np.random.uniform(-1, 1, size=(n_outputs,))

    # Função de ativação sigmoide
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Treinamento
    def train(self, inputs, outputs, learning_rate=0.01, epochs=500):
        self.loss_history = []
        for _ in range(epochs):
            total_error = 0
            for i in range(len(inputs)):
                soma = np.dot(inputs[i], self.w) + self.bias
                y_pred = self.sigmoid(soma)

                # erro
                error = outputs[i] - y_pred
                total_error += np.mean(error**2)

                # atualização de pesos e bias
                self.w += learning_rate * np.outer(inputs[i], error)
                self.bias += learning_rate * error

            self.loss_history.append(total_error)

    # Previsão
    def predict(self, entrada):
        soma = np.dot(entrada, self.w) + self.bias
        return self.sigmoid(soma)


if __name__ == "__main__":
    # Base de treinamento
    dados_treino = [
        {"piso": 2, "poeira": 2, "obstaculos": 0, "potencia": 1, "velocidade": 3},
        {"piso": 1, "poeira": 8, "obstaculos": 2, "potencia": 3, "velocidade": 1},
        {"piso": 3, "poeira": 5, "obstaculos": 4, "potencia": 2, "velocidade": 1},
        {"piso": 2, "poeira": 1, "obstaculos": 1, "potencia": 1, "velocidade": 4},
        {"piso": 1, "poeira": 9, "obstaculos": 3, "potencia": 3, "velocidade": 2},
        {"piso": 3, "poeira": 6, "obstaculos": 0, "potencia": 2, "velocidade": 3},
        {"piso": 2, "poeira": 3, "obstaculos": 2, "potencia": 1, "velocidade": 2},
        {"piso": 1, "poeira": 7, "obstaculos": 1, "potencia": 3, "velocidade": 1},
        {"piso": 3, "poeira": 4, "obstaculos": 3, "potencia": 2, "velocidade": 4},
        {"piso": 2, "poeira": 0, "obstaculos": 0, "potencia": 1, "velocidade": 5},
    ]

    # Preparando dados
    inputs = np.array([[d["piso"], d["poeira"], d["obstaculos"]] for d in dados_treino])
    outputs = np.array([[d["potencia"], d["velocidade"]] for d in dados_treino])

    perceptron = Perceptron(n_inputs=3, n_outputs=2)

    print("Treinando aspirador inteligente...")
    perceptron.train(inputs, outputs, learning_rate=0.01, epochs=500)

    # Teste com os próprios dados
    print("\nResultados (saída prevista):")
    for entrada in inputs:
        pred = perceptron.predict(entrada)
        print(entrada, "->", pred.round(0))  # arredonda para inteiros

    # Gráfico do erro
    plt.plot(perceptron.loss_history)
    plt.xlabel("Épocas")
    plt.ylabel("Erro médio")
    plt.title("Treinamento do Perceptron")
    plt.show()
