# Perceptron - Aspirador Inteligente

## 1. Descrição do Problema
O objetivo deste trabalho é implementar um **Perceptron simples** para simular o funcionamento de um aspirador inteligente.  
O modelo recebe **3 parâmetros de entrada**:  
- Tipo de piso  
- Nível de poeira  
- Quantidade de obstáculos  

E gera como **saída**:  
- Potência de sucção  
- Velocidade de deslocamento  

## 2. Resolução do Problema
O perceptron foi desenvolvido em Python utilizando:  
- **NumPy** para operações matemáticas  
- **Matplotlib** para visualizar o erro de treinamento  

A rede neural possui:  
- **3 entradas** (piso, poeira, obstáculos)  
- **2 saídas** (potência, velocidade)  
- **Função de ativação sigmoide**  
- **Ajuste de pesos e bias** via aprendizado supervisionado  

Os dados de treinamento foram definidos com base em situações do aspirador em diferentes ambientes.

## Procedimento de execução

1. **Pré-requisitos**  
   - Ter o Python 3 instalado.  
   - Instalar as bibliotecas usadas:  
     ```bash
     pip install numpy matplotlib
     ```

2. **Salvar o código**  
   - Copie o código em um arquivo chamado `aspirador_perceptron.py`.

3. **Rodar o programa**  
   - No terminal, vá até a pasta onde salvou o arquivo e execute:  
     ```bash
     python aspirador_perceptron.py
     ```

4. **O que acontece ao rodar**  
   - O perceptron será treinado por 500 épocas.  
   - No terminal, aparecerão as previsões do modelo para os dados de treino, arredondadas para inteiros.  
   - Um gráfico será exibido mostrando o **erro médio durante o treinamento**. Ele mostra como o perceptron foi aprendendo com o tempo.

5. **Como interpretar os resultados**  
   - Os números impressos mostram a potência e velocidade que o perceptron prevê para cada entrada.  
   - O gráfico deve mostrar o erro diminuindo com o passar das épocas. Se o erro não diminuir, significa que o modelo não aprendeu direito.

