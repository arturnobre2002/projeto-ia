# Fruit Catcher AI

Projeto desenvolvido para a UC de Inteligência Artificial (2024/25) no ISCTE. O objetivo é controlar uma cesta para apanhar frutas e evitar bombas disfarçadas de frutas utilizando agentes inteligentes.

## O que é o projeto?
O projeto utiliza duas técnicas principais de IA para automatizar o jogo:

* **Árvore de Decisão:** Um classificador treinado do zero (baseado em Ganho de Informação) que analisa as características dos objetos como nome, cor e formato para distinguir frutas de bombas.
* **Rede Neuronal + Algoritmo Genético:** A movimentação da cesta é controlada por uma rede neuronal *feed-forward*. Os pesos desta rede são otimizados através de um algoritmo genético que utiliza seleção (elitismo), crossover e mutação.

## Estrutura de Ficheiros
* `dt.py`: Implementação da Árvore de Decisão.
* `nn.py`: Arquitetura da Rede Neuronal.
* `genetic.py`: Lógica do Algoritmo Genético.
* `game.py`: Motor do jogo desenvolvido em Pygame.
* `main.py`: Interface principal para treino e execução.

## Como Correr
Certifica-te de que tens o `pygame` e o `numpy` instalados.

1.  **Treinar a IA:**
    ```bash
    python main.py --train --population 100 --generations 100
    ```
    *(Gera o ficheiro `best_individual.txt` com os melhores pesos encontrados)*

2.  **Ver a IA a jogar:**
    ```bash
    python main.py
    ```

3.  **Testar sem gráficos (Headless):**
    ```bash
    python main.py --headless
    ```

---
*Realizado por: Joana Guerra, nº 122712 e Artur Nobre, nº 99087
