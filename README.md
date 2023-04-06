
# Card29 Game using Reinforcement Learning

This repository contains an implementation of the card game Card29 using reinforcement learning techniques. The goal of this project is to train an AI agent to play Card29 at a high level using deep reinforcement learning.

## Introduction To Game
The 29 Points or 29 card game is a well-known trick-taking game that is popular in South Asia. Its origins can be traced back to the Jass family of games, which originated in the Netherlands and were brought to the Indian subcontinent by Dutch traders.

This game is widely played in many parts of northern India, including Bombay and West Bengal, as well as in Nepal and Bangladesh. One of the unique features of the game is its card ranking system, where the Jack (J) is the highest-ranking card, followed by the Nines (9), Aces (A), Tens (10), King (K), Queen (Q), and finally the Eights (8) and Sevens (7).

Interestingly, the 2s, 3s, 4s, 5s, and 6s are removed from the deck, making it a 32-card game. For more information on the rules of the 29 Point game

In the game of 29, four players form fixed partnerships with their partners sitting opposite each other. 

The objective of the game is to win tricks that contain high-value cards. The cards are assigned point values as follows: Jacks (3 points each), Nines (2 points each), Aces (1 point each), Tens (1 point each), and the other cards (K, Q, 8, 7) are worth no points. 

The total value of all the cards in the game is 28 points, but some versions of the game include an extra point for the last trick, bringing the total to 29, which explains the name of the game. 

While most players today do not count the extra point for the last trick, the game is still called 29 even when played without it, with a total of 28 points.
## Requirements

- Python 3.6 or higher

- TensorFlow 2.0 or higher

- Keras 2.0 or higher
## Images

![bots1](https://user-images.githubusercontent.com/60442599/230402452-24ff0a07-8242-4876-af72-7eb94db8be1f.png)

![bots2](https://user-images.githubusercontent.com/60442599/230402591-e8100edb-720e-4b8c-8f7e-48d49740467f.png)


## Getting Started

To run the Card29 game with reinforcement learning, first clone this repository:

```bash
git clone https://github.com/SubinMaharjan/Card29_RL.git
```

Next, install the required dependencies using pip:

```bash
pip install python
pip install tensorflow
pip install keras
```

To simulate and train the AI agent, run the following command:

```bash
python simulateGame.py
```

To first simulate and gather the game data, call GenerateData in simGame.py and run the following command:

```bash
python simulateGame.py
```

Now, to train the model run the following command:

```bash
python train.py
```
## Training Details

The AI agent is trained using a deep reinforcement learning algorithm called Deep Q-Networks (DQNs). The state of the game is represented as a vector of features, including the cards in the player's hand, the cards played so far, and the current score. The AI agent learns to play Card29 by maximizing its expected reward over time, using a combination of exploration and exploitation.

During training, the AI agent plays many games of Card29 against itself, using the current state of the game as input to the neural network and using the output to select an action. The neural network is updated after each batch using a variant of the Q-learning algorithm, which updates the expected reward for each action taken by the agent.
## Conclusion

Card29 is a challenging game that requires both strategic thinking and a bit of luck. By training an AI agent using deep reinforcement learning techniques, we hope to improve our understanding of how to play the game at a high level, and to develop new strategies for winning. We encourage other researchers and developers to build on our work and continue to explore the exciting possibilities of reinforcement learning.
