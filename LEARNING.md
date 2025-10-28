# Learning Path — AlphaZero from Scratch


This file lists the concepts, ordered learning steps, and resources (tutorials, papers, books) so contributors can understand every part of the project.


## 1) Prerequisites (what you must know)
- Python (comfortable writing modular code)
- Basic linear algebra (vectors, matrices)
- Probability & statistics (expectation, distributions)
- Calculus basics (gradients)
- Basic machine learning concepts: supervised learning, loss, gradient descent
- PyTorch (or TensorFlow) fundamentals: tensors, autograd, DataLoader


## 2) Core topics to study (ordered)
1. **Reinforcement Learning (intro)**
- Markov Decision Processes (MDPs), rewards, episodes, policy vs value.
2. **Monte Carlo Tree Search (MCTS)**
- UCT, exploration vs exploitation, tree nodes, PUCT (policy-based prior) variant used in AlphaZero.
3. **Policy-Value Networks**
- Architecture principles: conv layers for board games, separate policy head and value head.
4. **Self-play & Data Generation**
- Generation of training targets from MCTS visit counts and outcome as value target.
5. **Training & Replay Buffer**
- Experience replay, sampling strategies (uniform, prioritized), batch formation.
6. **Evaluation and Model Selection**
- Elo-like rating or direct match counts; continuous evaluation pipeline.
7. **Scaling & Parallelism (advanced)**
- Running many self-play workers, asynchronous training, distributed replay buffer.


## 3) Practical tools & libraries
- **PyTorch** — model implementation and training.
- **NumPy** — numeric operations and board representation.
- **Joblib / multiprocessing / Ray** — parallel self-play or MCTS workers.
- **Matplotlib / imageio** — produce GIFs of game play.
- **Hydra / argparse** — experiment configuration management.


## 4) Recommended tutorials & papers
- Silver et al., *Mastering the game of Go with deep neural networks and tree search* (AlphaGo), and *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (AlphaZero) — read the AlphaZero paper carefully.
- YouTube tutorial: `AlphaZero from Scratch – Machine Learning Tutorial` (the video link in this repo).
- PyTorch official tutorial: Basics of training and models.
- Blog posts on MCTS and PUCT (search for ‘PUCT explained’).


## 5) Small learning projects (practice exercises)
- Implement Tic-Tac-Toe environment and a random-agent baseline.
- Implement simple MCTS without NN to play Connect Four.
- Add a tiny neural net that predicts random-play statistics on small boards.
- Combine NN + MCTS and run self-play for 1000 short games.


