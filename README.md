# AlphaZero-Chess
A self-learning chess AI inspired by DeepMind’s AlphaZero. This project implements reinforcement learning, Monte Carlo Tree Search (MCTS), and a neural network to train an agent that learns to play chess from scratch without human data.


![tictactoe](https://raw.githubusercontent.com/foersterrobert/AlphaZero/master/assets/tictactoe.gif)
![connectfour](https://raw.githubusercontent.com/foersterrobert/AlphaZero/master/assets/connectfour.gif)
## Quick overview (what this project does — step-by-step)


1. **Game environment(s)**
- Implements deterministic, turn-based games (e.g., Tic-Tac-Toe, Connect Four) with a unified API: `state`, `legal_moves()`, `apply_move()`, `is_terminal()`, `winner()`.


2. **MCTS (Monte Carlo Tree Search)**
- A neural-guided MCTS: the search uses a neural network to provide a prior probability distribution over moves and a value estimate for leaf evaluation.
- MCTS outputs a visit-count distribution used as training targets and for action selection during self-play.


3. **Neural network model**
- A small convolutional / fully-connected policy-value network (PyTorch) that maps game states → (policy logits, scalar value).
- Models are saved as `model_*.pt`.


4. **Self-play**
- The current model plays many games against itself using MCTS to generate training examples: (state, MCTS-policy, game-outcome) triplets.
- Data augmentation (if applicable) is applied (symmetries for board games).


5. **Training loop**
- A dataset pool (replay buffer) stores recent self-play games.
- Batches are sampled to update the policy-value network using a combined loss: policy cross-entropy + value MSE + optional regularization.


6. **Evaluation/Matches**
- New model checkpoints are evaluated against past bests or a fixed baseline. Wins/draws/loss counts decide promotion of checkpoints.


7. **Tweaks & experiments**
- Hyperparameter tuning, parallel self-play (AlphaParallel), alternating model architectures (model_2.pt, model_7_... etc.), optimizer variations saved as `optimizer_*.pth`.


## Repo structure (suggested)

### Some Helpful Resources
* AlphaZero-Paper: https://arxiv.org/pdf/1712.01815.pdf
* Paper-Walkthrough: https://youtu.be/0slFo1rV0EM
* MCTS-Explained: https://youtu.be/UXW2yZndl7U
* AlphaZero-Explained: https://youtu.be/62nq4Zsn8vc

❤️
