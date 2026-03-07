"""
AlphaZero-style self-play training for Connect 4 Plus.

Each episode:
  1. Play a full game using MCTS to choose moves
  2. Store (state, mcts_policy, game_outcome) tuples
  3. Train the network on mini-batches from a replay buffer

Saves weights to  weights/model.safetensors  every 500 episodes.

Usage:
    python train_selfplay.py
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

# Add project root so connect4plus can be found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from connect4plus.game import env as make_env

# Import our architecture and MCTS
from model import Connect4PlusNet, Board, MCTS

# Also import ruleBot for evaluation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sample_submission"))
from ruleBot.model import RuleBot  # noqa: E402


# ─── Hyper-parameters ────────────────────────────────────────────────────────

EPISODES         = 20_000       # total training games
BATCH_SIZE       = 256          # mini-batch size
LR               = 2e-3        # learning rate (higher for AZ-style)
LR_DROP_AT       = 10_000      # drop LR after this many episodes
LR_DROP_FACTOR   = 0.1         # multiply LR by this
WEIGHT_DECAY     = 1e-4         # L2 regularization
REPLAY_SIZE      = 50_000       # max (state, policy, value) entries
TRAIN_STEPS      = 4            # gradient steps per episode
NUM_SIMS         = 200          # MCTS simulations per move during training
SAVE_EVERY       = 500          # save weights every N episodes
EVAL_EVERY       = 100          # evaluate every N episodes
TEMPERATURE      = 1.0          # for first 15 moves use this temp
TEMP_DROP_MOVE   = 15           # after this many moves, use temp → 0

# Opponent mix for evaluation (not training — training is pure self-play)
OPP_RULE_PROB    = 0.7

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


# ─── Replay Buffer ───────────────────────────────────────────────────────────


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0

    def push(self, state, policy, value):
        """state: (3,6,7) numpy, policy: (7,) numpy, value: float"""
        item = (state, policy, value)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return (
            torch.from_numpy(np.array(states)),
            torch.from_numpy(np.array(policies)),
            torch.tensor(values, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─── Self-Play ───────────────────────────────────────────────────────────────


def self_play_game(network, device, num_sims=200):
    """
    Play one game using MCTS self-play.
    Returns list of (state_tensor, mcts_policy, current_player) tuples,
    plus the game outcome.
    """
    board = Board()

    # Place a random neutral coin on the bottom row
    neutral_col = random.randint(0, 6)
    board.grid[5, neutral_col] = 3

    mcts = MCTS(network, device=device, num_simulations=num_sims,
                time_limit=30.0, add_noise=True)

    history = []  # (state_planes, mcts_policy, current_player)
    move_count = 0

    while True:
        legal = board.legal_moves()
        if not legal:
            break

        winner = board.check_winner()
        if winner >= 0:
            break

        # Get MCTS policy
        action_probs = mcts.search(board)

        # Store training data
        state_planes = board.to_tensor().squeeze(0).numpy()  # (3, 6, 7)
        history.append((state_planes, action_probs.copy(), board.current_player))

        # Choose action with temperature
        if move_count < TEMP_DROP_MOVE:
            # Sample proportional to visit counts (with temperature)
            temp_probs = action_probs ** (1.0 / TEMPERATURE)
            temp_probs /= temp_probs.sum() if temp_probs.sum() > 0 else 1.0
            action = np.random.choice(7, p=temp_probs)
        else:
            # Greedy
            action = int(np.argmax(action_probs))

        board.drop(action)
        move_count += 1

    # Determine outcome
    winner = board.check_winner()
    # winner: 0 = draw, 1 = player 1 won, 2 = player 2 won, -1 = ongoing (shouldn't happen)

    # Build training data with correct value assignments
    training_data = []
    for state, policy, player in history:
        if winner == 0 or winner == -1:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        training_data.append((state, policy, value))

    return training_data, winner


# ─── Training ────────────────────────────────────────────────────────────────


def train_step(network, optimizer, replay, device):
    """One gradient step. Returns loss or None."""
    if len(replay) < BATCH_SIZE:
        return None

    states, target_policies, target_values = replay.sample(BATCH_SIZE)
    states = states.to(device)
    target_policies = target_policies.to(device)
    target_values = target_values.to(device)

    log_policy, value = network(states)
    value = value.squeeze(1)

    # Policy loss: cross-entropy with MCTS policy targets
    policy_loss = -torch.sum(target_policies * log_policy, dim=1).mean()

    # Value loss: MSE with game outcome
    value_loss = F.mse_loss(value, target_values)

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


# ─── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_vs_rulebot(network, device, num_games=20):
    """Evaluate by playing vs ruleBot using MCTS. Returns win rate."""
    rule_bot = RuleBot()
    mcts = MCTS(network, device=device, num_simulations=100,
                time_limit=30.0, add_noise=False)
    wins = 0

    for game_idx in range(num_games):
        # Alternate seats
        learner_seat = "player_0" if game_idx % 2 == 0 else "player_1"

        ev = make_env()
        ev.reset()
        learner_reward = 0.0

        for agent in ev.agent_iter():
            obs, reward, termination, truncation, info = ev.last()
            done = termination or truncation

            if agent == learner_seat and done:
                learner_reward = reward

            if done:
                action = None
            elif agent == learner_seat:
                # Build Board from observation
                board_obs = obs["observation"]
                board = Board()
                board.grid[board_obs[:, :, 0] == 1] = 1
                board.grid[board_obs[:, :, 1] == 1] = 2
                board.grid[board_obs[:, :, 2] == 1] = 3
                board.current_player = 1

                legal = [c for c in range(7) if obs["action_mask"][c] == 1]
                if len(legal) == 1:
                    action = legal[0]
                else:
                    probs = mcts.search(board)
                    for c in range(7):
                        if obs["action_mask"][c] == 0:
                            probs[c] = 0.0
                    action = int(np.argmax(probs))
            else:
                action = rule_bot.act(obs)

            ev.step(action)

        if learner_reward > 0:
            wins += 1
        ev.close()

    return wins / num_games


def evaluate_vs_random(network, device, num_games=20):
    """Evaluate using raw network policy (fast). Returns win rate."""
    wins = 0

    for game_idx in range(num_games):
        learner_seat = "player_0" if game_idx % 2 == 0 else "player_1"

        ev = make_env()
        ev.reset()
        learner_reward = 0.0

        for agent in ev.agent_iter():
            obs, reward, termination, truncation, info = ev.last()
            done = termination or truncation

            if agent == learner_seat and done:
                learner_reward = reward

            if done:
                action = None
            elif agent == learner_seat:
                # Use raw network (fast)
                board_obs = obs["observation"]
                mask = obs["action_mask"]
                tensor = torch.from_numpy(board_obs.astype(np.float32))
                tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    log_p, _ = network(tensor)
                    p = torch.exp(log_p).squeeze(0).cpu().numpy()
                for c in range(7):
                    if mask[c] == 0:
                        p[c] = 0.0
                action = int(np.argmax(p))
            else:
                mask = obs["action_mask"]
                valid = [c for c in range(7) if mask[c] == 1]
                action = random.choice(valid)

            ev.step(action)

        if learner_reward > 0:
            wins += 1
        ev.close()

    return wins / num_games


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"AlphaZero-lite training on {device}")

    network = Connect4PlusNet(num_res_blocks=4, channels=64).to(device)

    # Warm-start from existing weights if available
    weights_path = WEIGHTS_DIR / "model.safetensors"
    if weights_path.exists():
        try:
            state_dict = {k: v.clone() for k, v in load_file(weights_path).items()}
            network.load_state_dict(state_dict)
            print(f"  Loaded existing weights from {weights_path}")
        except Exception as e:
            print(f"  Could not load weights (architecture changed?): {e}")
            print("  Starting from scratch.")

    optimizer = torch.optim.Adam(network.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    replay = ReplayBuffer(REPLAY_SIZE)

    WEIGHTS_DIR.mkdir(exist_ok=True)
    start_time = time.time()
    recent_losses = []
    best_eval = 0.0

    for episode in range(1, EPISODES + 1):
        # LR schedule
        if episode == LR_DROP_AT:
            for pg in optimizer.param_groups:
                pg["lr"] = LR * LR_DROP_FACTOR
            print(f"  LR dropped to {LR * LR_DROP_FACTOR}")

        # Self-play game
        network.eval()
        game_data, winner = self_play_game(network, device, num_sims=NUM_SIMS)

        # Add all positions to replay buffer (with data augmentation: mirror)
        for state, policy, value in game_data:
            replay.push(state, policy, value)
            # Mirror augmentation: flip columns
            mirrored_state = state[:, :, ::-1].copy()
            mirrored_policy = policy[::-1].copy()
            replay.push(mirrored_state, mirrored_policy, value)

        # Train
        network.train()
        for _ in range(TRAIN_STEPS):
            loss = train_step(network, optimizer, replay, device)
            if loss is not None:
                recent_losses.append(loss)

        # Progress log
        if episode % 50 == 0:
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            recent_losses.clear()
            elapsed = (time.time() - start_time) / 60
            print(f"Ep {episode:>5} | Loss {avg_loss:.4f} | "
                  f"Replay {len(replay):>6} | Winner p{winner} | {elapsed:.1f}m")

        # Evaluate
        if episode % EVAL_EVERY == 0:
            network.eval()
            rand_wr = evaluate_vs_random(network, device, num_games=20)
            rule_wr = evaluate_vs_rulebot(network, device, num_games=10)
            combined = 0.3 * rand_wr + 0.7 * rule_wr
            elapsed = (time.time() - start_time) / 60

            print(f"  EVAL | vs Random {rand_wr:.0%} | vs RuleBot {rule_wr:.0%} | "
                  f"Combined {combined:.0%} | Best {best_eval:.0%} | {elapsed:.1f}m")

            if combined > best_eval:
                best_eval = combined
                save_file(network.state_dict(), WEIGHTS_DIR / "model.safetensors")
                print(f"  ★ New best! ({combined:.0%}) Saved.")

        # Periodic checkpoint
        if episode % SAVE_EVERY == 0:
            save_file(network.state_dict(), WEIGHTS_DIR / "checkpoint.safetensors")
            print(f"  → Checkpoint saved (ep {episode})")

    # Final save
    save_file(network.state_dict(), WEIGHTS_DIR / "model.safetensors")
    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining complete in {elapsed:.1f} minutes.")
    print(f"  Best combined score: {best_eval:.0%}")
    print(f"  Weights: {WEIGHTS_DIR / 'model.safetensors'}")


if __name__ == "__main__":
    main()
