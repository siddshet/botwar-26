"""
Connect 4 Plus — AlphaZero-lite submission.

Architecture: Dual-headed ResNet (policy + value).
Inference:    MCTS guided by the network, with transposition table.
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


# ═══════════════════════════════════════════════════════════════════════════════
#  Neural Network
# ═══════════════════════════════════════════════════════════════════════════════


class ResBlock(nn.Module):
    """Lightweight residual block."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class Connect4PlusNet(nn.Module):
    """
    Dual-headed CNN for Connect 4 Plus.

    Input:  (B, 3, 6, 7)  — my pieces / opponent pieces / neutral coin
    Output: policy (B, 7)  — log-probabilities over columns
            value  (B, 1)  — estimated game outcome in [-1, +1]
    """

    def __init__(self, num_res_blocks=4, channels=64):
        super().__init__()

        # Shared trunk
        self.conv_in = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 6 * 7, 7)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(6 * 7, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Shared trunk
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)

        # Policy head → log-softmax over 7 columns
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        # Value head → tanh scalar
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


# ═══════════════════════════════════════════════════════════════════════════════
#  Connect 4 Plus — fast board logic for MCTS (no PettingZoo dependency)
# ═══════════════════════════════════════════════════════════════════════════════


class Board:
    """
    Minimal Connect 4 Plus board for MCTS rollouts.

    Internal representation: 6×7 int8 array
        0 = empty, 1 = player-to-move, 2 = opponent, 3 = neutral coin
    """

    __slots__ = ("grid", "current_player")

    def __init__(self, grid=None, current_player=1):
        if grid is not None:
            self.grid = grid.copy()
        else:
            self.grid = np.zeros((6, 7), dtype=np.int8)
        self.current_player = current_player

    def copy(self):
        return Board(self.grid, self.current_player)

    def legal_moves(self):
        """Columns where the top cell is empty."""
        return [c for c in range(7) if self.grid[0, c] == 0]

    def drop(self, col):
        """Drop current_player's piece into col. Modifies board in-place."""
        for r in range(5, -1, -1):
            if self.grid[r, col] == 0:
                self.grid[r, col] = self.current_player
                break
        # Swap player
        self.current_player = 3 - self.current_player  # 1↔2

    def check_winner(self):
        """Return winning player (1 or 2), 0 for draw, -1 for ongoing."""
        for piece in (1, 2):
            # Horizontal
            for r in range(6):
                for c in range(4):
                    if (self.grid[r, c] == piece and self.grid[r, c+1] == piece
                            and self.grid[r, c+2] == piece and self.grid[r, c+3] == piece):
                        return piece
            # Vertical
            for c in range(7):
                for r in range(3):
                    if (self.grid[r, c] == piece and self.grid[r+1, c] == piece
                            and self.grid[r+2, c] == piece and self.grid[r+3, c] == piece):
                        return piece
            # Diagonal ↘
            for r in range(3):
                for c in range(4):
                    if (self.grid[r, c] == piece and self.grid[r+1, c+1] == piece
                            and self.grid[r+2, c+2] == piece and self.grid[r+3, c+3] == piece):
                        return piece
            # Diagonal ↗
            for r in range(3, 6):
                for c in range(4):
                    if (self.grid[r, c] == piece and self.grid[r-1, c+1] == piece
                            and self.grid[r-2, c+2] == piece and self.grid[r-3, c+3] == piece):
                        return piece
        # Draw?
        if all(self.grid[0, c] != 0 for c in range(7)):
            return 0
        return -1  # ongoing

    def to_tensor(self):
        """Convert to (1, 3, 6, 7) tensor from current player's perspective."""
        planes = np.zeros((3, 6, 7), dtype=np.float32)
        planes[0] = (self.grid == self.current_player).astype(np.float32)
        planes[1] = (self.grid == (3 - self.current_player)).astype(np.float32)
        planes[2] = (self.grid == 3).astype(np.float32)
        return torch.from_numpy(planes).unsqueeze(0)

    def state_key(self):
        """Hashable key for transposition table."""
        return (self.grid.tobytes(), self.current_player)


# ═══════════════════════════════════════════════════════════════════════════════
#  MCTS with Transposition Table
# ═══════════════════════════════════════════════════════════════════════════════

C_PUCT = 1.5          # exploration constant
DIRICHLET_ALPHA = 0.5  # root noise for exploration (training only)
DIRICHLET_FRAC = 0.25


class MCTSNode:
    """A node in the MCTS tree."""

    __slots__ = ("parent", "action", "prior", "visit_count",
                 "value_sum", "children", "is_expanded")

    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = []
        self.is_expanded = False

    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits):
        """Upper Confidence Bound for Trees (PUCT)."""
        exploration = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value() + exploration


class MCTS:
    """
    Monte Carlo Tree Search guided by a neural network.
    Uses a transposition table to cache network evaluations.
    """

    def __init__(self, network, device="cpu", num_simulations=600,
                 time_limit=4.0, add_noise=False):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.time_limit = time_limit  # hard cap in seconds
        self.add_noise = add_noise
        self.transposition_table = {}  # state_key → (policy, value)

    def _evaluate(self, board):
        """Get (policy, value) from network, using transposition table cache."""
        key = board.state_key()
        if key in self.transposition_table:
            return self.transposition_table[key]

        state_tensor = board.to_tensor().to(self.device)
        with torch.no_grad():
            log_policy, value = self.network(state_tensor)
        policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
        val = value.item()

        # Mask illegal moves and re-normalize
        legal = board.legal_moves()
        mask = np.zeros(7, dtype=np.float32)
        for m in legal:
            mask[m] = 1.0
        policy = policy * mask
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Fallback: uniform over legal moves
            policy = mask / mask.sum()

        self.transposition_table[key] = (policy, val)
        return policy, val

    def _expand(self, node, board):
        """Expand a leaf node using network evaluation."""
        policy, value = self._evaluate(board)
        legal = board.legal_moves()

        for action in legal:
            child = MCTSNode(parent=node, action=action, prior=policy[action])
            node.children.append(child)
        node.is_expanded = True

        return value

    def _select_child(self, node):
        """Pick child with highest UCB score."""
        best_score = -float("inf")
        best_child = None
        for child in node.children:
            score = child.ucb_score(node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _backpropagate(self, node, value):
        """Walk back up the tree, flipping value at each level."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # opponent's perspective
            node = node.parent

    def search(self, board):
        """
        Run MCTS from the given board position.
        Returns: action_probs (7,) - visit count distribution over columns
        """
        self.transposition_table.clear()

        root = MCTSNode()
        self._expand(root, board)

        # Dirichlet noise at root for exploration during training
        if self.add_noise and root.children:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.children))
            for child, n in zip(root.children, noise):
                child.prior = (1 - DIRICHLET_FRAC) * child.prior + DIRICHLET_FRAC * n

        start_time = time.time()

        for sim in range(self.num_simulations):
            # Time check every 50 simulations
            if sim % 50 == 0 and sim > 0:
                if time.time() - start_time > self.time_limit:
                    break

            node = root
            sim_board = board.copy()

            # SELECT — traverse tree using UCB until we hit an unexpanded node
            while node.is_expanded and node.children:
                node = self._select_child(node)
                sim_board.drop(node.action)

            # Check terminal state
            winner = sim_board.check_winner()
            if winner >= 0:
                # Terminal node
                if winner == 0:
                    value = 0.0  # draw
                else:
                    # winner is from the perspective of the player who just moved
                    # which is the PARENT's player. So from current node's
                    # player perspective, they lost.
                    value = -1.0
                self._backpropagate(node, value)
                continue

            # EXPAND & EVALUATE
            value = self._expand(node, sim_board)

            # BACKPROPAGATE — value is from current player's perspective
            self._backpropagate(node, -value)

        # Build action probability distribution from visit counts
        action_probs = np.zeros(7, dtype=np.float32)
        for child in root.children:
            action_probs[child.action] = child.visit_count

        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        return action_probs


# ═══════════════════════════════════════════════════════════════════════════════
#  Bot — competition entry point
# ═══════════════════════════════════════════════════════════════════════════════


class Bot:
    def __init__(self):
        self.device = torch.device("cpu")
        torch.set_num_threads(2)  # small parallelism for conv ops

        self.model = Connect4PlusNet(num_res_blocks=4, channels=64)
        self.model.eval()

        weights_path = os.path.join(os.path.dirname(__file__), "model.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            self.model.load_state_dict(state_dict)

        # MCTS: 600 sims with 4-second hard time limit (leaves 1s margin)
        self.mcts = MCTS(
            self.model, device=self.device,
            num_simulations=600, time_limit=4.0, add_noise=False
        )

    def act(self, observation):
        board_obs = observation["observation"]   # (6, 7, 3)
        action_mask = observation["action_mask"]  # (7,)

        # Build internal Board from the observation
        board = Board()
        board.grid = np.zeros((6, 7), dtype=np.int8)
        board.grid[board_obs[:, :, 0] == 1] = 1  # my pieces
        board.grid[board_obs[:, :, 1] == 1] = 2  # opponent pieces
        board.grid[board_obs[:, :, 2] == 1] = 3  # neutral coin
        board.current_player = 1  # always "me" from observation perspective

        legal = [c for c in range(7) if action_mask[c] == 1]
        if not legal:
            return 0  # should never happen

        # If only one legal move, return immediately
        if len(legal) == 1:
            return legal[0]

        # Run MCTS
        try:
            action_probs = self.mcts.search(board)

            # Mask illegal moves (safety)
            for c in range(7):
                if action_mask[c] == 0:
                    action_probs[c] = 0.0

            total = action_probs.sum()
            if total > 0:
                return int(np.argmax(action_probs))
        except Exception:
            pass

        # Fallback: use raw network policy
        state_tensor = torch.from_numpy(board_obs.astype(np.float32))
        state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            log_policy, _ = self.model(state_tensor)
            policy = torch.exp(log_policy).squeeze(0).numpy()

        for c in range(7):
            if action_mask[c] == 0:
                policy[c] = 0.0

        return int(np.argmax(policy))
