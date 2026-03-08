"""
Connect 4 Plus — Bitboard Alpha-Beta Solver.

Inspired by PascalPons/connect4, adapted for the neutral coin variant.
Pure algorithmic approach — no neural network, no training needed.

Techniques:
  - Bitboard representation (two int bitmasks + neutral bitmask)
  - Negamax alpha-beta pruning
  - Transposition table with upper/lower bounds
  - Move ordering: center-first + winning spots heuristic
  - Non-losing moves filter
  - Iterative deepening with time limit
  - Double-threat (trap) detection
"""

import time

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  Bitboard Position
# ═══════════════════════════════════════════════════════════════════════════════

WIDTH = 7
HEIGHT = 6
H1 = HEIGHT + 1
TOTAL_CELLS = WIDTH * HEIGHT

BOTTOM_MASK = 0
for _c in range(WIDTH):
    BOTTOM_MASK |= 1 << (_c * H1)

BOARD_MASK = 0
for _c in range(WIDTH):
    for _r in range(HEIGHT):
        BOARD_MASK |= 1 << (_c * H1 + _r)

COL_MASKS = []
for _c in range(WIDTH):
    _m = 0
    for _r in range(HEIGHT):
        _m |= 1 << (_c * H1 + _r)
    COL_MASKS.append(_m)

TOP_MASKS = [1 << (_c * H1 + HEIGHT - 1) for _c in range(WIDTH)]
BOTTOM_MASKS_COL = [1 << (_c * H1) for _c in range(WIDTH)]

COL_ORDER = []
for _i in range(WIDTH):
    COL_ORDER.append(WIDTH // 2 + (1 - 2 * (_i % 2)) * (_i + 1) // 2)

# Row bitmasks for each row (used for heuristic)
ROW_MASKS = []
for _r in range(HEIGHT):
    _m = 0
    for _c in range(WIDTH):
        _m |= 1 << (_c * H1 + _r)
    ROW_MASKS.append(_m)


def popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def compute_winning_positions(position, mask):
    # Vertical
    r = (position << 1) & (position << 2) & (position << 3)
    # Horizontal
    p = (position << H1) & (position << (2 * H1))
    r |= p & (position << (3 * H1))
    r |= p & (position >> H1)
    p = (position >> H1) & (position >> (2 * H1))
    r |= p & (position << H1)
    r |= p & (position >> (3 * H1))
    # Diagonal /
    d1 = H1 - 1
    p = (position << d1) & (position << (2 * d1))
    r |= p & (position << (3 * d1))
    r |= p & (position >> d1)
    p = (position >> d1) & (position >> (2 * d1))
    r |= p & (position << d1)
    r |= p & (position >> (3 * d1))
    # Diagonal \
    d2 = H1 + 1
    p = (position << d2) & (position << (2 * d2))
    r |= p & (position << (3 * d2))
    r |= p & (position >> d2)
    p = (position >> d2) & (position >> (2 * d2))
    r |= p & (position << d2)
    r |= p & (position >> (3 * d2))

    return r & (BOARD_MASK ^ mask)


class Position:
    __slots__ = ("current_position", "mask", "neutral", "moves")

    def __init__(self):
        self.current_position = 0
        self.mask = 0
        self.neutral = 0
        self.moves = 0

    def copy(self):
        p = Position()
        p.current_position = self.current_position
        p.mask = self.mask
        p.neutral = self.neutral
        p.moves = self.moves
        return p

    def key(self):
        return self.current_position + self.mask

    def can_play(self, col):
        return (self.mask & TOP_MASKS[col]) == 0

    def play_col(self, col):
        move = (self.mask + BOTTOM_MASKS_COL[col]) & COL_MASKS[col]
        self.play(move)

    def play(self, move):
        self.current_position ^= self.mask
        self.mask |= move
        self.moves += 1

    def is_winning_move(self, col):
        return (self.winning_position() & self.possible() & COL_MASKS[col]) != 0

    def can_win_next(self):
        return (self.winning_position() & self.possible()) != 0

    def winning_position(self):
        return compute_winning_positions(self.current_position, self.mask)

    def opponent_winning_position(self):
        return compute_winning_positions(self.current_position ^ self.mask, self.mask)

    def possible(self):
        return (self.mask + BOTTOM_MASK) & BOARD_MASK

    def possible_non_losing_moves(self):
        possible_mask = self.possible()
        opponent_win = self.opponent_winning_position()
        forced_moves = possible_mask & opponent_win

        if forced_moves:
            if forced_moves & (forced_moves - 1):
                return 0
            possible_mask = forced_moves

        return possible_mask & ~(opponent_win >> 1)

    def move_score(self, move):
        return popcount(compute_winning_positions(
            self.current_position | move, self.mask
        ))

    def nb_moves(self):
        return self.moves

    def drop_row(self, col):
        """Return the bit position where a piece would land in this column."""
        return (self.mask + BOTTOM_MASKS_COL[col]) & COL_MASKS[col]

    @staticmethod
    def from_observation(obs_array):
        pos = Position()
        my_bitmask = 0
        opp_bitmask = 0
        neutral_bitmask = 0
        total_pieces = 0

        for col in range(WIDTH):
            for row in range(HEIGHT):
                obs_row = HEIGHT - 1 - row
                bit = 1 << (col * H1 + row)
                if obs_array[obs_row, col, 0] == 1:
                    my_bitmask |= bit
                    total_pieces += 1
                elif obs_array[obs_row, col, 1] == 1:
                    opp_bitmask |= bit
                    total_pieces += 1
                elif obs_array[obs_row, col, 2] == 1:
                    neutral_bitmask |= bit

        pos.current_position = my_bitmask
        pos.mask = my_bitmask | opp_bitmask | neutral_bitmask
        pos.neutral = neutral_bitmask
        pos.moves = total_pieces
        return pos


# ═══════════════════════════════════════════════════════════════════════════════
#  Transposition Table
# ═══════════════════════════════════════════════════════════════════════════════

EXACT = 0
LOWER_BOUND = 1
UPPER_BOUND = 2


class TranspositionTable:
    __slots__ = ("table",)

    def __init__(self):
        self.table = {}

    def put(self, key, flag, value, depth):
        entry = self.table.get(key)
        if entry is None or depth >= entry[2]:
            self.table[key] = (flag, value, depth)

    def get(self, key):
        return self.table.get(key)

    def clear(self):
        self.table.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  Solver
# ═══════════════════════════════════════════════════════════════════════════════

MAX_SCORE = (TOTAL_CELLS + 1) // 2
MIN_SCORE = -TOTAL_CELLS // 2


class Solver:
    def __init__(self):
        self.tt = TranspositionTable()
        self.node_count = 0
        self.deadline = 0.0
        self.timed_out = False

    def _negamax(self, pos, alpha, beta, depth):
        self.node_count += 1

        if self.node_count & 2047 == 0:
            if time.time() > self.deadline:
                self.timed_out = True
                raise TimeoutError()

        if pos.nb_moves() >= TOTAL_CELLS - 1:
            return 0

        if pos.can_win_next():
            return (TOTAL_CELLS + 1 - pos.nb_moves()) // 2

        if depth <= 0:
            return self._evaluate(pos)

        possible = pos.possible_non_losing_moves()
        if possible == 0:
            return -(TOTAL_CELLS - pos.nb_moves()) // 2

        # Tighten bounds
        max_possible = (TOTAL_CELLS - 1 - pos.nb_moves()) // 2
        if beta > max_possible:
            beta = max_possible
            if alpha >= beta:
                return beta

        min_possible = -(TOTAL_CELLS - 2 - pos.nb_moves()) // 2
        if alpha < min_possible:
            alpha = min_possible
            if alpha >= beta:
                return alpha

        # TT lookup
        key = pos.key()
        entry = self.tt.get(key)
        if entry is not None:
            flag, val, entry_depth = entry
            if entry_depth >= depth:
                if flag == EXACT:
                    return val
                elif flag == LOWER_BOUND:
                    if val > alpha:
                        alpha = val
                        if alpha >= beta:
                            return alpha
                elif flag == UPPER_BOUND:
                    if val < beta:
                        beta = val
                        if alpha >= beta:
                            return beta

        # Generate and sort moves
        moves = []
        for col in COL_ORDER:
            move = possible & COL_MASKS[col]
            if move:
                score = pos.move_score(move)
                moves.append((score, move))
        moves.sort(key=lambda x: x[0], reverse=True)

        orig_alpha = alpha
        for _, move in moves:
            child = pos.copy()
            child.play(move)
            score = -self._negamax(child, -beta, -alpha, depth - 1)

            if score >= beta:
                self.tt.put(key, LOWER_BOUND, score, depth)
                return score
            if score > alpha:
                alpha = score

        if alpha > orig_alpha:
            self.tt.put(key, EXACT, alpha, depth)
        else:
            self.tt.put(key, UPPER_BOUND, alpha, depth)
        return alpha

    def _evaluate(self, pos):
        """Heuristic: count threats and center control."""
        my_wp = compute_winning_positions(pos.current_position, pos.mask)
        opp_pos = pos.current_position ^ pos.mask
        opp_wp = compute_winning_positions(opp_pos, pos.mask)

        my_threats = popcount(my_wp)
        opp_threats = popcount(opp_wp)

        # Center control (columns are worth more toward center)
        my_center = 0
        opp_center = 0
        center_weights = [0, 1, 2, 3, 2, 1, 0]
        for c in range(WIDTH):
            my_in_col = popcount(pos.current_position & COL_MASKS[c])
            opp_in_col = popcount(opp_pos & COL_MASKS[c])
            my_center += my_in_col * center_weights[c]
            opp_center += opp_in_col * center_weights[c]

        # Playable threats (threats on cells we can actually reach next)
        possible = pos.possible()
        my_playable_threats = popcount(my_wp & possible)
        opp_playable_threats = popcount(opp_wp & possible)

        return (my_threats * 3 + my_playable_threats * 5 + my_center) - \
               (opp_threats * 3 + opp_playable_threats * 5 + opp_center)

    def _find_double_threat(self, pos, col):
        """
        Check if playing `col` creates a double-threat (two winning spots that
        the opponent can't both block). This is the key tactical pattern.
        """
        if not pos.can_play(col):
            return False

        child = pos.copy()
        child.play_col(col)

        # After our move, it's opponent's turn. Check if from opponent's
        # perspective, they have non-losing moves. If possibleNonLosing == 0 after
        # our move, we've created an unblockable situation.
        # But more precisely: check if WE have 2+ winning spots after playing.
        # Our winning positions are computed from the opponent's perspective as
        # opponent_winning_position (since we just switched players).
        our_wins = child.opponent_winning_position()  # "opponent" of child = us
        playable = child.possible()
        playable_wins = our_wins & playable

        return popcount(playable_wins) >= 2

    def analyze(self, pos, time_limit=4.0):
        """Iterative deepening analysis."""
        start_time = time.time()
        self.deadline = start_time + time_limit
        self.tt.clear()
        self.timed_out = False

        legal_cols = [c for c in range(WIDTH) if pos.can_play(c)]
        if not legal_cols:
            return []

        # === Instant tactical checks ===

        # 1. Win immediately
        for col in legal_cols:
            if pos.is_winning_move(col):
                return [(col, MAX_SCORE)]

        # 2. Block opponent's winning move
        opp_wins = pos.opponent_winning_position()
        possible = pos.possible()
        forced = possible & opp_wins
        if forced:
            # Find forced column(s)
            forced_cols = [c for c in legal_cols if forced & COL_MASKS[c]]
            if len(forced_cols) >= 2:
                # Opponent has 2+ winning threats — we lose, but block one
                return [(forced_cols[0], -MAX_SCORE)]
            if forced_cols:
                return [(forced_cols[0], 0)]

        # 3. Check for double-threat (trap) moves
        trap_moves = []
        for col in COL_ORDER:
            if col in legal_cols and self._find_double_threat(pos, col):
                trap_moves.append(col)
        if trap_moves:
            # Pick the trap move — this is almost always winning
            return [(trap_moves[0], MAX_SCORE - 5)]

        # 4. Filter non-losing moves
        non_losing = pos.possible_non_losing_moves()
        if non_losing == 0:
            return [(col, -MAX_SCORE) for col in COL_ORDER if col in legal_cols]

        safe_cols = [c for c in COL_ORDER if c in legal_cols and (non_losing & COL_MASKS[c])]
        if not safe_cols:
            safe_cols = [c for c in COL_ORDER if c in legal_cols]

        # === Iterative deepening on safe moves ===
        best_results = [(col, 0) for col in safe_cols]

        for depth in range(1, TOTAL_CELLS + 1):
            if time.time() > self.deadline:
                break

            try:
                self.node_count = 0
                self.timed_out = False
                scores = {}

                search_order = [r[0] for r in best_results]

                for col in search_order:
                    if time.time() > self.deadline:
                        raise TimeoutError()
                    if not pos.can_play(col):
                        continue
                    if pos.is_winning_move(col):
                        scores[col] = (TOTAL_CELLS + 1 - pos.nb_moves()) // 2
                        continue

                    child = pos.copy()
                    child.play_col(col)
                    score = -self._negamax(child, -MAX_SCORE, MAX_SCORE, depth - 1)
                    scores[col] = score

                if not self.timed_out and len(scores) == len(search_order):
                    results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    best_results = results
                    if results and abs(results[0][1]) > TOTAL_CELLS // 2 - 5:
                        break

            except TimeoutError:
                break

        return best_results


# ═══════════════════════════════════════════════════════════════════════════════
#  Bot
# ═══════════════════════════════════════════════════════════════════════════════


class Bot:
    def __init__(self):
        self.solver = Solver()

    def act(self, observation):
        board_obs = observation["observation"]
        action_mask = observation["action_mask"]

        legal = [c for c in range(7) if action_mask[c] == 1]
        if not legal:
            return 0
        if len(legal) == 1:
            return legal[0]

        pos = Position.from_observation(board_obs)

        try:
            results = self.solver.analyze(pos, time_limit=4.0)
            if results:
                best_col = results[0][0]
                if action_mask[best_col] == 1:
                    return best_col
        except Exception:
            pass

        for col in [3, 4, 2, 5, 1, 6, 0]:
            if col in legal:
                return col
        return legal[0]
