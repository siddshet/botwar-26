# BotWars 26 — Connect 4 Plus

> **Register:** [Unstop](https://unstop.com/competitions/bot-wars-nitk-surathkal-1648256) | **Discord:** [Join Server](https://discord.gg/JSN3C9zT)

Train a bot to play **Connect 4 Plus** and compete in the BotWars tournament!

---

## Important Rules

- `training.py` and `myBot` are **just samples** — feel free to use any architecture, algorithm, or approach you like.
- Your submitted model must **run on CPU only** (no GPU/CUDA). Submissions that require a GPU will be disqualified.
- At least **1 NITK student** must be on your team to be eligible for prizes.
- Need an extra library? Ask in **Discord** before the deadline so we can add it to the tournament environment.
- All updates and announcements will be posted on **Discord** — make sure you've joined.
- For any doubts or questions, ask in **Discord**.

---

## The Game

Connect 4 Plus is classic Connect Four with a twist: a **neutral coin** is randomly placed on the bottom row at the start. Neither player owns it, but pieces stack on top of it. The neutral coin **does not** count toward anyone's four-in-a-row.

- **Board**: 6 rows × 7 columns
- **Players**: 2 (taking turns)
- **Win**: Connect 4 of your pieces in a line (horizontal / vertical / diagonal)
- **Draw**: Board full, no winner
- **Illegal move**: Instant loss (`-1` reward)

## Observation & Action

| Key | Shape | Description |
|-----|-------|-------------|
| `observation` | `(6, 7, 3)` | Binary planes: `[your pieces, opponent pieces, neutral coin]` |
| `action_mask` | `(7,)` | `1` = legal column, `0` = full |

**Action**: integer `0–6` (column to drop your piece in).

## Quick Start

```bash
# Install PyTorch first (not in requirements.txt so you can pick CPU or CUDA):
pip install torch                                                  # CPU-only
# pip install torch --index-url https://download.pytorch.org/whl/cu126  # CUDA 12.6
# See https://pytorch.org/get-started/locally/ for all options.

pip install -r requirements.txt
python training.py              # sample DQN self-play training
python main.py ruleBot myBot    # run a match (each bot gets first move once)
```

Weights are saved to `weights/model.safetensors` every 500 episodes. Match GIFs appear in `recordings/`.

## Submission Format

Submit a **folder** with at least a `model.py`:

```
yourBotName/
    model.py
    model.safetensors   # optional weights
```

### `model.py` Requirements

```python
class YourBot:
    def __init__(self):
        # Load weights here (use paths relative to __file__)
        pass

    def act(self, observation):
        # observation has keys: 'observation' (6,7,3) and 'action_mask' (7,)
        # Return an int 0-6 (must be a legal column)
        return action
```

- Class name doesn't matter — the loader picks the first class with an `act` method.
- **Always respect `action_mask`** — illegal moves forfeit the game.
- Load weights relative to `__file__` so paths work on the tournament server.

## Project Structure

```
BotWars26/
    connect4plus/          # Game environment (do not modify)
    sample_submission/
        myBot/             # Sample DQN bot
        ruleBot/           # Sample rule-based bot
    main.py                # Run matches between bots
    training.py            # Sample training script
    requirements.txt
```

## Tips

- **Always mask illegal actions** — set logits/Q-values of illegal moves to `-inf`.
- **Self-play** is a great starting point, but you can train against the rule bot or any other opponent.
- You're not limited to DQN — try PPO, AlphaZero, MCTS, or anything else.
- Test locally: `python main.py ruleBot yourBot`
- Check GIF recordings in `recordings/` to debug your bot's play.

Good luck! 🎯
