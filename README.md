# BotWars 26 — Connect 4 Plus

> **Register:** [Unstop](https://unstop.com/competitions/bot-wars-nitk-surathkal-1648256) | **Discord:** [Join Server](https://discord.gg/JSN3C9zT)

Train a bot to play **Connect 4 Plus** and compete in the BotWars tournament!

> **Full rules, scoring, and tournament format → [`rules.txt`](rules.txt)**

---

## My Submission: Bitboard Alpha-Beta Solver

This bot uses a high-performance **Bitboard Alpha-Beta Solver**. It is a purely algorithmic approach with **perfect play** potential, needing no neural network or training.

### Key Features
- **Bitboard Engine**: Board state encoded in two 64-bit integers. Win detection via 4-direction bit-shifts is ~100x faster than traditional arrays.
- **Negamax α-β Pruning**: Efficiently searches millions of nodes per second to find the optimal move.
- **Transposition Table**: Caches board evaluations to avoid redundant work.
- **Iterative Deepening**: Dynamically adjusts search depth to stay within the 5s time limit.
- **Double-Threat Detection**: Instantly identifies and plays "trap" moves that create two simultaneous winning spots.

### Performance Results
- **Vs RuleBot**: **4 - 0 Sweep** ✅ (Won as both first and second mover)
- **Time per move**: 1 - 4 seconds
- **Submission size**: ~12 KB (Single file, no weights)

---



## The Game

Connect 4 Plus has a **neutral coin** randomly placed on the bottom row at the start. It blocks cells but doesn't count toward any player's win.

- **Board**: 6 rows × 7 columns
- **Win**: Connect 4 pieces in a line
- **Illegal move**: Instant loss

## Project Structure

```
BotWars26/
    connect4plus/          # Game environment
    sample_submission/
        my_submission/     # Final Solver Bot (model.py)
        ruleBot/           # Sample rule-based bot
    main.py                # Run tournament matches
    test_match.py          # Custom 4-game test script
    requirements.txt
```

## How to Run

```bash
# Run a match vs RuleBot
python main.py ruleBot my_submission

# Check GIF recordings in recordings/ to see the solver in action!
```

Good luck! 🎯

