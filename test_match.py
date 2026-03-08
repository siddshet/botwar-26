"""Quick 4-game test: MyBot vs RuleBot."""
import sys
sys.path.insert(0, ".")
from connect4plus.game import env as make_env
sys.path.insert(0, "sample_submission")
from ruleBot.model import RuleBot
from my_submission.model import Bot

rule = RuleBot()
my = Bot()

wins = {"MyBot": 0, "RuleBot": 0, "Draw": 0}

for game_num in range(1, 5):
    ev = make_env()
    ev.reset()
    if game_num % 2 == 1:
        bots = {"player_0": my, "player_1": rule}
        first = "MyBot"
    else:
        bots = {"player_0": rule, "player_1": my}
        first = "RuleBot"

    for agent in ev.agent_iter():
        obs, reward, term, trunc, info = ev.last()
        if term or trunc:
            break
        action = bots[agent].act(obs)
        ev.step(action)

    r = ev.rewards
    if r["player_0"] > r["player_1"]:
        winner = first
    elif r["player_1"] > r["player_0"]:
        winner = "MyBot" if first == "RuleBot" else "RuleBot"
    else:
        winner = "Draw"

    wins[winner] = wins.get(winner, 0) + 1
    print("Game %d: %s wins (first mover: %s)" % (game_num, winner, first))
    ev.close()

print("\nFinal: MyBot %d - RuleBot %d - Draws %d" % (wins["MyBot"], wins["RuleBot"], wins["Draw"]))
