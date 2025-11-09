import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult
from strategies import HybridStrategy, VoronoiStrategy, MinimaxStrategy, EndgamePathStrategy, BoostStrategy
from improvements import EliteStrategy, ImprovedVoronoiStrategy

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"

# Initialize strategy - can be changed via environment variable
# Options: "elite", "hybrid", "endgame", "improvedvoronoi", "voronoi", "minimax"
STRATEGY_NAME = os.environ.get("STRATEGY", "elite").lower()
BOOST_STRATEGY = os.environ.get("BOOST_STRATEGY", "conservative").lower()  # Conservative performed better

# Map boost strategy string to enum
boost_map = {
    "conservative": BoostStrategy.CONSERVATIVE,
    "aggressive": BoostStrategy.AGGRESSIVE,
    "adaptive": BoostStrategy.ADAPTIVE,
    "never": BoostStrategy.NEVER,
    "always": BoostStrategy.ALWAYS,
}

boost_enum = boost_map.get(BOOST_STRATEGY, BoostStrategy.CONSERVATIVE)  # Default to conservative

# Initialize the strategy
if STRATEGY_NAME == "voronoi":
    strategy = VoronoiStrategy(boost_enum)
elif STRATEGY_NAME == "improvedvoronoi":
    strategy = ImprovedVoronoiStrategy(boost_enum)
elif STRATEGY_NAME == "minimax":
    strategy = MinimaxStrategy(boost_enum, depth=1)
elif STRATEGY_NAME == "endgame":
    strategy = EndgamePathStrategy(boost_enum)
elif STRATEGY_NAME == "elite":
    strategy = EliteStrategy(boost_enum)  # Best performing strategy
else:
    strategy = EliteStrategy(boost_enum)  # Default to elite (best performer)

print(f"Initialized strategy: {strategy.name} with boost strategy: {boost_enum.value}")


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"])
            # Update direction from trail
            if len(GLOBAL_GAME.agent1.trail) >= 2:
                head = GLOBAL_GAME.agent1.trail[-1]
                prev = GLOBAL_GAME.agent1.trail[-2]
                dx = head[0] - prev[0]
                dy = head[1] - prev[1]
                # Normalize for torus
                if abs(dx) > GLOBAL_GAME.board.width // 2:
                    dx = -1 if dx > 0 else 1
                if abs(dy) > GLOBAL_GAME.board.height // 2:
                    dy = -1 if dy > 0 else 1
                # Update direction
                if dx == 1:
                    GLOBAL_GAME.agent1.direction = Direction.RIGHT
                elif dx == -1:
                    GLOBAL_GAME.agent1.direction = Direction.LEFT
                elif dy == 1:
                    GLOBAL_GAME.agent1.direction = Direction.DOWN
                elif dy == -1:
                    GLOBAL_GAME.agent1.direction = Direction.UP
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"])
            # Update direction from trail
            if len(GLOBAL_GAME.agent2.trail) >= 2:
                head = GLOBAL_GAME.agent2.trail[-1]
                prev = GLOBAL_GAME.agent2.trail[-2]
                dx = head[0] - prev[0]
                dy = head[1] - prev[1]
                # Normalize for torus
                if abs(dx) > GLOBAL_GAME.board.width // 2:
                    dx = -1 if dx > 0 else 1
                if abs(dy) > GLOBAL_GAME.board.height // 2:
                    dy = -1 if dy > 0 else 1
                # Update direction
                if dx == 1:
                    GLOBAL_GAME.agent2.direction = Direction.RIGHT
                elif dx == -1:
                    GLOBAL_GAME.agent2.direction = Direction.LEFT
                elif dy == 1:
                    GLOBAL_GAME.agent2.direction = Direction.DOWN
                elif dy == -1:
                    GLOBAL_GAME.agent2.direction = Direction.UP 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
   
    # -----------------your code here-------------------
    # Use the selected strategy to determine the best move
    try:
        move = strategy.get_move(GLOBAL_GAME, player_number)
    except Exception as e:
        # Fallback to safe move if strategy fails
        print(f"Strategy error: {e}")
        # Get valid directions
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        valid_dirs = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        # Remove opposite direction
        if len(my_agent.trail) >= 2:
            head = my_agent.trail[-1]
            prev = my_agent.trail[-2]
            dx = head[0] - prev[0]
            dy = head[1] - prev[1]
            
            # Normalize for torus
            if abs(dx) > GLOBAL_GAME.board.width // 2:
                dx = -1 if dx > 0 else 1
            if abs(dy) > GLOBAL_GAME.board.height // 2:
                dy = -1 if dy > 0 else 1
            
            opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
            if dx == 1:
                current_dir = "RIGHT"
            elif dx == -1:
                current_dir = "LEFT"
            elif dy == 1:
                current_dir = "DOWN"
            elif dy == -1:
                current_dir = "UP"
            else:
                current_dir = None
            
            if current_dir and current_dir in opposite:
                try:
                    valid_dirs.remove(opposite[current_dir])
                except ValueError:
                    pass
        
        # Pick first safe direction
        move = valid_dirs[0] if valid_dirs else "UP"
    # -----------------end code here--------------------

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
