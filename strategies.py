"""
Multiple strategy implementations for Case Closed game.
Includes Voronoi, Minimax, Endgame Optimization, and Hybrid approaches.
"""

from collections import deque
from typing import Optional, Tuple, List, Dict
from enum import Enum

from case_closed_game import Direction, Game, GameBoard, EMPTY, AGENT


class BoostStrategy(Enum):
    """Different boost usage strategies"""
    CONSERVATIVE = "conservative"  # Save for endgame
    AGGRESSIVE = "aggressive"  # Use early
    ADAPTIVE = "adaptive"  # Use based on game state
    NEVER = "never"  # Never use
    ALWAYS = "always"  # Use whenever available


class BaseStrategy:
    """Base class for all strategies"""
    
    def __init__(self, boost_strategy: BoostStrategy = BoostStrategy.ADAPTIVE):
        self.boost_strategy = boost_strategy
        self.name = "Base"
    
    def get_move(self, game: Game, player_number: int) -> str:
        """Returns move in format "DIRECTION" or "DIRECTION:BOOST" """
        my_agent = game.agent1 if player_number == 1 else game.agent2
        other_agent = game.agent2 if player_number == 1 else game.agent1
        
        if not my_agent.alive:
            return "UP"  # Default if dead
        
        direction = self._choose_direction(game, player_number)
        use_boost = self._should_boost(game, player_number, direction)
        
        if use_boost:
            return f"{direction}:BOOST"
        return direction
    
    def _choose_direction(self, game: Game, player_number: int) -> str:
        """Override in subclasses"""
        return "RIGHT"
    
    def _should_boost(self, game: Game, player_number: int, direction: str) -> bool:
        """Decide whether to use boost based on strategy"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        turn_count = game.turns
        
        if my_agent.boosts_remaining <= 0:
            return False
        
        if self.boost_strategy == BoostStrategy.NEVER:
            return False
        elif self.boost_strategy == BoostStrategy.ALWAYS:
            return True
        elif self.boost_strategy == BoostStrategy.CONSERVATIVE:
            # Save for last 50 turns
            return turn_count >= 150
        elif self.boost_strategy == BoostStrategy.AGGRESSIVE:
            # Use in first 100 turns
            return turn_count < 100
        elif self.boost_strategy == BoostStrategy.ADAPTIVE:
            # Use when in danger or when it gives significant advantage
            return self._adaptive_boost(game, player_number, direction)
        
        return False
    
    def _adaptive_boost(self, game: Game, player_number: int, direction: str) -> bool:
        """Adaptive boost logic - override in subclasses if needed"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        turn_count = game.turns
        
        # Use boost if we're behind in length
        other_agent = game.agent2 if player_number == 1 else game.agent1
        if my_agent.length < other_agent.length and turn_count > 50:
            return True
        
        # Use boost in endgame if we have them
        if turn_count >= 150:
            return True
        
        return False
    
    def _get_valid_directions(self, game: Game, player_number: int) -> List[str]:
        """Get list of valid directions (not opposite to current direction)"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        
        if len(my_agent.trail) < 2:
            return ["UP", "DOWN", "LEFT", "RIGHT"]
        
        # Get current direction
        head = my_agent.trail[-1]
        prev = my_agent.trail[-2]
        dx = head[0] - prev[0]
        dy = head[1] - prev[1]
        
        # Normalize for torus
        if abs(dx) > game.board.width // 2:
            dx = -1 if dx > 0 else 1
        if abs(dy) > game.board.height // 2:
            dy = -1 if dy > 0 else 1
        
        current_dir = None
        if dx == 1:
            current_dir = "RIGHT"
        elif dx == -1:
            current_dir = "LEFT"
        elif dy == 1:
            current_dir = "DOWN"
        elif dy == -1:
            current_dir = "UP"
        
        all_dirs = ["UP", "DOWN", "LEFT", "RIGHT"]
        opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        
        if current_dir and current_dir in opposite:
            all_dirs.remove(opposite[current_dir])
        
        return all_dirs
    
    def _is_safe_move(self, game: Game, player_number: int, direction: str) -> bool:
        """Check if a move is safe (won't hit a trail immediately)"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        other_agent = game.agent2 if player_number == 1 else game.agent1
        
        if len(my_agent.trail) == 0:
            return True
        
        head = my_agent.trail[-1]
        dx, dy = Direction[direction].value
        new_pos = ((head[0] + dx) % game.board.width, (head[1] + dy) % game.board.height)
        
        # Check if position is occupied
        if game.board.get_cell_state(new_pos) == AGENT:
            # Check if it's our own trail (excluding head)
            if new_pos in list(my_agent.trail)[:-1]:
                return False
            # Check if it's opponent's trail
            if other_agent.alive and new_pos in other_agent.trail:
                return False
        
        return True


class VoronoiStrategy(BaseStrategy):
    """Voronoi heuristic - maximize territory control"""
    
    def __init__(self, boost_strategy: BoostStrategy = BoostStrategy.ADAPTIVE):
        super().__init__(boost_strategy)
        self.name = "Voronoi"
    
    def _choose_direction(self, game: Game, player_number: int) -> str:
        my_agent = game.agent1 if player_number == 1 else game.agent2
        other_agent = game.agent2 if player_number == 1 else game.agent1
        
        valid_dirs = self._get_valid_directions(game, player_number)
        safe_dirs = [d for d in valid_dirs if self._is_safe_move(game, player_number, d)]
        
        if not safe_dirs:
            return valid_dirs[0] if valid_dirs else "UP"
        
        # Calculate Voronoi scores for each direction
        best_dir = None
        best_score = float('-inf')
        
        for direction in safe_dirs:
            score = self._evaluate_voronoi(game, player_number, direction)
            if score > best_score:
                best_score = score
                best_dir = direction
        
        return best_dir if best_dir else safe_dirs[0]
    
    def _evaluate_voronoi(self, game: Game, player_number: int, direction: str) -> float:
        """Evaluate move using Voronoi diagram - assign tiles to nearest player"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        other_agent = game.agent2 if player_number == 1 else game.agent1
        
        # Simulate move
        head = my_agent.trail[-1]
        dx, dy = Direction[direction].value
        new_pos = ((head[0] + dx) % game.board.width, 
                   (head[1] + dy) % game.board.height)
        
        # BFS from both players simultaneously - assign tiles to whoever reaches first
        my_territory = 0
        other_territory = 0
        
        # Track which player owns each tile
        tile_owner = {}
        visited = set()
        
        # Mark all trails as blocked
        for pos in my_agent.trail:
            visited.add(pos)
        for pos in other_agent.trail:
            visited.add(pos)
        
        # Start BFS from both positions
        queue = deque([(new_pos, 0, True), (other_agent.trail[-1], 0, False)])
        visited.add(new_pos)
        visited.add(other_agent.trail[-1])
        
        max_depth = 5  # Reduced from min(width, height) // 2 for performance
        
        while queue:
            pos, depth, is_mine = queue.popleft()
            
            if depth > max_depth:
                continue
            
            # Assign tile to this player if not already assigned
            if pos not in tile_owner:
                tile_owner[pos] = is_mine
                if is_mine:
                    my_territory += 1
                else:
                    other_territory += 1
            
            # Check neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = (pos[0] + dx) % game.board.width
                new_y = (pos[1] + dy) % game.board.height
                neighbor = (new_x, new_y)
                
                if neighbor not in visited and game.board.get_cell_state(neighbor) == EMPTY:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1, is_mine))
        
        # Score is difference in territory
        return my_territory - other_territory


class MinimaxStrategy(BaseStrategy):
    """Minimax with alpha-beta pruning"""
    
    def __init__(self, boost_strategy: BoostStrategy = BoostStrategy.ADAPTIVE, depth: int = 1):
        super().__init__(boost_strategy)
        self.name = "Minimax"
        self.depth = depth  # Reduced default depth for performance
    
    def _choose_direction(self, game: Game, player_number: int) -> str:
        valid_dirs = self._get_valid_directions(game, player_number)
        safe_dirs = [d for d in valid_dirs if self._is_safe_move(game, player_number, d)]
        
        if not safe_dirs:
            return valid_dirs[0] if valid_dirs else "UP"
        
        best_dir = None
        best_score = float('-inf')
        
        for direction in safe_dirs:
            score = self._minimax(game, player_number, direction, self.depth, float('-inf'), float('inf'), True)
            if score > best_score:
                best_score = score
                best_dir = direction
        
        return best_dir if best_dir else safe_dirs[0]
    
    def _minimax(self, game: Game, player_number: int, direction: str, depth: int, 
                 alpha: float, beta: float, maximizing: bool) -> float:
        """Minimax with alpha-beta pruning"""
        if depth == 0:
            return self._evaluate_position(game, player_number)
        
        # Create a copy of the game for simulation
        game_copy = self._copy_game_state(game)
        my_agent = game_copy.agent1 if player_number == 1 else game_copy.agent2
        other_agent = game_copy.agent2 if player_number == 1 else game_copy.agent1
        
        if maximizing:
            # Our turn
            if not my_agent.alive:
                return float('-inf')
            
            valid_dirs = self._get_valid_directions(game_copy, player_number)
            safe_dirs = [d for d in valid_dirs if self._is_safe_move(game_copy, player_number, d)]
            
            if not safe_dirs:
                return float('-inf')
            
            max_eval = float('-inf')
            for dir in safe_dirs:
                # Simulate move
                game_sim = self._copy_game_state(game_copy)
                agent_sim = game_sim.agent1 if player_number == 1 else game_sim.agent2
                other_sim = game_sim.agent2 if player_number == 1 else game_sim.agent1
                
                # Simple move simulation
                head = agent_sim.trail[-1]
                dx, dy = Direction[dir].value
                new_pos = ((head[0] + dx) % game_sim.board.width, 
                          (head[1] + dy) % game_sim.board.height)
                
                if game_sim.board.get_cell_state(new_pos) == EMPTY:
                    agent_sim.trail.append(new_pos)
                    agent_sim.length += 1
                    game_sim.board.set_cell_state(new_pos, AGENT)
                    game_sim.turns += 1
                
                eval_score = self._minimax(game_sim, player_number, dir, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return max_eval
        else:
            # Opponent's turn
            if not other_agent.alive:
                return float('inf')
            
            other_player = 2 if player_number == 1 else 1
            valid_dirs = self._get_valid_directions(game_copy, other_player)
            safe_dirs = [d for d in valid_dirs if self._is_safe_move(game_copy, other_player, d)]
            
            if not safe_dirs:
                return float('inf')
            
            min_eval = float('inf')
            for dir in safe_dirs:
                game_sim = self._copy_game_state(game_copy)
                agent_sim = game_sim.agent2 if player_number == 1 else game_sim.agent1
                other_sim = game_sim.agent1 if player_number == 1 else game_sim.agent2
                
                head = agent_sim.trail[-1]
                dx, dy = Direction[dir].value
                new_pos = ((head[0] + dx) % game_sim.board.width, 
                          (head[1] + dy) % game_sim.board.height)
                
                if game_sim.board.get_cell_state(new_pos) == EMPTY:
                    agent_sim.trail.append(new_pos)
                    agent_sim.length += 1
                    game_sim.board.set_cell_state(new_pos, AGENT)
                    game_sim.turns += 1
                
                eval_score = self._minimax(game_sim, player_number, dir, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval
    
    def _evaluate_position(self, game: Game, player_number: int) -> float:
        """Evaluate current position"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        other_agent = game.agent2 if player_number == 1 else game.agent1
        
        if not my_agent.alive:
            return float('-inf')
        if not other_agent.alive:
            return float('inf')
        
        # Score based on length difference and space control
        length_diff = my_agent.length - other_agent.length
        
        # Add space control component
        my_space = self._count_reachable_space(game, my_agent.trail[-1])
        other_space = self._count_reachable_space(game, other_agent.trail[-1])
        space_diff = my_space - other_space
        
        return length_diff * 10 + space_diff * 0.1
    
    def _count_reachable_space(self, game: Game, start_pos: tuple) -> int:
        """Count reachable empty spaces using BFS"""
        visited = set()
        queue = deque([start_pos])
        visited.add(start_pos)
        count = 0
        
        while queue:
            pos = queue.popleft()
            if game.board.get_cell_state(pos) == EMPTY:
                count += 1
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = (pos[0] + dx) % game.board.width
                new_y = (pos[1] + dy) % game.board.height
                neighbor = (new_x, new_y)
                
                if neighbor not in visited and game.board.get_cell_state(neighbor) == EMPTY:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return count
    
    def _copy_game_state(self, game: Game) -> Game:
        """Create a deep copy of game state for simulation"""
        new_game = Game()
        new_game.board = GameBoard(game.board.height, game.board.width)
        new_game.board.grid = [row[:] for row in game.board.grid]
        new_game.turns = game.turns
        
        # Copy agent1
        new_game.agent1.trail = deque(list(game.agent1.trail))
        new_game.agent1.direction = game.agent1.direction
        new_game.agent1.alive = game.agent1.alive
        new_game.agent1.length = game.agent1.length
        new_game.agent1.boosts_remaining = game.agent1.boosts_remaining
        
        # Copy agent2
        new_game.agent2.trail = deque(list(game.agent2.trail))
        new_game.agent2.direction = game.agent2.direction
        new_game.agent2.alive = game.agent2.alive
        new_game.agent2.length = game.agent2.length
        new_game.agent2.boosts_remaining = game.agent2.boosts_remaining
        
        return new_game


class EndgamePathStrategy(BaseStrategy):
    """Endgame strategy - maximize path length in enclosed area"""
    
    def __init__(self, boost_strategy: BoostStrategy = BoostStrategy.CONSERVATIVE):
        super().__init__(boost_strategy)
        self.name = "EndgamePath"
    
    def _choose_direction(self, game: Game, player_number: int) -> str:
        my_agent = game.agent1 if player_number == 1 else game.agent2
        
        # Check if we're in endgame (high turn count or low available space)
        is_endgame = game.turns >= 150 or self._is_enclosed(game, player_number)
        
        if is_endgame:
            return self._maximize_path_length(game, player_number)
        else:
            # Use Voronoi in early/mid game
            voronoi = VoronoiStrategy(self.boost_strategy)
            return voronoi._choose_direction(game, player_number)
    
    def _is_enclosed(self, game: Game, player_number: int) -> bool:
        """Check if agent is in an enclosed area - optimized"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        
        # Quick estimate: use trail lengths as proxy for occupied cells
        # This avoids expensive full board iteration
        total_cells = game.board.width * game.board.height
        occupied = len(game.agent1.trail) + len(game.agent2.trail)
        total_empty = total_cells - occupied
        
        # Early exit: if we have lots of space, we're not enclosed
        if total_empty > total_cells * 0.5:
            return False
        
        # Count reachable space with early termination
        threshold = total_empty * 0.3
        reachable = self._count_reachable_space_fast(game, my_agent.trail[-1], threshold)
        
        return reachable < threshold
    
    def _maximize_path_length(self, game: Game, player_number: int) -> str:
        """Find direction that maximizes path length using DFS"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        valid_dirs = self._get_valid_directions(game, player_number)
        safe_dirs = [d for d in valid_dirs if self._is_safe_move(game, player_number, d)]
        
        if not safe_dirs:
            return valid_dirs[0] if valid_dirs else "UP"
        
        best_dir = None
        best_path_length = 0
        
        for direction in safe_dirs:
            head = my_agent.trail[-1]
            dx, dy = Direction[direction].value
            new_pos = ((head[0] + dx) % game.board.width, 
                       (head[1] + dy) % game.board.height)
            
            path_length = self._dfs_path_length(game, new_pos, set(my_agent.trail))
            if path_length > best_path_length:
                best_path_length = path_length
                best_dir = direction
        
        return best_dir if best_dir else safe_dirs[0]
    
    def _dfs_path_length(self, game: Game, start_pos: tuple, blocked: set) -> int:
        """Calculate maximum path length using DFS with depth limit for performance"""
        visited = set(blocked)
        max_path = 0
        max_depth = 15  # Reduced from 20 for speed
        max_nodes = 100  # Limit total nodes explored
        
        def dfs(pos, length, nodes_explored):
            nonlocal max_path
            if length > max_depth or nodes_explored > max_nodes:
                return nodes_explored
            max_path = max(max_path, length)
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = (pos[0] + dx) % game.board.width
                new_y = (pos[1] + dy) % game.board.height
                neighbor = (new_x, new_y)
                
                if neighbor not in visited and game.board.get_cell_state(neighbor) == EMPTY:
                    visited.add(neighbor)
                    nodes_explored = dfs(neighbor, length + 1, nodes_explored + 1)
                    visited.remove(neighbor)
                    if nodes_explored > max_nodes:
                        return nodes_explored
            
            return nodes_explored
        
        if start_pos not in visited and game.board.get_cell_state(start_pos) == EMPTY:
            visited.add(start_pos)
            dfs(start_pos, 1, 1)
        
        return max_path
    
    def _count_reachable_space(self, game: Game, start_pos: tuple) -> int:
        """Count reachable empty spaces"""
        return self._count_reachable_space_fast(game, start_pos, float('inf'))
    
    def _count_reachable_space_fast(self, game: Game, start_pos: tuple, threshold: float) -> int:
        """Count reachable empty spaces with early termination for performance"""
        visited = set()
        queue = deque([start_pos])
        visited.add(start_pos)
        count = 0
        max_count = int(threshold) + 10  # Add small buffer
        
        while queue and count < max_count:
            pos = queue.popleft()
            if game.board.get_cell_state(pos) == EMPTY:
                count += 1
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = (pos[0] + dx) % game.board.width
                new_y = (pos[1] + dy) % game.board.height
                neighbor = (new_x, new_y)
                
                if neighbor not in visited and game.board.get_cell_state(neighbor) == EMPTY:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return count


class HybridStrategy(BaseStrategy):
    """Hybrid strategy combining multiple approaches"""
    
    def __init__(self, boost_strategy: BoostStrategy = BoostStrategy.ADAPTIVE):
        super().__init__(boost_strategy)
        self.name = "Hybrid"
        self.voronoi = VoronoiStrategy(boost_strategy)
        self.endgame = EndgamePathStrategy(boost_strategy)
        self.minimax = MinimaxStrategy(boost_strategy, depth=2)
    
    def _choose_direction(self, game: Game, player_number: int) -> str:
        my_agent = game.agent1 if player_number == 1 else game.agent2
        other_agent = game.agent2 if player_number == 1 else game.agent1
        turn_count = game.turns
        
        # Early game: Voronoi for space control
        if turn_count < 50:
            return self.voronoi._choose_direction(game, player_number)
        
        # Mid game: Voronoi (Minimax is too slow for tournament)
        elif turn_count < 150:
            return self.voronoi._choose_direction(game, player_number)
        
        # Endgame: Path optimization
        else:
            return self.endgame._choose_direction(game, player_number)
    
    def _adaptive_boost(self, game: Game, player_number: int, direction: str) -> bool:
        """Adaptive boost for hybrid strategy"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        other_agent = game.agent2 if player_number == 1 else game.agent1
        turn_count = game.turns
        
        # Use boost if we're significantly behind
        if my_agent.length < other_agent.length - 5:
            return True
        
        # Use boost in endgame if available
        if turn_count >= 150 and my_agent.boosts_remaining > 0:
            return True
        
        # Use boost to escape danger
        if not self._is_safe_move(game, player_number, direction):
            return True
        
        return False


class WallFollowingStrategy(BaseStrategy):
    """Wall-following strategy - stick to edges"""
    
    def __init__(self, boost_strategy: BoostStrategy = BoostStrategy.ADAPTIVE):
        super().__init__(boost_strategy)
        self.name = "WallFollowing"
    
    def _choose_direction(self, game: Game, player_number: int) -> str:
        my_agent = game.agent1 if player_number == 1 else game.agent2
        valid_dirs = self._get_valid_directions(game, player_number)
        safe_dirs = [d for d in valid_dirs if self._is_safe_move(game, player_number, d)]
        
        if not safe_dirs:
            return valid_dirs[0] if valid_dirs else "UP"
        
        head = my_agent.trail[-1]
        
        # Prefer directions that keep us near walls/edges
        # Calculate distance to nearest edge
        dist_to_edge = min(
            head[0], game.board.width - head[0],
            head[1], game.board.height - head[1]
        )
        
        # If far from edge, try to move towards center
        # If near edge, follow the wall
        if dist_to_edge > 3:
            # Move towards center
            center_x, center_y = game.board.width // 2, game.board.height // 2
            best_dir = None
            min_dist = float('inf')
            
            for direction in safe_dirs:
                dx, dy = Direction[direction].value
                new_pos = ((head[0] + dx) % game.board.width, 
                          (head[1] + dy) % game.board.height)
                dist = abs(new_pos[0] - center_x) + abs(new_pos[1] - center_y)
                if dist < min_dist:
                    min_dist = dist
                    best_dir = direction
            
            return best_dir if best_dir else safe_dirs[0]
        else:
            # Follow wall - prefer turning right
            # Try to maintain current direction if safe
            if len(my_agent.trail) >= 2:
                prev = my_agent.trail[-2]
                dx = head[0] - prev[0]
                dy = head[1] - prev[1]
                
                # Normalize
                if abs(dx) > game.board.width // 2:
                    dx = -1 if dx > 0 else 1
                if abs(dy) > game.board.height // 2:
                    dy = -1 if dy > 0 else 1
                
                # Try to turn right relative to current direction
                if dx == 1:  # Moving right, try down
                    if "DOWN" in safe_dirs:
                        return "DOWN"
                elif dx == -1:  # Moving left, try up
                    if "UP" in safe_dirs:
                        return "UP"
                elif dy == 1:  # Moving down, try left
                    if "LEFT" in safe_dirs:
                        return "LEFT"
                elif dy == -1:  # Moving up, try right
                    if "RIGHT" in safe_dirs:
                        return "RIGHT"
            
            return safe_dirs[0]
