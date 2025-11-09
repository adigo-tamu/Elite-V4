"""
Improved strategies based on tournament results analysis.
Key findings:
- EndgamePath_conservative: 100% win rate
- Hybrid_conservative: 100% win rate  
- Voronoi: Underperforming (28-43%)
- WallFollowing: Very poor (0-14%)
- Conservative boost generally better than adaptive
"""

from collections import deque
from typing import Optional, Tuple, List

from case_closed_game import Direction, Game, GameBoard, EMPTY, AGENT
from strategies import BaseStrategy, VoronoiStrategy, EndgamePathStrategy, BoostStrategy


class ImprovedVoronoiStrategy(BaseStrategy):
    """Improved Voronoi that combines territory control with safety"""
    
    def __init__(self, boost_strategy: BoostStrategy = BoostStrategy.CONSERVATIVE):
        super().__init__(boost_strategy)
        self.name = "ImprovedVoronoi"
    
    def _choose_direction(self, game: Game, player_number: int) -> str:
        my_agent = game.agent1 if player_number == 1 else game.agent2
        other_agent = game.agent2 if player_number == 1 else game.agent1
        
        valid_dirs = self._get_valid_directions(game, player_number)
        safe_dirs = [d for d in valid_dirs if self._is_safe_move(game, player_number, d)]
        
        if not safe_dirs:
            return valid_dirs[0] if valid_dirs else "UP"
        
        # Calculate scores combining Voronoi with safety and reachability
        best_dir = None
        best_score = float('-inf')
        
        for direction in safe_dirs:
            voronoi_score = self._evaluate_voronoi(game, player_number, direction)
            safety_score = self._evaluate_safety(game, player_number, direction)
            reachability_score = self._evaluate_reachability(game, player_number, direction)
            
            # Weighted combination: prioritize safety, then territory, then reachability
            score = safety_score * 3.0 + voronoi_score * 1.0 + reachability_score * 0.5
            
            if score > best_score:
                best_score = score
                best_dir = direction
        
        return best_dir if best_dir else safe_dirs[0]
    
    def _evaluate_voronoi(self, game: Game, player_number: int, direction: str) -> float:
        """Improved Voronoi evaluation - deeper search to compete with counter-strategies"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        other_agent = game.agent2 if player_number == 1 else game.agent1
        
        head = my_agent.trail[-1]
        dx, dy = Direction[direction].value
        new_pos = ((head[0] + dx) % game.board.width, 
                   (head[1] + dy) % game.board.height)
        
        # Deeper BFS to compete with counter-strategies
        my_territory = 0
        other_territory = 0
        visited = set()
        
        # Mark trails (only recent ones for speed)
        for pos in list(my_agent.trail)[-10:]:  # Only last 10 positions
            visited.add(pos)
        for pos in list(other_agent.trail)[-10:]:
            visited.add(pos)
        
        queue = deque([(new_pos, 0, True), (other_agent.trail[-1], 0, False)])
        visited.add(new_pos)
        visited.add(other_agent.trail[-1])
        
        max_depth = 5  # Increased to match AggressiveRush (was 4)
        max_nodes = 100  # Increased to match AggressiveRush (was 75)
        
        while queue and (my_territory + other_territory) < max_nodes:
            pos, depth, is_mine = queue.popleft()
            if depth > max_depth:
                continue
            
            if is_mine:
                my_territory += 1
            else:
                other_territory += 1
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = (pos[0] + dx) % game.board.width
                new_y = (pos[1] + dy) % game.board.height
                neighbor = (new_x, new_y)
                
                if neighbor not in visited and game.board.get_cell_state(neighbor) == EMPTY:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1, is_mine))
        
        return my_territory - other_territory
    
    def _evaluate_safety(self, game: Game, player_number: int, direction: str) -> float:
        """Evaluate how safe a move is (look ahead 2-3 moves)"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        
        head = my_agent.trail[-1]
        dx, dy = Direction[direction].value
        new_pos = ((head[0] + dx) % game.board.width, 
                   (head[1] + dy) % game.board.height)
        
        # Count safe moves from new position
        safe_count = 0
        for dx2, dy2 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_pos = ((new_pos[0] + dx2) % game.board.width,
                       (new_pos[1] + dy2) % game.board.height)
            if game.board.get_cell_state(next_pos) == EMPTY:
                safe_count += 1
        
        return safe_count
    
    def _evaluate_reachability(self, game: Game, player_number: int, direction: str) -> float:
        """Evaluate how much space is reachable from this position - optimized"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        
        head = my_agent.trail[-1]
        dx, dy = Direction[direction].value
        new_pos = ((head[0] + dx) % game.board.width, 
                   (head[1] + dy) % game.board.height)
        
        # Quick BFS to count reachable space with reduced limit
        visited = set()
        queue = deque([new_pos])
        visited.add(new_pos)
        count = 0
        max_count = 20  # Reduced from 30 for speed
        
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


class EliteStrategy(BaseStrategy):
    """Elite strategy combining best aspects of EndgamePath and Hybrid"""
    
    def __init__(self, boost_strategy: BoostStrategy = BoostStrategy.CONSERVATIVE):
        super().__init__(boost_strategy)
        self.name = "Elite"
        self.endgame = EndgamePathStrategy(boost_strategy)
        self.improved_voronoi = ImprovedVoronoiStrategy(boost_strategy)
        self._cached_total_empty = None
        self._cached_turn = -1
    
    def _choose_direction(self, game: Game, player_number: int) -> str:
        my_agent = game.agent1 if player_number == 1 else game.agent2
        other_agent = game.agent2 if player_number == 1 else game.agent1
        turn_count = game.turns
        
        # Early game: Improved Voronoi for space control, but check if opponent is aggressive
        if turn_count < 30:
            # If opponent is already ahead in early game, they're being aggressive - switch earlier
            if other_agent.length > my_agent.length:
                return self.endgame._choose_direction(game, player_number)
            return self.improved_voronoi._choose_direction(game, player_number)
        
        # Mid game: Adaptive strategy selection
        elif turn_count < 130:
            # Check if we're behind - use more aggressive strategy immediately
            if my_agent.length < other_agent.length:  # Changed from -2 to 0 (immediate response)
                # When behind at all, prioritize endgame path optimization
                return self.endgame._choose_direction(game, player_number)
            
            # Check if opponent is being aggressive (ahead by 1+ in early mid-game)
            if turn_count < 80 and other_agent.length > my_agent.length:
                # Opponent is aggressive, switch to defensive mode immediately
                return self.endgame._choose_direction(game, player_number)
            
            # Check if we're in danger (enclosed or low safe moves)
            safe_moves = sum(1 for d in ["UP", "DOWN", "LEFT", "RIGHT"] 
                            if self._is_safe_move(game, player_number, d))
            
            # Use fast heuristic to check if enclosed (check every 3 turns for better responsiveness)
            if turn_count % 3 == 0:
                if self._is_enclosed_fast(game, player_number):
                    return self.endgame._choose_direction(game, player_number)
            
            # If low safe moves, switch to endgame path
            if safe_moves <= 2:
                return self.endgame._choose_direction(game, player_number)
            
            # Otherwise use improved voronoi
            return self.improved_voronoi._choose_direction(game, player_number)
        
        # Endgame: Always use path optimization
        else:
            return self.endgame._choose_direction(game, player_number)
    
    def _is_enclosed_fast(self, game: Game, player_number: int) -> bool:
        """Fast heuristic to check if enclosed - uses only quick checks"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        
        # Quick estimate: use trail lengths as proxy for occupied cells
        total_cells = game.board.width * game.board.height
        occupied = len(game.agent1.trail) + len(game.agent2.trail)
        total_empty = total_cells - occupied
        
        # Early exit: if we have lots of space, we're not enclosed
        if total_empty > total_cells * 0.5:
            return False
        
        # Quick heuristic: count immediate safe moves
        safe_moves = sum(1 for d in ["UP", "DOWN", "LEFT", "RIGHT"] 
                        if self._is_safe_move(game, player_number, d))
        if safe_moves <= 1:
            return True
        
        # Quick BFS with very limited depth
        threshold = min(total_empty * 0.35, 30)  # Reduced threshold for speed
        reachable = self._count_reachable_space_fast(game, my_agent.trail[-1], threshold)
        
        return reachable < threshold
    
    def _is_enclosed(self, game: Game, player_number: int) -> bool:
        """Check if agent is in an enclosed area - optimized with caching and fast heuristic"""
        return self._is_enclosed_fast(game, player_number)
    
    def _count_reachable_space(self, game: Game, start_pos: tuple) -> int:
        """Count reachable empty spaces - optimized with early termination"""
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
    
    def _adaptive_boost(self, game: Game, player_number: int, direction: str) -> bool:
        """Improved boost logic - more aggressive to counter aggressive opponents"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        other_agent = game.agent2 if player_number == 1 else game.agent1
        turn_count = game.turns
        
        # Always use boost in endgame (last 60 turns)
        if turn_count >= 140:
            return True
        
        # Use boost if behind (very aggressive threshold to counter aggressive opponents)
        if my_agent.length < other_agent.length - 1:  # Changed from -3 to -1 (very aggressive)
            return True
        
        # Use boost if in danger (few safe moves)
        safe_moves = sum(1 for d in ["UP", "DOWN", "LEFT", "RIGHT"] 
                        if self._is_safe_move(game, player_number, d))
        if safe_moves <= 1:
            return True
        
        # Use boost in early/mid-game if opponent is ahead or tied (very aggressive)
        if turn_count < 100:
            # If opponent is ahead or tied, use boost to gain/regain advantage
            if my_agent.length <= other_agent.length:
                if self._is_safe_boost_move(game, player_number, direction):
                    return True
            # In very early game (first 40 turns), use boost even if slightly ahead to maintain advantage
            elif turn_count < 40 and my_agent.length <= other_agent.length + 1:
                if self._is_safe_boost_move(game, player_number, direction):
                    return True
        
        return False
    
    def _is_safe_boost_move(self, game: Game, player_number: int, direction: str) -> bool:
        """Check if boost move (2 moves) is safe"""
        my_agent = game.agent1 if player_number == 1 else game.agent2
        other_agent = game.agent2 if player_number == 1 else game.agent1
        
        head = my_agent.trail[-1]
        dx, dy = Direction[direction].value
        
        # First move position
        pos1 = ((head[0] + dx) % game.board.width, 
               (head[1] + dy) % game.board.height)
        
        # Second move position (boost moves twice)
        pos2 = ((pos1[0] + dx) % game.board.width,
               (pos1[1] + dy) % game.board.height)
        
        # Check if both positions are safe
        if game.board.get_cell_state(pos1) != EMPTY:
            return False
        if game.board.get_cell_state(pos2) != EMPTY:
            return False
        
        # Check trails
        if pos1 in list(my_agent.trail)[:-1]:
            return False
        if pos2 in list(my_agent.trail)[:-1]:
            return False
        if other_agent.alive:
            if pos1 in other_agent.trail:
                return False
            if pos2 in other_agent.trail:
                return False
        
        return True


