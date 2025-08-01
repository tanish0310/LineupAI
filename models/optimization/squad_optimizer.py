import pulp
from pulp import LpProblem, LpMaximize, LpVariable, lpSum
from collections import defaultdict

class SquadOptimizer:
    def __init__(self):
        self.BUDGET = 1000  # £100.0m in tenths
        self.POSITIONS = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        self.MAX_PER_TEAM = 3
        self.SQUAD_SIZE = 15

    def build_optimal_squad(self, predictions):
        prob = LpProblem("FPL_Squad_Optimization", LpMaximize)

        # Decision variables
        player_vars = {pid: LpVariable(f"player_{pid}", cat="Binary") for pid in predictions}

        # Objective: Maximize predicted points
        prob += lpSum([predictions[pid]['points'] * player_vars[pid] for pid in predictions])

        # Budget constraint
        prob += lpSum([predictions[pid]['price'] * player_vars[pid] for pid in predictions]) <= self.BUDGET

        # Squad size constraint
        prob += lpSum([player_vars[pid] for pid in predictions]) == self.SQUAD_SIZE

        # Position constraints
        for pos in self.POSITIONS:
            prob += lpSum([player_vars[pid] for pid in predictions if predictions[pid]['position'] == pos]) == self.POSITIONS[pos]

        # Team limit constraint (max 3 per team)
        team_counts = defaultdict(list)
        for pid in predictions:
            team_counts[predictions[pid]['team']].append(player_vars[pid])
        for team_players in team_counts.values():
            prob += lpSum(team_players) <= self.MAX_PER_TEAM

        # Solve
        prob.solve()

        # Extract selected player IDs
        selected_ids = [pid for pid in predictions if player_vars[pid].varValue == 1]
        return selected_ids

    def optimize_starting_xi(self, squad, predictions):
        valid_formations = [
            (1, 3, 4, 3), (1, 3, 5, 2), (1, 4, 3, 3),
            (1, 4, 4, 2), (1, 4, 5, 1), (1, 5, 3, 2), (1, 5, 4, 1)
        ]

        best_xi = None
        best_score = -1

        for formation in valid_formations:
            gk, def_, mid, fwd = formation
            gk_list = [p for p in squad if predictions[p]['position'] == 'GK']
            def_list = [p for p in squad if predictions[p]['position'] == 'DEF']
            mid_list = [p for p in squad if predictions[p]['position'] == 'MID']
            fwd_list = [p for p in squad if predictions[p]['position'] == 'FWD']

            if len(gk_list) < gk or len(def_list) < def_ or len(mid_list) < mid or len(fwd_list) < fwd:
                continue

            xi = gk_list[:gk] + def_list[:def_] + mid_list[:mid] + fwd_list[:fwd]
            total_points = sum(predictions[p]['points'] for p in xi)

            if total_points > best_score:
                best_xi = xi
                best_score = total_points

        return best_xi

    def recommend_captain_vice(self, starting_xi, predictions):
        def safe_score(pid):
            return predictions[pid].get('consistency', 0.8)

        def fixture_score(pid):
            return predictions[pid].get('fixture_difficulty_impact', 1.0)

        ranking = []
        for pid in starting_xi:
            base = predictions[pid]['points']
            score = base * 2 * (safe_score(pid) + fixture_score(pid)) / 2
            ranking.append((pid, score))

        ranking.sort(key=lambda x: x[1], reverse=True)
        return {
            'captain': ranking[0][0],
            'vice_captain': ranking[1][0],
            'top_3': [pid for pid, _ in ranking[:3]]
        }
