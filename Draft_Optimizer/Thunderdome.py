import pandas as pd
import json
from collections import defaultdict
import torch
import random
from DeepQlearning_Drafter import QNetwork

class QAgent:

    def __init__(self, team_id):
        """ Initialize an individual agent for a team. """
        self.team_id = team_id  # Team identification for this agent
        self.agent_type = "Q-Agent"
        self.drafted_players = []  # List to store drafted players for this agent
        self.total_reward = 0  # Store the total accumulated reward for this agent
        self.total_points = 0  # Store total projected fantasy points for this agent.
        self.position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}  # Track drafted positions

         # Load in the Q-table associated with the specific team_id
        with open(f"Trained_Agents/Q_Agents/QAgent_{team_id}_Q_table.json", 'r') as json_file:
            q_table_dict = json.load(json_file)
        q_table = defaultdict(float, {eval(k): v for k, v in q_table_dict.items()})
        self.q_table = q_table  # Q-table to store state-action values for this agent

    def reset_agent(self):
        """Reset the agent's state for a new episode."""
        self.drafted_players = []
        self.total_reward = 0
        self.total_points = 0
        self.position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}  # Track drafted positions

    def get_state(self):
        """Get the current state representation for the agent."""
        return tuple(sorted(self.position_counts.values()))

    def choose_action(self, state):
        """Choose best action from the Q-table."""
        return max(self.position_counts.keys(), key=lambda position: self.q_table[(state, position)])  # Best position


class DeepQAgent:

    def __init__(self, team_id):
        """ Initialize an individual agent for a team. """
        self.team_id = team_id  # Team identification for this agent
        self.agent_type = "Deep Q-Agent"
        self.drafted_players = []  # List to store drafted players for this agent
        self.total_reward = 0  # Store the total accumulated reward for this agent
        self.total_points = 0  # Store the total accumulated fantasy points for this agent.
        self.position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}  # Track drafted position counts

        # Load in the Q-network for this specific team_id
        network = QNetwork(state_size=48, action_size=4, hidden_layers=[36, 36, 36])
        network.load_state_dict(torch.load(f"Trained_Agents/Deep_Q_Agents/DeepQAgent_{team_id}_Q_Network.pt"))
        network.eval()
        self.q_network = network

    def reset_agent(self):
        """Reset the agent's initial state for a new episode."""
        self.drafted_players = []
        self.total_reward = 0
        self.total_points = 0
        self.position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}

    def get_state(self, all_agents):
        """Get the current state for the agent. We keep track of the position counts for all teams in the draft."""
        position_counts_tensor = torch.tensor(list(self.position_counts.values()), dtype=torch.float32) # Current agent's state

        # Other teams' position counts
        other_teams_counts = []
        for agent in all_agents:
            if agent.team_id != self.team_id:
                other_teams_counts.extend(agent.position_counts.values())
        other_teams_tensor = torch.tensor(other_teams_counts, dtype=torch.float32)

        # Combine into a single state tensor
        return torch.cat((position_counts_tensor, other_teams_tensor))

    def choose_action(self, state):
        """Choose the best action based on the Q-network."""
        state_tensor = state.unsqueeze(0)
        action = self.q_network(state_tensor).argmax(dim=1).item()
        return action


class FantasyDraft:
    def __init__(self, player_data, num_teams, num_rounds):
        """ Initialize the multi-agent draft simulation. """
        self.player_data = player_data.sort_values(by="projected_points", ascending=False)  # Expects a pandas DataFrame.
        self.num_teams = num_teams
        self.num_rounds = num_rounds
        self.draft_order = list(range(num_teams))

        # Track the number of wins by average points each type of agent gets.
        self.q_agent_wins = 0  #
        self.deep_q_agent_wins = 0

        self.reset_draft()

    def reset_draft(self):
        """Reset the draft for a new draft."""
        self.available_players = self.player_data.copy()
        self.current_round = 0
        self.current_team = 0
        self.draft_order = list(range(self.num_teams))  # Reset draft order

        # Randomly but equally distribute and initialize the agents.
        draft_positions = list(range(num_teams))
        random.shuffle(draft_positions)
        self.q_agent_ids = draft_positions[: num_teams // 2]
        self.deep_q_agent_ids = draft_positions[num_teams // 2:]
        self.agents = [QAgent(team_id=i) for i in self.q_agent_ids] + [DeepQAgent(team_id=j) for j in
                                                                       self.deep_q_agent_ids]
        self.agents.sort(key=lambda agent: agent.team_id)  # Reorder agents by increasing team_id



    def run_draft(self, verbose=False):
        """Run a single draft."""
        self.reset_draft()
        while self.current_round < self.num_rounds:
            for team in self.draft_order:
                agent = self.agents[team]
                # Run the agent through the correct action selection for their type.
                if team in self.q_agent_ids:
                    state = agent.get_state()
                    position = agent.choose_action(state)  # Select a position to draft.

                    # Find the best player available at that position.
                    available_position_players = self.available_players[self.available_players["position"] == position]
                    drafted_player = available_position_players.iloc[
                        0]  # Assumes draft board is sorted by projected_points
                    drafted_player_index = drafted_player.name

                    # Update agent stats
                    agent.total_points += drafted_player["projected_points"]
                    agent.drafted_players.append(drafted_player["player_name"] + " " + drafted_player["Rank"])
                    agent.position_counts[position] += 1

                    # Drop selected player from the draft board.
                    self.available_players = self.available_players.drop(drafted_player_index)

                else:
                    # Choose a position to draft.
                    state = agent.get_state(self.agents)
                    action = agent.choose_action(state)
                    position = list(agent.position_counts.keys())[action]

                    # Draft the best player at that position.
                    available_players = self.available_players[self.available_players["position"] == position]
                    drafted_player = available_players.iloc[0]
                    drafted_player_index = drafted_player.name

                    # Add this player to the team.
                    agent.total_points += drafted_player["projected_points"]
                    agent.drafted_players.append(drafted_player["player_name"] + " " + drafted_player["position"])
                    agent.position_counts[drafted_player["position"]] += 1

                    # Remove drafted player from the draft board.
                    self.available_players = self.available_players.drop(drafted_player_index)

            self.current_round += 1  # Move to next round after all teams have picked.
            self.draft_order.reverse()  # Reverse the draft order for snake draft formats.


        # Print episode summary
        if verbose:
            for agent in self.agents:
                print(
                    f"{agent.agent_type} {agent.team_id}: Drafted Players = {agent.drafted_players} ({round(agent.total_points, 2)} pts)")

    def run_evaluations(self, num_drafts):
        for draft in range(num_drafts):
            self.run_draft(verbose=False)

            # Compute average total points for each agent type.
            q_agent_totals, deep_q_agent_totals = 0, 0
            for agent in self.agents:
                if agent.team_id in self.q_agent_ids:
                    q_agent_totals += agent.total_points
                else:
                    deep_q_agent_totals += agent.total_points
            q_agent_avg = q_agent_totals / len(self.q_agent_ids)
            deep_q_agent_avg = deep_q_agent_totals / len(self.deep_q_agent_ids)
            if q_agent_avg >= deep_q_agent_avg:
                self.q_agent_wins += 1
            else:
                self.deep_q_agent_wins += 1

            print(f"Evaluation Draft {draft + 1} / {num_drafts} complete.")

        print(f"Q-learning agents wins: {self.q_agent_wins} | Deep Q-learning agent wins: {self.deep_q_agent_wins}")



# Pandas database of 400 player draft board from FantasyPros.com
player_data = pd.read_csv("../Best_Ball/Best_Ball_Draft_Board.csv").drop('Unnamed: 0', axis=1).rename(columns={
    "Player": "player_name", "POS": "position", "Fantasy Points": "projected_points"})
num_teams = 12
num_rounds = 20

draft_simulator = FantasyDraft(player_data, num_teams, num_rounds)

draft_simulator.run_evaluations(num_drafts=1000)




