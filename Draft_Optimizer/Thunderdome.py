"""This file runs a series of drafts to evaluate the performance of the various RL algorithms.
 We expect each algorithm is trained on the following parameters:
num_teams = 12
num_rounds = 20
position_limits = {"QB": 3, "RB": 7, "WR": 8, "TE": 3}
"""

import pandas as pd
import json
from collections import defaultdict
import torch
import random
import matplotlib.pyplot as plt

from DeepQlearning_Drafter import QNetwork
from A2C_Drafter import ActorNetwork as A2CActorNetwork
from PPO_Drafter import ActorNetwork as PPOActorNetwork

class QAgent:

    def __init__(self, team_id):
        """ Initialize an individual agent for a team. """
        self.team_id = team_id  # Team identification for this agent
        self.agent_type = "Q-Agent"

        self.drafted_players = []  # List to store drafted players for this agent
        self.total_reward = 0  # Store the total accumulated reward for this agent
        self.total_points = 0  # Store total projected fantasy points for this agent.
        self.position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}  # Track drafted positions

         # Load in the Q-table associated with the particular team_id
        with open(f"Trained_Agents/Q_Agents/QAgent_{self.team_id}_Q_table.json", 'r') as json_file:
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
        return tuple(self.position_counts.values())

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
        network.load_state_dict(torch.load(f"Trained_Agents/Deep_Q_Agents/DeepQAgent_{self.team_id}_Q_Network.pt"))
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
        position_counts_tensor = torch.tensor(list(self.position_counts.values()), dtype=torch.float32) # Current agent's composition.

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


class A2CAgent:

    def __init__(self, team_id):
        """ Initialize an individual agent for a team. """
        self.team_id = team_id  # Team identification for this agent
        self.agent_type = "A2C Agent"

        # Initialize Actor network for this specific team_id
        actor_network = A2CActorNetwork(state_size=48, action_size=4, hidden_layers=[36, 36, 36])
        actor_network.load_state_dict(torch.load(f"Trained_Agents/A2C_Agents/A2CAgent_{team_id}_Actor_Network.pt"))
        actor_network.eval()
        self.actor_network = actor_network

        self.drafted_players = []  # List to store drafted players for this agent
        self.total_reward = 0  # Store the total accumulated reward for this agent
        self.total_points = 0  # Store the total accumulated fantasy points for this agent.
        self.position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}  # Track drafted position counts

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
        """Choose an action using the actor network."""
        probs = self.actor_network(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


class PPOAgent:

    def __init__(self, team_id):
        """ Initialize an individual agent for a team. """
        self.team_id = team_id  # Team identification for this agent
        self.agent_type = "PPO Agent"

        # Initialize Actor network for this specific team_id
        actor_network = PPOActorNetwork(state_size=48, action_size=4, hidden_layers=[36, 36, 36])
        actor_network.load_state_dict(torch.load(f"Trained_Agents/PPO_Agents/PPOAgent_{team_id}_Actor_Network.pt"))
        actor_network.eval()
        self.actor_network = actor_network

        self.drafted_players = []  # List to store drafted players for this agent
        self.total_reward = 0  # Store the total accumulated reward for this agent
        self.total_points = 0  # Store the total accumulated fantasy points for this agent.
        self.position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}  # Track drafted position counts

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
        """Choose an action using the actor network."""
        probs = self.actor_network(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()


class FantasyDraft:

    def __init__(self, player_data, num_teams, num_rounds):
        """ Initialize the multi-agent draft simulation. """
        self.player_data = player_data.sort_values(by="projected_points", ascending=False)  # Expects a pandas DataFrame.
        self.num_teams = num_teams
        self.num_rounds = num_rounds
        self.draft_order = list(range(num_teams))

        # Preload all agents.
        self.QAgents = [QAgent(team_id=i) for i in range(num_teams)]
        self.DeepQAgents = [DeepQAgent(team_id=j) for j in range(num_teams)]
        self.A2CAgents = [A2CAgent(team_id=k) for k in range(num_teams)]
        self.PPOAgents = [PPOAgent(team_id=l) for l in range(num_teams)]

        # Track the number of wins by total points of each type of agent.
        self.agent_type_totals = {"Q-Agent": 0, "Deep Q-Agent": 0, "A2C Agent": 0, "PPO Agent": 0}
        self.agent_type_wins = {"Q-Agent": 0, "Deep Q-Agent": 0, "A2C Agent": 0, "PPO Agent": 0}
        self.win_history = {i: [] for i in ["Q-Agent", "Deep Q-Agent", "A2C Agent", "PPO Agent"]}

    def reset_draft(self):
        """Reset the draft for a new draft."""
        self.available_players = self.player_data.copy()
        self.current_round = 0
        self.current_team = 0
        self.draft_order = list(range(self.num_teams))  # Reset draft order
        self.agent_type_totals = {"Q-Agent": 0, "Deep Q-Agent": 0, "A2C Agent": 0, "PPO Agent": 0}

        # Randomly but equally distribute a selection of the preloaded agents.
        early_picks, mid_picks, late_picks = self.draft_order[0:4], self.draft_order[4:8], self.draft_order[8:12]
        random.shuffle(early_picks), random.shuffle(mid_picks), random.shuffle(late_picks)
        self.q_agent_ids = [early_picks[0], mid_picks[0], late_picks[0]]
        self.deep_q_agent_ids = [early_picks[1], mid_picks[1], late_picks[1]]
        self.a2c_agent_ids = [early_picks[2], mid_picks[2], late_picks[2]]
        self.ppo_agent_ids = [early_picks[3], mid_picks[3], late_picks[3]]
        self.agents = ([self.QAgents[i] for i in self.q_agent_ids] +
                       [self.DeepQAgents[j] for j in self.deep_q_agent_ids] +
                       [self.A2CAgents[k] for k in self.a2c_agent_ids] +
                       [self.PPOAgents[l] for l in self.ppo_agent_ids])
        self.agents.sort(key=lambda agent: agent.team_id)  # Reorder agents by increasing team_id

        for agent in self.agents:
            agent.reset_agent()

    def run_draft(self, verbose=False):
        """Run a single draft."""
        self.reset_draft()
        while self.current_round < self.num_rounds:
            for team in self.draft_order:
                agent = self.agents[team]

                # Run the agent through the correct action selection for their type.
                # The Q-learning agent has a slightly different state and action representation needing distinct handling.
                if team in self.q_agent_ids:
                    state = agent.get_state()
                    position = agent.choose_action(state)  # Select a position to draft.

                else:
                    state = agent.get_state(self.agents)
                    action = agent.choose_action(state)  # Use the neural network to choose an action.
                    position = list(agent.position_counts.keys())[action]  # Get the position of the chosen action

                # Find the best player available at that position.
                available_position_players = self.available_players[self.available_players["position"] == position]
                drafted_player = available_position_players.iloc[0]  # Assumes draft board is sorted by projected_points
                drafted_player_index = drafted_player.name

                # Update agent stats
                agent.total_points += drafted_player["projected_points"]
                agent.drafted_players.append(drafted_player["player_name"] + " " + drafted_player["Rank"])
                agent.position_counts[position] += 1

                # Drop selected player from the draft board.
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

            # Total up how many points each type of agent got.
            for agent in self.agents:
                self.agent_type_totals[agent.agent_type] += agent.total_points

            # Give the win to the agent type with most total points.
            winning_type = max(self.agent_type_totals, key=lambda total: self.agent_type_totals[total])
            self.agent_type_wins[winning_type] += 1
            for type in self.agent_type_wins:
                self.win_history[type].append(self.agent_type_wins[type])

        for type in self.agent_type_wins:
            print(f"{type} Wins : {self.agent_type_wins[type]}")

    def plot_results(self):
        """Plot the learning progress for debug purposes."""
        plt.figure(figsize=(12, 6))
        for type, wins in self.win_history.items():
            plt.plot(wins, label=f"{type} Wins")

        plt.title("Total Wins Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Total Wins")
        plt.legend()
        plt.show()

# Pandas database of 400 player draft board from FantasyPros.com
player_data = pd.read_csv("../Best_Ball/Best_Ball_Draft_Board.csv").drop('Unnamed: 0', axis=1).rename(columns={
    "Player": "player_name", "POS": "position", "Fantasy Points": "projected_points"})
num_teams = 12
num_rounds = 20

draft_simulator = FantasyDraft(player_data, num_teams, num_rounds)

draft_simulator.run_evaluations(num_drafts=1000)
draft_simulator.plot_results()




