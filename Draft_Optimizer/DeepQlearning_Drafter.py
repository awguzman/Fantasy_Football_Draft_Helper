import pandas as pd
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """Define neural network used to predict our Q-values."""
    def __init__(self, state_size, action_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class QAgent:
    def __init__(self, team_id, state_size, action_size, learning_rate=0.001, discount_factor=0.9, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        """ Initialize an individual agent for a team. """
        self.team_id = team_id  # Team identification for this agent
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_network = QNetwork(state_size, action_size, hidden_size=action_size*2)
        self.target_network = QNetwork(state_size, action_size, hidden_size=action_size*2)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.drafted_players = []  # List to store drafted players for this agent
        self.total_reward = 0  # Store the total accumulated reward for this agent
        self.position_counts = {position: 0 for position in position_limits}  # Track drafted positions

    def reset_agent(self):
        """Reset the agent's state for a new episode."""
        self.drafted_players = []
        self.total_reward = 0
        self.position_counts = {position: 0 for position in position_limits}  # Reset position counts

    def get_state(self, round_number, all_agents):
        """Get the current state representation for the agent, including information about other teams."""
        # Current agent's state
        position_counts_tensor = torch.tensor(list(self.position_counts.values()), dtype=torch.float32)
        round_tensor = torch.tensor([round_number], dtype=torch.float32)

        # Other teams' position counts
        other_teams_counts = []
        for agent in all_agents:
            if agent.team_id != self.team_id:
                other_teams_counts.extend(agent.position_counts.values())
        other_teams_tensor = torch.tensor(other_teams_counts, dtype=torch.float32)

        # Combine into a single state tensor
        return torch.cat((position_counts_tensor, round_tensor, other_teams_tensor))

    def choose_action(self, state, available_players):
        """Choose an action using an epsilon-greedy policy."""
        state_tensor = state.unsqueeze(0)  # Add batch dimension
        available_indices = available_players.index.tolist()  # Get valid indices

        if random.random() < self.epsilon:  # With probability epsilon, choose a random action.
            return random.choice(available_indices)
        else:  # Otherwise, choose the best action.
            with torch.no_grad():
                q_values = self.q_network(state_tensor).squeeze(0)  # Get Q-values for all actions

                # Mask out unavailable actions
                q_values_masked = torch.full_like(q_values, float('-inf'))
                q_values_masked[available_indices] = q_values[available_indices]

                return q_values_masked.argmax().item()

    def update_q_network(self, state, action, reward, next_state, done):
        """Update the Q-network using the Q-learning formula."""
        state_tensor = state.unsqueeze(0)
        next_state_tensor = next_state.unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.long)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        done_tensor = torch.tensor([done], dtype=torch.float32)

        # Compute current Q-value
        q_value = self.q_network(state_tensor).gather(1, action_tensor.unsqueeze(1))

        # Compute target Q-value
        with torch.no_grad():
            next_q_value = self.target_network(next_state_tensor).max(1)[0]
            target_q_value = reward_tensor + (1 - done_tensor) * self.discount_factor * next_q_value

        # Compute loss and update Q-network
        loss = self.loss_fn(q_value, target_q_value.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class FantasyDraft:
    def __init__(self, player_data, num_teams, num_rounds, state_size, action_size):
        """ Initialize the multi-agent draft simulation. """
        self.player_data = player_data  # Expects a pandas DataFrame.
        self.num_teams = num_teams
        self.num_rounds = num_rounds
        self.agents = [QAgent(team_id=i, state_size=state_size, action_size=action_size) for i in range(num_teams)]
        self.reset_draft()
        self.reward_history = {i: [] for i in range(num_teams)}  # Track rewards for debug purposes.
        self.draft_order = list(range(num_teams))

    def reset_draft(self):
        """Reset the draft for a new episode."""
        self.available_players = self.player_data.copy()
        self.current_round = 0
        self.current_team = 0
        self.draft_order = list(range(self.num_teams))  # Reset draft order
        for agent in self.agents:
            agent.reset_agent()

    def run_episode(self, verbose=False):
        """Run a single episode of the draft."""
        self.reset_draft()
        while self.current_round < self.num_rounds:
            for team in self.draft_order:
                agent = self.agents[team]
                state = agent.get_state(self.current_round, self.agents)

                # Filter available players to respect position caps
                valid_players = self.available_players[
                    self.available_players['position'].apply(
                        lambda pos: agent.position_counts[pos] < position_limits[pos]
                    )
                ]

                # Check if there are any draftable players.
                if valid_players.empty:
                    raise Exception("There are no valid players for the agent to draft from!")

                action = agent.choose_action(state, valid_players)

                drafted_player = self.available_players.loc[action]
                agent.drafted_players.append(drafted_player["player_name"])
                agent.position_counts[drafted_player["position"]] += 1  # Increment position count

                # Compute reward for this action.
                reward = self.get_reward(drafted_player)
                agent.total_reward += reward

                self.available_players = self.available_players.drop(action)

                next_state = agent.get_state(self.current_round + 1, self.agents)
                agent.update_q_network(state, action, reward, next_state, done=False)

            self.current_round += 1  # Move to next round after all teams have picked.
            self.draft_order.reverse()  # Reverse the draft order for snake draft formats.

        if verbose:
            for agent in self.agents:
                print(
                    f"  Team {agent.team_id}: Total Reward = {round(agent.total_reward, 2)}, Drafted Players = {agent.drafted_players}")

    def get_reward(self, drafted_player):
       """Calculate the reward attained for drafting a given player"""
       proj_points = drafted_player["projected_points"]

       # Get total value lost from not picking the best possible player.
       max_points_by_position = self.available_players.groupby("position")["projected_points"].max()
       loss_penalty = drafted_player["projected_points"] - max_points_by_position[drafted_player["position"]]

       total_reward = proj_points + loss_penalty
       return total_reward

    def train(self, num_episodes, verbose=False):
        """Train the agents over multiple episodes."""
        target_update_frequency = 10
        for episode in range(num_episodes):
            self.run_episode(verbose=False)
            for agent in self.agents:
                agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)  # decay epsilon value.
                self.reward_history[agent.team_id].append(agent.total_reward)  # Log rewards for debug purposes.

            if episode % target_update_frequency == 0:
                for agent in self.agents:
                    agent.target_network.load_state_dict(agent.q_network.state_dict())

            # Print episode summary
            print(f"Episode {episode + 1}/{num_episodes} completed.")
            if verbose:
                for agent in self.agents:
                    print(
                        f"  Team {agent.team_id}: Total Reward = {round(agent.total_reward, 2)}, Drafted Players = {agent.drafted_players}")

    def plot_results(self):
        """Plot the learning progress for debug purposes."""
        # Plot total rewards
        plt.figure(figsize=(12, 6))
        for team_id, rewards in self.reward_history.items():
            # Compute a moving average for total rewards.
            smoothed_rewards = pd.Series(rewards).rolling(window=10).mean()
            plt.plot(smoothed_rewards, label=f"Team {team_id} Total Rewards")
        plt.title("Total Rewards Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward (Moving Average)")
        plt.legend()
        plt.show()

# Debug draft environment
player_data = pd.DataFrame({
    "player_name": ["QB1", "QB2", "QB3", "QB4", "QB5", "RB1", "RB2", "RB3", "RB4", "RB5",
                    "WR1", "WR2", "WR3", "WR4", "WR5", "TE1", "TE2", "TE3", "TE4", "TE5"],
    "position": ["QB", "QB", "QB", "QB", "QB", "RB", "RB", "RB", "RB", "RB",
                 "WR", "WR", "WR", "WR", "WR", "TE", "TE", "TE", "TE", "TE"],
    "projected_points": [360, 330, 300, 270, 240, 280, 220, 180, 150, 120,
                         210, 170, 150, 140, 120, 140, 110, 80, 70, 60]
})

# player_data = pd.read_csv("../Best_Ball/Best_Ball_Draft_Board.csv").drop('Unnamed: 0', axis=1).rename(columns={
#     "Player": "player_name", "POS": "position", "Fantasy Points": "projected_points"})

num_teams = 2
num_rounds = 4
position_limits = {"QB": 1, "RB": 1, "WR": 1, "TE": 1}
state_size = len(position_limits) + 1 + (len(position_limits) * (num_teams - 1))  # position_counts + round_number + other_teams_position_counts
action_size = len(player_data)
draft_simulator = FantasyDraft(player_data, num_teams, num_rounds, state_size, action_size)

# Debug Training
draft_simulator.train(1000, verbose=False)
draft_simulator.plot_results()
# Now run a draft with no exploration.
for agent in draft_simulator.agents:
    agent.epsilon = 0
draft_simulator.run_episode(verbose=True)
