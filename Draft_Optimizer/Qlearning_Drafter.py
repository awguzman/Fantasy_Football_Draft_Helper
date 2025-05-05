"""
This file constructs and trains a multi-agent Q-Learning algorithm with epsilon-greedy exploration to optimally draft
a fantasy football team.
"""

import pandas as pd
import random
import json
import matplotlib.pyplot as plt
from collections import defaultdict

class QAgent:

    def __init__(self, team_id: int):
        """ Initialize an individual agent for a team. """
        self.team_id = team_id  # Team identification for this agent

        # Learning hyperparameters
        self.learning_rate = 0.2
        self.discount_factor = 0.9

        # Epsilon_greedy exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.q_table = defaultdict(float)  # Q-table to store state-action values for this agent

        self.drafted_players = []  # List to store drafted players for this agent
        self.total_reward = 0.0  # Store the total accumulated reward for this agent
        self.total_points = 0.0  # Store total projected fantasy points for this agent.
        self.position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}  # Track drafted positions

    def reset_agent(self) -> None:
        """Reset the agent's state for a new episode."""
        self.drafted_players = []
        self.total_reward = 0.0
        self.total_points = 0.0
        self.position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}

    def get_state(self) -> tuple:
        """Get the current state representation for the agent."""
        return tuple(self.position_counts.values())

    def choose_action(self, state: tuple) -> str:
        """Choose an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(list(self.position_counts.keys()))  # Random position with probability epsilon
        else:
            return max(self.position_counts.keys(),
                       key=lambda position: self.q_table[(state, position)])  # Otherwise, best position

    def update_q_table(self, state: tuple, action: str, reward: float, next_state: tuple) -> None:
        """Update the Q-table using the Q-learning formula."""
        best_next_action = max(self.position_counts.keys(),
                               key=lambda position: self.q_table[(next_state, position)])

        # Compute temporal difference.
        td_target = reward + self.discount_factor * self.q_table[(next_state, best_next_action)]
        td_delta = td_target - self.q_table[(state, action)]

        # Update Q-table state-action pair.
        self.q_table[(state, action)] += self.learning_rate * td_delta


class FantasyDraft:

    def __init__(self, player_data: pd.DataFrame, num_teams: int, num_rounds: int, position_limits: dict):
        """ Initialize the multi-agent draft simulation. """
        self.player_data = player_data.sort_values(by="projected_points", ascending=False)  # Expects a pandas DataFrame.
        self.num_teams = num_teams
        self.num_rounds = num_rounds
        self.position_limits = position_limits  # Max position count prior to penalization.
        self.draft_order = list(range(num_teams))

        self.agents = [QAgent(team_id=i) for i in range(num_teams)]  # Initialize an agent for each team.
        self.reward_history = {i: [] for i in range(num_teams)}  # Track rewards for debug purposes.

        # Cache max possible points by position for reward normalization.
        self.max_points_by_position = player_data.groupby("position")["projected_points"].max()

    def reset_draft(self) -> None:
        """Reset the draft for a new episode."""
        self.available_players = self.player_data.copy()
        self.current_round = 0
        self.current_team = 0
        self.draft_order = list(range(self.num_teams))  # Reset draft order

        # Reset all agents
        for agent in self.agents:
            agent.reset_agent()

    def run_episode(self, verbose=False) -> None:
        """Run a single episode of the draft."""
        self.reset_draft()

        while self.current_round < self.num_rounds:
            for team in self.draft_order:
                agent = self.agents[team]
                state = agent.get_state()

                # Agent chooses a position to draft given the current state.
                position = agent.choose_action(state)

                # Get the top player for the chosen position
                available_position_players = self.available_players[self.available_players["position"] == position]

                # If there are no available players at the action position, punish and lose turn.
                if available_position_players.empty:
                    reward = -1
                    agent.total_reward += reward
                    next_state = state
                    agent.update_q_table(state, position, reward, next_state)
                    continue

                drafted_player = available_position_players.iloc[0]  # Assumes draft board is sorted by projected_points
                drafted_player_index = drafted_player.name

                # Update agent stats with the drafted players information.
                agent.total_points += drafted_player["projected_points"]
                agent.drafted_players.append(drafted_player["player_name"] + " " + drafted_player["Rank"])
                agent.position_counts[position] += 1

                # Compute and store reward.
                reward = self.get_reward(drafted_player, agent)
                agent.total_reward += reward

                # Drop the drafted player from the draft board.
                self.available_players = self.available_players.drop(drafted_player_index)

                # Update the Q-table state-action pair.
                next_state = agent.get_state()
                agent.update_q_table(state, position, reward, next_state)

            self.current_round += 1  # Move to next round after all teams have picked.
            self.draft_order.reverse()  # Reverse the draft order for snake draft formats.

        # Print episode summary
        if verbose:
            sum_rewards, sum_points = 0, 0
            for agent in self.agents:
                sum_rewards += agent.total_reward
                sum_points += agent.total_points
                print(
                    f"  Team {agent.team_id}: Total Reward = {round(agent.total_reward, 2)}, Position Counts = {agent.position_counts}, Drafted Players = {agent.drafted_players} ({round(agent.total_points, 2)} pts)")
            avg_reward = sum_rewards / self.num_teams
            avg_points = sum_points / self.num_teams
            print(f"Average total reward = {avg_reward}, Average total fantasy points = {avg_points}")


    def get_reward(self, drafted_player: pd.Series, agent: QAgent) -> float:
        """Calculate the reward attained for drafting a given player by normalizing it with respect to the maximum
        possible points for that position. If we are exceeding position limits, give negative reward."""
        reward = drafted_player["projected_points"] / self.max_points_by_position[drafted_player["position"]]

        # Penalty for overdrafting a position.
        if agent.position_counts[drafted_player["position"]] > self.position_limits[drafted_player["position"]]:
            over_draft_penalty = agent.position_counts[drafted_player["position"]] - self.position_limits[
                drafted_player["position"]]

            # Stronger penalty if overdrafting while another position is empty.
            if 0 in agent.position_counts.values():
                over_draft_penalty += 1
            reward = -(over_draft_penalty * reward)

            # Clip maximum negative reward.
            if reward < -1:
                reward = -1

        return reward

    def train(self, num_episodes: int, verbose=False) -> None:
        """Train the agents over multiple episodes."""
        for episode in range(num_episodes):
            self.run_episode(verbose=False)
            for agent in self.agents:
                agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)  # decay epsilon value.
                self.reward_history[agent.team_id].append(agent.total_reward)  # Log rewards for debug purposes.

            # Print training status
            if verbose:
                print(f"Episode {episode + 1}/{num_episodes} completed.")

    def plot_rewards(self) -> None:
        """Plot the learning progress."""
        plt.figure(figsize=(12, 6))
        for team_id, rewards in self.reward_history.items():
            # Compute a moving average for total rewards.
            smoothed_rewards = pd.Series(rewards).rolling(window=50).mean()
            plt.plot(smoothed_rewards, label=f"Team {team_id + 1} Total Rewards")

        plt.title("Total Rewards Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward (Moving Average)")
        plt.legend()
        plt.show()


def run_training_routine() -> None:
    """Runs a full training routine and saves resulting Q-table for competitive evaluation."""
    # Pandas database of 400 player draft board from FantasyPros.com
    player_data = pd.read_csv("../Best_Ball/Best_Ball_Draft_Board.csv").drop('Unnamed: 0', axis=1).rename(columns={
        "Player": "player_name", "POS": "position", "Fantasy Points": "projected_points"})

    # Setup draft environment.
    num_teams = 12
    num_rounds = 20
    position_limits = {"QB": 3, "RB": 7, "WR": 8, "TE": 3}
    draft_simulator = FantasyDraft(player_data, num_teams, num_rounds, position_limits)

    # Setup training routine.
    epsilons = [1.0, 0.5, 0.25]
    epsilon_mins = [0.1, 0.05, 0]
    epsilon_decays = [0.9993, 0.9985, 0.99]
    num_episodes = [3000, 1500, 500]

    # Run agents through an incremented training routine.
    for phase in range(len(num_episodes)):
        for agent in draft_simulator.agents:  # Update epsilon+greedy parameters each phase.
            agent.epsilon = epsilons[phase]
            agent.epsilon_min = epsilon_mins[phase]
            agent.epsilon_decay = epsilon_decays[phase]

        print(f"\nBeginning training phase {phase + 1}. Number of episodes in this phase is {num_episodes[phase]}.")
        draft_simulator.train(num_episodes=num_episodes[phase], verbose=False)
        print(f"Training phase {phase + 1} complete. Running a test draft with no exploitation.")
        for agent in draft_simulator.agents:
            agent.epsilon, agent.epsilon_decay, agent.epsilon_min = 0, 0, 0  # Pure exploitation.
        draft_simulator.run_episode(verbose=True)

    # Plot rewards for evaluation.
    draft_simulator.plot_rewards()

    # Save trained Q-tables as JSON for competitive evaluation.
    def save_q_table_to_json(q_table: defaultdict, filename: str) -> None:
        # Convert defaultdict to a dictionary then to a .json
        q_table = {str(k): v for k, v in q_table.items()}
        with open(filename, 'w') as json_file:
            json.dump(q_table, json_file, indent=4)

    for agent in draft_simulator.agents:
        save_q_table_to_json(agent.q_table, filename=f"Trained_Agents/Q_Agents/QAgent_{agent.team_id}_Q_table.json")

if __name__ == "__main__":
    run_training_routine()