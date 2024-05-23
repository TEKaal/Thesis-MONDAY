import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.layers import Input
import csv
import matplotlib.pyplot as plt
from datetime import datetime


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate, memory_size, num_layers, layers_size, epsilon_d, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)  # Increased! Memory for experience replay
        self.gamma = gamma  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = epsilon_d
        self.epsilon_min = 0.01
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.layer_size = layers_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    # def _build_model(self):
    #     """Builds a deeper neural network for the DQN agent."""
    #     model = Sequential()
    #     model.add(Input(shape=(self.state_size,)))
    #     model.add(Dense(128, activation='relu'))
    #     model.add(Dense(128, activation='relu'))
    #     model.add(Dense(64, activation='relu'))
    #     model.add(Dense(64, activation='relu'))
    #     model.add(Dense(self.action_size, activation='linear'))
    #     model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
    #     return model

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        for _ in range(self.num_layers):
            model.add(Dense(self.layer_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def save(self, filepath, rewards_data):
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Total Reward'])
            writer.writerows(rewards_data)

    def update_target_model(self):
        """Updates the target model to be the same as the model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in the memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Returns actions using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if (len(self.memory) < batch_size):
            return  # Not enough memory to sample from

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([t[0].reshape(self.state_size) for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3].reshape(self.state_size) for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # Predict the Q values of next states
        next_Q_values = self.target_model.predict(next_states)
        # Update the Q values for each state based on whether episode is done
        target_Q_values = rewards + self.gamma * np.max(next_Q_values, axis=1) * (1 - dones)

        # Predict the Q values of initial states
        target_f = self.model.predict(states)
        for i, action in enumerate(actions):
            target_f[i][action] = target_Q_values[i]

        # Train the model with states and their new Q-values
        self.model.fit(states, target_f, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn_agent(env, outputfolder, episodes, nr_steps, batch_size, learning_rate, memory_size, num_layers, layer_size, epsilon_decay, gamma):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, learning_rate, memory_size, num_layers, layer_size, epsilon_decay, gamma)
    rewards_data = []
    total_rewards = 0

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        counter = 0
        while (counter < nr_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            counter = counter + 1
            print(counter)
            print(env.current_step)

        print(f'Episode: {e + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2}')
        rewards_data.append(total_reward)  # Append episode number and total reward
        total_rewards =+ total_reward
    average_reward = total_rewards / episodes
    print(f'Average Reward: {average_reward}')

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_data, marker='o', linestyle='-', color='b')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode Training')
    plt.grid(True)
    # Huidige datum en tijd verkrijgen en formatteren
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Opslaan van de figuur met datum en tijd in de bestandsnaam
    plt.savefig(f"{outputfolder}\\plot_training_rewards_{current_datetime}.png")
    #plt.show()

    agent.update_target_model()
    # Save rewards to CSV for training
    with open((f"{outputfolder}\\training_rewards_{current_datetime}.csv"), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Reward'])
        writer.writerows([[i + 1, r] for i, r in enumerate(rewards_data)])

    return agent


def evaluate_dqn_agent(env, outputfolder, agent, episodes, nr_steps):
    episode_rewards_list = []
    total_rewards = 0
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    for e in range(episodes):
        state = env.reset()
        print(env.initial_step)
        print(env.final_step)
        state = np.reshape(state, [1, agent.state_size])
        done = False
        episode_rewards = 0
        counter = 0
        actions = []
        while (counter < nr_steps):
            action = np.argmax(agent.model.predict(state)[0])
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            state = next_state
            episode_rewards += reward
            counter = counter + 1
            print(counter)
            print(env.current_step)

        episode_rewards_list.append(episode_rewards)
        print(f'Episode: {e + 1}/{episodes}, Reward: {episode_rewards}')
        total_rewards += episode_rewards

    average_reward = total_rewards / episodes
    print(f'Average Reward: {average_reward}')

    # Plot the rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards_list, marker='o', linestyle='-', color='r')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode Evaluation')
    plt.grid(True)

    plt.savefig(f"{outputfolder}\\plot_evaluation_rewards_{current_datetime}.png")
    #plt.show()

    # Save actions to CSV for evaluation
    with open((f"{outputfolder}\\actions_{current_datetime}.csv"), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Actions'])
        for i, actions in enumerate(actions):
            writer.writerow([i + 1, actions])

    # Save rewards to CSV for evaluation
    with open((f"\\output_rewards_evaluation_{current_datetime}.csv"), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Reward'])
        writer.writerows([[i + 1, r] for i, r in enumerate(episode_rewards_list)])

    return average_reward, episode_rewards_list