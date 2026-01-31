import numpy as np
import matplotlib.pyplot as plt
import time

# ===================== MAZE CLASS =====================

class Maze:
    def __init__(self, maze, start_position, goal_position):
        self.maze = maze
        self.maze_height = maze.shape[0]
        self.maze_width = maze.shape[1]
        self.start_position = start_position  # (row, col)
        self.goal_position = goal_position    # (row, col)

    def show_maze(self):
        plt.figure(figsize=(5,5))
        plt.imshow(self.maze, cmap='gray')
        # note: plt.text uses (x=col, y=row)
        plt.text(self.start_position[1], self.start_position[0], 'S',
                 ha='center', va='center', color='red', fontsize=20)
        plt.text(self.goal_position[1], self.goal_position[0], 'G',
                 ha='center', va='center', color='green', fontsize=20)
        plt.xticks([]); plt.yticks([])
        plt.show()


# ===================== MAZE LAYOUT (7x7 with dead-ends) =====================

maze_layout = np.array([
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0]
])

maze = Maze(maze_layout, (0, 0), (6, 6))
maze.show_maze()


# ===================== Q-LEARNING AGENT =====================

# actions are (drow, dcol): UP, DOWN, LEFT, RIGHT
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class QLearningAgent:
    def __init__(self, maze,
                 learning_rate=0.3,
                 discount_factor=0.95,
                 exploration_start=1.0,
                 exploration_end=0.001,
                 num_episodes=500):
        self.maze = maze
        self.q_table = np.zeros((maze.maze_height, maze.maze_width, len(actions)))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_start = exploration_start
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes

    def get_exploration_rate(self, current_episode):
        # exponential decay between start and end across self.num_episodes
        # guard divide-by-zero
        if self.num_episodes <= 0:
            return self.exploration_end
        frac = current_episode / float(self.num_episodes)
        return self.exploration_start * (self.exploration_end / self.exploration_start) ** frac

    def get_action(self, state, current_episode, forcing_greedy=False):
        """
        state: (row, col)
        forcing_greedy: if True, always choose greedy (used in testing)
        """
        if forcing_greedy:
            return int(np.argmax(self.q_table[state]))
        if np.random.rand() < self.get_exploration_rate(current_episode):
            return np.random.randint(len(actions))
        return int(np.argmax(self.q_table[state]))

    def update_q_table(self, state, action, next_state, reward):
        best_next = np.argmax(self.q_table[next_state])
        current_q = self.q_table[state][action]
        target = reward + self.discount_factor * self.q_table[next_state][best_next]
        new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table[state][action] = new_q


# ===================== REWARDS (tweakable) =====================

goal_reward = 500.0
wall_penalty = -5.0
step_penalty = -0.1


# ===================== EPISODE SIMULATION =====================

def finish_episode(agent, maze, current_episode, train=True, max_steps=1000, greedy_on_test=True):
    """
    Simulate one episode from start until goal (or max_steps).
    If train=True, update Q-table; otherwise do not update.
    If greedy_on_test=True and train=False, picks greedy action (no exploration).
    Returns: (episode_reward, episode_steps, path_list)
    """
    current_state = maze.start_position
    is_done = False
    episode_reward = 0.0
    episode_step = 0
    path = [current_state]

    while not is_done and episode_step < max_steps:
        action = agent.get_action(current_state, current_episode, forcing_greedy=(not train and greedy_on_test))
        new_r = current_state[0] + actions[action][0]
        new_c = current_state[1] + actions[action][1]
        next_state = (new_r, new_c)

        # OUT OF BOUNDS or WALL
        if (new_r < 0 or new_r >= maze.maze_height or
            new_c < 0 or new_c >= maze.maze_width or
            maze.maze[new_r][new_c] == 1):
            reward = wall_penalty
            next_state = current_state  # stay in place
        # REACHED GOAL
        elif next_state == maze.goal_position:
            reward = goal_reward
            path.append(next_state)
            is_done = True
        # NORMAL STEP
        else:
            reward = step_penalty
            path.append(next_state)

        episode_reward += reward
        episode_step += 1

        if train:
            agent.update_q_table(current_state, action, next_state, reward)

        current_state = next_state

    return episode_reward, episode_step, path


# ===================== TRAINING LOOP =====================

def train_agent(agent, maze, num_episodes=500, max_steps_per_episode=500, verbose=False):
    # keep agent.num_episodes in sync so exploration decay uses same horizon
    agent.num_episodes = num_episodes

    rewards = []
    steps = []

    for episode in range(num_episodes):
        ep_reward, ep_steps, _ = finish_episode(agent, maze, episode, train=True, max_steps=max_steps_per_episode)
        rewards.append(ep_reward)
        steps.append(ep_steps)
        if verbose and (episode % max(1, num_episodes//10) == 0):
            print(f"Episode {episode}/{num_episodes} -> reward {ep_reward:.1f}, steps {ep_steps}")

    # plot results
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(rewards)
    plt.title("Reward per Episode")
    plt.xlabel("Episode"); plt.ylabel("Reward")

    plt.subplot(1,2,2)
    plt.plot(steps)
    plt.title("Steps per Episode")
    plt.xlabel("Episode"); plt.ylabel("Steps")
    plt.tight_layout()
    plt.show()

    return rewards, steps


# ===================== TEST / VISUALIZE =====================

def test_agent(agent, maze, greedy=True):
    # use a high episode index so exploration rate is minimal if not forcing greedy
    episode_index_for_test = agent.num_episodes
    ep_reward, ep_steps, path = finish_episode(agent, maze, episode_index_for_test, train=False, greedy_on_test=greedy)

    print("Learned Path:")
    for r, c in path:
        print(f"({r},{c})-> ", end='')
    print("GOAL")
    print(f"Steps: {ep_steps}, Total reward: {ep_reward:.1f}")

    plt.figure(figsize=(5,5))
    plt.imshow(maze.maze, cmap='gray')
    plt.text(maze.start_position[1], maze.start_position[0], 'S',
             ha='center', va='center', color='red', fontsize=18)
    plt.text(maze.goal_position[1], maze.goal_position[0], 'G',
             ha='center', va='center', color='green', fontsize=18)

    for r, c in path:
        plt.text(c, r, '#', color='blue', fontsize=14)

    plt.xticks([]); plt.yticks([])
    plt.show()

    return ep_steps, ep_reward


# ===================== ANIMATION (step-by-step) =====================

def animate_agent(agent, maze, delay=0.2):
    """Animate one greedy episode (agent must be trained for best result)."""
    # produce greedy path
    _, _, path = finish_episode(agent, maze, agent.num_episodes, train=False, greedy_on_test=True)

    fig, ax = plt.subplots(figsize=(5,5))
    for pos in path:
        ax.clear()
        ax.imshow(maze.maze, cmap='gray')
        ax.text(maze.start_position[1], maze.start_position[0], 'S',
                ha='center', va='center', color='red', fontsize=18)
        ax.text(maze.goal_position[1], maze.goal_position[0], 'G',
                ha='center', va='center', color='green', fontsize=18)
        ax.text(pos[1], pos[0], 'A', color='blue', fontsize=18)
        # draw path so far
        idx = path.index(pos)
        for p in path[:idx+1]:
            ax.text(p[1], p[0], '.', color='blue', fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        plt.pause(delay)
    plt.show()


# ===================== RUN: train -> test -> animate =====================

# create agent with chosen hyperparams (you can tweak these)
agent = QLearningAgent(maze,
                       learning_rate=0.3,
                       discount_factor=0.95,
                       exploration_start=1.0,
                       exploration_end=0.001,
                       num_episodes=500)

# Train (set num_episodes you want)
train_agent(agent, maze, num_episodes=400, max_steps_per_episode=200, verbose=True)

# Test the trained agent (greedy)
test_agent(agent, maze, greedy=True)

# Animate the agent moving along its learned path
animate_agent(agent, maze, delay=0.25)
