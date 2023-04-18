import numpy as np
import torch
from collections import defaultdict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class roadGrid:
    def __init__(self, graph, n_agents, size):
        self.last_agent_transits = None
        self.agents_in_transit = None
        self.success = None
        self.step_counter = None
        self.done = None
        self.trajectory = None
        self.actions_taken = None
        self.final_states = None
        self.S = None
        self.T = None
        self.base_state = None
        self.agents_at_base_state = None
        self.timeout = 32
        self.G = graph
        self.n_agents = n_agents
        self.n_actions = 4  # max node degree
        self.n_states = len(self.G.nodes)  # to be defined
        self.size = size
        self.one_hot_enc = {(l, r): np.concatenate(
            [np.array([0 if i != l else 1 for i in range(self.size)]), np.array([0 if i != r else 1 for i in range(self.size)])]) for
            (l, r) in self.G.nodes()}

    def reset(self):
        self.step_counter = np.zeros(self.n_agents)
        self.T = np.zeros(self.n_agents)
        self.S = np.zeros((self.n_agents, 2)).astype(int)  # two dim, for two dim states
        self.final_states = np.ones((self.n_agents, 2)) * np.array([self.size-1, self.size-1])  # np.random.randint(0, 7, size=(self.n_agents, 2))
        self.actions_taken = np.zeros((self.size, self.size, self.n_actions)).astype(int)
        self.trajectory = []
        self.done = np.zeros(self.n_agents)
        info = {}
        state = [self.one_hot_enc[tuple(self.S[n])] for n in range(self.n_agents)]

        remaining_agents = np.ones(self.n_agents, dtype=bool)
        first_agents = remaining_agents
        uni = np.unique(self.S[first_agents], axis=0)

        self.base_state = tuple(uni[np.random.randint(len(uni))])
        self.agents_at_base_state = np.where((self.S == self.base_state).all(axis=1), True, False) * first_agents
        self.agents_in_transit = defaultdict(lambda: 0)
        self.last_agent_transits = defaultdict(lambda: None)

        # print(self.base_state, self.agents_at_base_state)

        return state, info, self.base_state, self.agents_at_base_state

    def step(self, actions):
        neighbour_nodes = [edge[1] for edge in self.G.edges(self.base_state)]
        # print(self.base_state)

        # actions = np.clip(actions, 0, len(neighbour_nodes) - 1)
        actions = actions.flatten()
        # print(actions)
        counts = np.bincount(actions, minlength=self.n_actions)
        self.actions_taken[self.base_state[0], self.base_state[1]] += counts

        reward_per_action = [self.G.adj[self.base_state][edge[1]]["cost"](
            counts[i]+self.agents_in_transit[edge]) for i, edge in
                             enumerate(self.G.edges(self.base_state))]

        if len(neighbour_nodes) < self.n_actions:
            reward_per_action = np.concatenate([reward_per_action, 1000*np.ones((self.n_actions-len(neighbour_nodes)))])
            for i in range(self.n_actions - len(neighbour_nodes)):
                neighbour_nodes.append(self.base_state)

        rewards = np.array([reward_per_action[a] for a in actions])

        self.S[self.agents_at_base_state] = np.array([neighbour_nodes[a] for a in actions]).astype(int)

        observations = [neighbour_nodes[a] for a in actions]

        self.T[self.agents_at_base_state] += rewards

        occupied_states, occupied_states_counts = np.unique(self.S, return_counts=True, axis=0)
        states_counts = dict(zip([tuple(s) for s in occupied_states], list(occupied_states_counts)))
        edges_actions = {(self.base_state, neighbour): counts[a] if a < len(neighbour_nodes) else None for a, neighbour in enumerate(neighbour_nodes)}
        self.trajectory.append(tuple([self.base_state, states_counts, edges_actions, rewards]))

        self.agents_in_transit.update(edges_actions)

        transitions = []

        for i, n in enumerate(np.argwhere(self.agents_at_base_state == True).flatten()):
            # print("in loop")
            self.last_agent_transits[n] = tuple([self.base_state, tuple(observations[i])])
            observation = self.one_hot_enc[tuple(observations[i])]
            action = actions[i]
            terminated = True if (observation == self.one_hot_enc[tuple(self.final_states[n])]).all() else False
            reward = -rewards[i] if not terminated else 10000
            truncated = False

            #if terminated:
            #    print("agent", n, self.base_state, action, observation, reward)

            # Store the transition in memory
            state = torch.tensor(self.one_hot_enc[self.base_state], dtype=torch.float32, device=device).unsqueeze(0)
            action = torch.tensor(action, dtype=torch.int64, device=device).unsqueeze(0)
            next_state = torch.tensor(observation, dtype=torch.float32, device=device)  # .unsqueeze(0)
            reward = torch.tensor([reward], device=device)
            self.done[n] = terminated or truncated

            transitions.append((n, (state, action, next_state, reward)))

            self.step_counter[n] += 1

        next_state = [self.one_hot_enc[tuple(self.S[n])] for n in range(self.n_agents)]

        if np.where((self.S == self.final_states).all(axis=1), False, True).sum() > 0:
            if (self.step_counter >= self.timeout).any():  # step counter is an array
                self.success = np.where((self.S == self.final_states).all(axis=1), 1, 0)
                done = True
                self.base_state = None
                self.agents_at_base_state = None
            else:
                done = False
                non_terminated_agents = np.where((self.S != self.final_states).any(axis=1), True, False)
                first_agents = np.where(self.T == self.T[non_terminated_agents].min(), True,
                                        False) * non_terminated_agents
                uni = np.unique(self.S[first_agents], axis=0)

                self.base_state = tuple(uni[np.random.randint(len(uni))])
                # print(uni, self.base_state, self.final_states[non_terminated_agents])
                self.agents_at_base_state = np.where((self.S == self.base_state).all(axis=1), True, False) * first_agents

                indices = np.argwhere(self.agents_at_base_state == True)
                for n in indices:
                    edge = self.last_agent_transits[int(n)]
                    self.agents_in_transit[edge] -= 1

        else:
            self.success = np.ones(self.n_agents)
            done = True
            self.base_state = None
            self.agents_at_base_state = None

        self.step_counter += 1

        return next_state, self.base_state, self.agents_at_base_state, transitions, done


class roadNetwork:
    def __init__(self, graph, n_agents):
        self.done = None
        self.trajectory = None
        self.actions_taken = None
        self.final_states = None
        self.S = None
        self.T = None
        self.base_state = None
        self.agents_at_base_state = None
        self.G = graph
        self.n_agents = n_agents
        self.n_actions = 2  # max node degree
        self.n_states = len(G.nodes)  # to be defined

        self.one_hot_enc = {
            0: np.array([0, 0, 0, 0, 1]),
            1: np.array([0, 0, 0, 1, 0]),
            2: np.array([0, 0, 1, 0, 0]),
            3: np.array([0, 1, 0, 0, 0]),
            4: np.array([1, 0, 0, 0, 0])
        }
        self.one_hot_enc = {(l, r): np.concatenate(
            [np.array([0 if i != l else 1 for i in range(8)]), np.array([0 if i != r else 1 for i in range(8)])]) for
                       (l, r) in G.nodes()}

    def reset(self):
        self.T = np.zeros(self.n_agents)
        self.S = np.zeros(self.n_agents).astype(int)
        self.final_states = np.array([(i % 2) + 2 for i in range(self.n_agents)])
        self.actions_taken = np.zeros((self.n_states, self.n_actions)).astype(int)
        self.trajectory = []
        self.done = np.zeros(self.n_agents)
        info = {}
        state = [self.one_hot_enc[self.S[n]] for n in range(self.n_agents)]

        remaining_agents = np.where(self.S != self.final_states, True, False)
        first_agents = np.where(self.T == self.T[remaining_agents].min(), True, False) * remaining_agents
        uni = np.unique(self.S[first_agents])
        self.base_state = np.random.choice(uni)
        self.agents_at_base_state = np.where(self.S == self.base_state, True, False) * first_agents

        # print(self.base_state, self.agents_at_base_state)

        return state, info, self.base_state, self.agents_at_base_state

    def step(self, actions):
        edges = [neighbour[1] for neighbour in G.edges(self.base_state)]

        actions = np.clip(actions, 0, len(edges) - 1)
        actions = actions.flatten()
        # print(actions)
        counts = np.bincount(actions, minlength=n_actions)
        self.actions_taken[self.base_state] += counts

        reward_per_action = [G.adj[self.base_state][neighbour[1]]["cost"](counts[i]) for i, neighbour in
                             enumerate(G.edges(self.base_state))]

        rewards = np.array([reward_per_action[a] for a in actions])

        self.S[self.agents_at_base_state] = np.array([edges[a] for a in actions]).astype(int)

        observations = np.array([edges[a] for a in actions]).astype(int)

        # reward = -np.mean(rewards)

        transitions = []

        for i, n in enumerate(np.argwhere(self.agents_at_base_state == True).flatten()):
            # print("in loop")
            driver = drivers[n]
            observation = self.one_hot_enc[observations[i]]
            action = actions[i]
            terminated = True if (observation == self.one_hot_enc[self.final_states[n]]).all() else False
            reward = 100 if terminated else -rewards[i]
            truncated = False

            reward = torch.tensor([reward], device=device)
            self.done[n] = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device)  # .unsqueeze(0)

            # Store the transition in memory
            state = torch.tensor(self.one_hot_enc[self.S[n]], dtype=torch.float32, device=device).unsqueeze(0)
            action = torch.tensor(action, dtype=torch.int64, device=device).unsqueeze(0)

            transitions.append((n, (state, action, next_state, reward)))

        self.T[self.agents_at_base_state] += rewards

        self.trajectory.append(tuple([self.base_state, counts, rewards, edges]))

        next_state = [self.one_hot_enc[self.S[n]] for n in range(self.n_agents)]

        if np.where(self.S == self.final_states, False, True).sum() > 0:
            done = False
            non_terminated_agents = np.where(self.S != self.final_states, True, False)
            first_agents = np.where(self.T == self.T[non_terminated_agents].min(), True,
                                    False) * non_terminated_agents
            uni = np.unique(self.S[first_agents])
            self.base_state = np.random.choice(uni)
            # print(uni, self.base_state, self.final_states[non_terminated_agents])
            self.agents_at_base_state = np.where(self.S == self.base_state, True, False) * first_agents
            # print(self.agents_at_base_state)
            # print(self.S)
        else:
            done = True
            self.base_state = None
            self.agents_at_base_state = None

        return next_state, self.base_state, self.agents_at_base_state, transitions, done
