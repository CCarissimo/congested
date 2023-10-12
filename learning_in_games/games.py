import numpy as np
from dataclasses import dataclass


@dataclass
class gameConfig:
    n_agents: int
    n_actions: int
    n_states: int
    # m: float
    # beta: float


@dataclass
class BraessConfig(gameConfig):
    cost: float


def braess_augmented_network(actions, cost=0):
    n_agents = len(actions)
    n_up = (actions == 0).sum()
    n_down = (actions == 1).sum()
    n_cross = (actions == 2).sum()

    r_0 = 1 + (n_up + n_cross) / n_agents
    r_1 = 1 + (n_down + n_cross) / n_agents
    r_2 = (n_up + n_cross) / n_agents + (n_down + n_cross) / n_agents + cost

    T = np.array([-r_0, -r_1, -r_2])
    R = T[actions]
    S = None
    return R, S


def braess_initial_network(actions):
    n_agents = len(actions)
    n_up = (actions == 0).sum()
    n_down = (actions == 1).sum()

    r_0 = 1 + n_up / n_agents
    r_1 = 1 + n_down / n_agents

    T = np.array([-r_0, -r_1])
    R = T[actions]
    return R, T


def two_route_game(actions, cost=1):
    n_agents = len(actions)
    n_up = (actions == 0).sum()

    r_0 = n_up / n_agents + cost
    r_1 = (1 - n_up / n_agents) + (1 - cost)

    T = np.array([-r_0, -r_1])
    R = T[actions]
    return R, T


def pigou(actions, cost=1):
    n_agents = len(actions)
    n_up = (actions == 0).sum()
    n_down = (actions == 1).sum()
    pct = n_down / n_agents

    r_0 = cost
    r_1 = pct

    T = np.array([-r_0, -r_1])
    R = T[actions]
    return R, T


def pigou3(actions):
    n_agents = len(actions)
    n_up = (actions == 0).sum()

    r_0 = n_up / n_agents
    r_1 = 1
    r_2 = 1

    T = np.array([-r_0, -r_1, -r_2])
    R = T[actions]
    return R, T


@dataclass
class MinorityConfig(gameConfig):
    threshold: float


def minority_game(actions, threshold=0.4):
    n_agents = len(actions)
    n_up = (actions == 0).sum()
    
    if n_agents * threshold >= n_up: # up is minority
        r_0 = 1
        r_1 = 0
    else:
        r_0 = 0
        r_1 = 1
    
    T = np.array([r_0, r_1])
    R = T[actions]
    return R, T


def minority_game_2(actions, threshold=0.4):
    n_agents = len(actions)
    n_a = (actions == 0).sum()
    fraction_a = n_a/n_agents
    fraction_b = 1 - fraction_a
    
    r_a = 1 - 2*fraction_a
    r_b = 1 - 2*fraction_b
    
    T = np.array([r_a, r_b])
    R = T[actions]
    return R, T


def el_farol_bar(actions):
    n_agents = len(actions)
    n_home = (actions == 0).sum()
    n_bar = (actions == 1).sum()
    pct = n_bar / n_agents

    r_0 = 1
    r_1 = 2 - 4 * pct if (pct < 0) else 4 * pct - 2

    T = np.array([-r_0, -r_1])
    R = T[actions]
    return R, T
