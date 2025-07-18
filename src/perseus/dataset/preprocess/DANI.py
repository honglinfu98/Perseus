"""
This algorithm is used to create netwrok of the channeles and based on the paper below:
DANI: A Fast Diffusion Aware Network Inference Algorithm
"""

from collections import defaultdict
import numpy as np
import networkx as nx


def sort_cascade(cascade: list) -> list:
    """
    This function is used to sort the cascade based on the infected time
    Return: sorted cascade
    """
    node_id = np.array(cascade[0])
    infected_time = np.array(cascade[1])
    index = np.argsort(infected_time)
    sorted_node_id = node_id[index].tolist()
    sorted_infected_time = infected_time[index].tolist()
    return [sorted_node_id, sorted_infected_time]


def trans_list(cascade_dict: dict) -> list:
    """
    This function is used to transform the cascade from dictionary to list
    Return: cascade in list format
    """
    cascade = []
    nodes = []
    times = []

    for node in cascade_dict.keys():
        if node != "T":
            nodes.append(int(node))
            times.append(cascade_dict[node])

    cascade.append(nodes)
    cascade.append(times)

    return cascade


def get_index(cascade: list) -> dict:
    """
    This function is used to get the index of the nodes in the cascade
    Return: index of the nodes in the cascade
    """
    S = sort_cascade(cascade)
    CV = {}
    nodes = S[0]
    for i, node in enumerate(nodes):
        CV[node] = i + 1

    return CV


def DANI(N: int, cascades: list) -> tuple:
    """
    This script is used to process the cascades and return the influence graph
    Return: Influence graph, edge list sorted, , P_dict
    """
    P = np.zeros([N, N])

    for cascade_dict in cascades:
        D = np.zeros([N, N])
        cascade = trans_list(cascade_dict)
        S = sort_cascade(cascade)
        CV = get_index(cascade)
        nodes = S[0]

        for i in range(len(nodes)):
            u = nodes[i]
            for j in range(i + 1, len(nodes)):
                v = nodes[j]
                D[u][v] = 1 / (CV[v] * (CV[v] - CV[u]))

        for i in range(N):
            D[i] = D[i] / np.sum(D[i])
            for j in range(N):
                if D[i][j] > 0:
                    P[(i, j)] += D[i][j]

    P_dict = defaultdict(float)
    for i in range(N):
        P[i] = P[i] / np.sum(P[i])
        for j in range(N):
            if i == j:
                continue
            else:
                if P[i][j] > 0:
                    P_dict[(i, j)] = P[i][j]

    cascade_buffer = dict()
    for c_i, cascade_dict in enumerate(cascades):
        cascade_buffer[c_i] = list(cascade_dict.keys())[:-1]

    theta = {}
    for u in range(N):
        for v in range(N):
            if u != v:
                count_uv = (
                    0  # Number of cascades with u and v where index(u) < index(v)
                )
                count_u_or_v = 0  # Number of cascades with at least u or v

                for c_i, nodes in cascade_buffer.items():
                    # Check if this cascade contains u or v
                    contains_u = u in nodes
                    contains_v = v in nodes

                    if contains_u or contains_v:
                        count_u_or_v += 1

                    # If it contains both, check their positions
                    if contains_u and contains_v:
                        if nodes.index(u) < nodes.index(v):
                            count_uv += 1

                # Compute theta_{u,v}
                if count_u_or_v > 0:
                    theta[(u, v)] = count_uv / count_u_or_v
                else:
                    # If neither u nor v ever appears, we can define theta_{u,v} = 0 or skip it
                    theta[(u, v)] = 0.0

    P_theta = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            else:
                P_theta[(i, j)] = P_dict[(i, j)] * theta[(i, j)]

    IG = nx.DiGraph()

    for i, c in P_theta.items():
        if c != 0:
            u = i[0]
            v = i[1]
            if P_theta[(u, v)] > P_theta[(v, u)]:
                IG.add_edge(u, v)
            else:
                IG.add_edge(v, u)

    return IG, P_dict, P_theta


if __name__ == "__main__":
    N1 = 5

    e1 = [
        {1: 0.0, 2: 1.0, 3: 2.0, 4: 195.0, "T": 195.0},
        {1: 0.0, 2: 39.0, "T": 39.0},
        {3: 0.0, 2: 1.0, "T": 1.0},
        {4: 0.0, 1: 1.0, "T": 1.0},
    ]

    IG, P_dict, P_theta = DANI(N1, e1)

    print(IG.edges())

    print(P_dict)

    N2 = 13

    e2 = [
        {
            8: 0.0,
            0: 0.0,
            5: 0.0,
            1: 195.0,
            7: 2293.0,
            6: 2293.0,
            4: 2293.0,
            "T": 2293.0,
        },
        {10: 0.0, 3: 39.0, 2: 39.0, 11: 108.0, "T": 108.0},
        {12: 0.0, 9: 1.0, "T": 1.0},
    ]

    IG, P_dict, P_theta = DANI(N2, e2)

    print(IG.edges())

    print(P_dict)
