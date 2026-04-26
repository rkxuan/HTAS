import torch
import random
import shortuuid
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from hats.topology.connection import Connection, Decentralized_Connection, Sequential_Connection
from hats.utils.unionfind import UnionFind


class Topology_Distribution(torch.nn.Module):
    def __init__(
        self,
        potential_connections: List[Tuple],  # record the list of node (agent) pairs
        decision_operation_id: str,  # record the id of the final decision operation
    ):
        super().__init__()
        self.potential_connections = potential_connections
        self.decision_operation_id = decision_operation_id

    def calculate_hybrid_longest_path(self, adjacency: Dict[str, List[str]], uf: UnionFind) -> torch.Tensor:
        if not adjacency:
            return torch.tensor(0.0)
        
        all_nodes = set(adjacency.keys())
        for neighbors in adjacency.values():
            all_nodes.update(neighbors)

        in_degree = {node: 0 for node in all_nodes}
        for neighbors in adjacency.values():
            for neighbor in neighbors:
                in_degree[neighbor] += 1

        start_nodes = [node for node in all_nodes if in_degree[node] == 0]
        longest_path = 0

        for start in start_nodes:
            if uf.contain(start):
                queue = [(start, 1)]  
            else:
                queue = [(start, 0)]  
            visited = {start: 0}

            while queue:
                node, path_len = queue.pop(0)
                longest_path = max(longest_path, path_len)

                if node in adjacency:
                    for neighbor in adjacency[node]:
                        new_len = path_len + 2 if uf.contain(neighbor) else path_len + 1
                        if neighbor not in visited or visited[neighbor] < new_len:
                            visited[neighbor] = new_len
                            queue.append((neighbor, new_len))

        return torch.tensor(float(longest_path))

    def calculate_longest_path(self, adjacency: Dict[str, List[str]]) -> torch.Tensor:
        """
        Calculate the longest path (number of steps) in a DAG.
        This represents the execution steps of the sequential topology.
        """
        if not adjacency:
            return torch.tensor(0.0)

        # Find all nodes
        all_nodes = set(adjacency.keys())
        for neighbors in adjacency.values():
            all_nodes.update(neighbors)

        # Calculate in-degree for each node
        in_degree = {node: 0 for node in all_nodes}
        for neighbors in adjacency.values():
            for neighbor in neighbors:
                in_degree[neighbor] += 1

        # Find starting nodes (in-degree = 0)
        start_nodes = [node for node in all_nodes if in_degree[node] == 0]

        # BFS to calculate longest path from each start node
        longest_path = 0
        for start in start_nodes:
            queue = [(start, 0)]  # (node, current_path_length)
            visited = {start: 0}

            while queue:
                node, path_len = queue.pop(0)
                longest_path = max(longest_path, path_len)

                if node in adjacency:
                    for neighbor in adjacency[node]:
                        new_len = path_len + 1
                        if neighbor not in visited or visited[neighbor] < new_len:
                            visited[neighbor] = new_len
                            queue.append((neighbor, new_len))

        return torch.tensor(float(longest_path))

    def check_cycle(
        self, 
        new_connection: Sequential_Connection, 
        adjacency: Dict[str, List[str]]
    ) -> bool:

        # simulation of adding the new connection to the adjacency list
        if new_connection.src not in adjacency:
            adjacency[new_connection.src] = []
        adjacency[new_connection.src].append(new_connection.dst)

        # Check for cycle using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            if node in adjacency:
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        if has_cycle(neighbor, visited, rec_stack):
                            return True
                    elif neighbor in rec_stack:
                        return True

            rec_stack.remove(node)
            return False

        # check
        has_cycle_flag = False
        for node in adjacency:
            if node not in visited:
                if has_cycle(node, visited, rec_stack):
                    has_cycle_flag = True
                    break

        # remove
        adjacency[new_connection.src].pop()
        if not adjacency[new_connection.src]:
            del adjacency[new_connection.src]

        return has_cycle_flag

    def check_cycle_(
        self, 
        adjacency: Dict[str, List[str]]
    ) -> bool:
        # Check for cycle using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            if node in adjacency:
                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        if has_cycle(neighbor, visited, rec_stack):
                            return True
                    elif neighbor in rec_stack:
                        return True

            rec_stack.remove(node)
            return False

        has_cycle_flag = False
        for node in adjacency:
            if node not in visited:
                if has_cycle(node, visited, rec_stack):
                    has_cycle_flag = True
                    break

        return has_cycle_flag

class Sequential_Topology_Distribution(Topology_Distribution):
    def __init__(
        self,
        potential_connections: List[Tuple],  # record the list of node (agent) pairs
        decision_operation_id: str,  # record the id of the final decision operation
        initial_probability: float = 0.5,  # initialization of the probability of a connection
    ):
        super().__init__(potential_connections, decision_operation_id)
        init_logit = torch.log(torch.tensor(initial_probability / (1 - initial_probability)))
        init_tensor = torch.ones(
            len(potential_connections),
            requires_grad=True) * init_logit
        self.edge_logits = torch.nn.Parameter(init_tensor)  # probs = sigmoid(logits)

    def realize_direct(self) -> List[Sequential_Connection]:
        edgelist = []
        for potential_connection in self.potential_connections:
            if potential_connection[1] == self.decision_operation_id:
                sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                edgelist.append(sequential_connection)
        return edgelist

    def realize_random(self) -> List[Sequential_Connection]:
        edgelist = []
        adjacency = defaultdict(list)  # for check_cycle
        reverse_adjacency = defaultdict(list)  # for reversed BFS from self.decision_agent

        for potential_connection in self.potential_connections:
            if potential_connection[1] == self.decision_operation_id:
                continue
            if random.randint(0, 1) > 0.5:
                sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                if not self.check_cycle(sequential_connection, adjacency):
                    edgelist.append(sequential_connection)
                    adjacency[sequential_connection.src].append(sequential_connection.dst)
                    reverse_adjacency[sequential_connection.dst].append(sequential_connection.src)

        while not any(e.dst == self.decision_operation_id for e in edgelist):  # to avoid
            for potential_connection in self.potential_connections:
                if potential_connection[1] == self.decision_operation_id:
                    if random.randint(0, 1) > 0.5:
                        sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                        edgelist.append(sequential_connection)
                        reverse_adjacency[sequential_connection.dst].append(sequential_connection.src)
        
        visited = set()
        visited.add(self.decision_operation_id)
        current_agents = [self.decision_operation_id]
        while current_agents:
            next_agents = []
            for current_agent in current_agents:
                for predecessor in reverse_adjacency[current_agent]:
                    if predecessor not in visited:
                        visited.add(predecessor)
                        next_agents.append(predecessor)
            current_agents = next_agents

        new_edgelist = []
        for edge in edgelist:
            if edge.dst in visited:
                new_edgelist.append(edge)
        edgelist = new_edgelist
        return edgelist

    def realize_full(self) -> List[Sequential_Connection]: # not strict full connection
        edgelist = []
        adjacency = defaultdict(list)  # for check_cycle
        random.shuffle(self.potential_connections)

        for potential_connection in self.potential_connections:
            if potential_connection[1] == self.decision_operation_id:
                continue
            sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
            if not self.check_cycle(sequential_connection, adjacency):
                edgelist.append(sequential_connection)
                adjacency[sequential_connection.src].append(sequential_connection.dst)

        for potential_connection in self.potential_connections:
            if potential_connection[1] == self.decision_operation_id:  # the addition to decision operation is directional, which will not lead to cycles
                sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                edgelist.append(sequential_connection)
        return edgelist

    def realize_chain(self) -> List[Sequential_Connection]:
        edgelist = []
        nodelist = []
        for (node1, node2) in self.potential_connections:
            if node1 != self.decision_operation_id and node1 not in nodelist:
                nodelist.append(node1)
            if node2 != self.decision_operation_id and node2 not in nodelist:
                nodelist.append(node2)

        random.shuffle(nodelist)

        for i in range(0, len(nodelist) - 1):
            sequential_connection = Sequential_Connection(nodelist[i], nodelist[i + 1])
            edgelist.append(sequential_connection)

        while not any(e.dst == self.decision_operation_id for e in edgelist):  # FIXED: check e.dst instead of e.src
            for potential_connection in self.potential_connections:
                if potential_connection[1] == self.decision_operation_id:
                    if random.randint(0, 1) > 0.5:
                        sequential_connection = Sequential_Connection(potential_connection[0],
                                                                         potential_connection[1])
                        edgelist.append(sequential_connection)

        return edgelist

    def realize_star(self) -> List[Sequential_Connection]:
        edgelist = []
        nodelist = []
        for (node1, node2) in self.potential_connections:
            if node1 != self.decision_operation_id and node1 not in nodelist:
                nodelist.append(node1)
            if node2 != self.decision_operation_id and node2 not in nodelist:
                nodelist.append(node2)
        random.shuffle(nodelist)
        center_node = nodelist[0]

        for potential_connection in self.potential_connections:
            if potential_connection[1] == center_node:
                sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                edgelist.append(sequential_connection)

        sequential_connection = Sequential_Connection(center_node, self.decision_operation_id)
        edgelist.append(sequential_connection)
        return edgelist

    def realize_tree(self) -> List[Sequential_Connection]:
        random.shuffle(self.potential_connections)
        edgelist = []
        adjacency = defaultdict(list)  # for check_cycle

        root = None
        root = self.potential_connections[0][0]

        nodelist = {root: 1}
        queue = [root]

        while queue:
            current_root = queue.pop(0)
            for potential_connection in self.potential_connections:
                if potential_connection[1] == self.decision_operation_id:
                    continue
                if potential_connection[0] == current_root and potential_connection[1] not in nodelist:
                    sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                    if not self.check_cycle(sequential_connection, adjacency):
                        edgelist.append(sequential_connection)
                        # Update adjacency
                        adjacency[sequential_connection.src].append(sequential_connection.dst)
                        queue.append(potential_connection[1])
                        nodelist[potential_connection[1]] = 1

        while not any(e.dst == self.decision_operation_id for e in edgelist):  # to avoid no connection to decision operation
            for potential_connection in self.potential_connections:
                if potential_connection[1] == self.decision_operation_id:
                    if random.randint(0, 1) > 0.5:
                        sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                        edgelist.append(sequential_connection)

        return edgelist

    def realize_para(
        self,
        temperature: float = 1.0,  # must be >= 1.0
    ) -> Tuple[List[Sequential_Connection], torch.Tensor, torch.Tensor, torch.Tensor]:

        log_probs = []
        probs = []
        edgelist = []
        adjacency = defaultdict(list)  # for check_cycle
        reverse_adjacency = defaultdict(list)  # for reversed BFS from self.decision_agent 

        for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
            if potential_connection[1] == self.decision_operation_id:
                continue
            sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
            edge_prob = torch.sigmoid(edge_logit / temperature)
            probs.append(edge_prob)
            if not self.check_cycle(sequential_connection, adjacency):  # Check if adding this connection would form a cycle
                if torch.rand(1) < edge_prob:
                    edgelist.append(sequential_connection)
                    log_probs.append(torch.log(edge_prob))
                    adjacency[sequential_connection.src].append(sequential_connection.dst)
                    reverse_adjacency[sequential_connection.dst].append(sequential_connection.src)
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    pass
            else:  # If cycle would be formed, treat as not selected
                #log_probs.append(torch.log(1 - edge_prob))
                log_probs.append(torch.log(torch.tensor(1.0)))
                

        decision_log_probs = []
        decision_probs = []
        nloops = 0
        while not any(e.dst == self.decision_operation_id for e in edgelist):
            assert nloops < 1000, "tend to have an infinite loop"
            decision_log_probs = []  # reset it to avoid repetition
            decision_probs = []  # reset it to avoid repetition
            for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
                if potential_connection[1] == self.decision_operation_id:
                    edge_prob = torch.sigmoid(edge_logit / temperature)
                    decision_probs.append(edge_prob)
                    if torch.rand(1) < edge_prob:
                        sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                        edgelist.append(sequential_connection)
                        decision_log_probs.append(torch.log(edge_prob))
                        reverse_adjacency[sequential_connection.dst].append(sequential_connection.src)
                    else:
                        decision_log_probs.append(torch.log(1 - edge_prob))
            nloops = nloops + 1

        visited = set()
        visited.add(self.decision_operation_id)
        current_agents = [self.decision_operation_id]
        while current_agents:
            next_agents = []
            for current_agent in current_agents:
                for predecessor in reverse_adjacency[current_agent]:
                    if predecessor not in visited:
                        visited.add(predecessor)
                        next_agents.append(predecessor)
            current_agents = next_agents
        
        for i, (potential_connection, edge_logit) in enumerate(zip(self.potential_connections, self.edge_logits)):
            if potential_connection[1] == self.decision_operation_id:
                continue
            if potential_connection[1] not in visited:
                edge_prob = torch.sigmoid(edge_logit / temperature)
                #log_probs[i] = torch.log(1 - edge_prob)
                log_probs[i] = torch.log(torch.tensor(1.0))

        new_edgelist = []
        for edge in edgelist:
            if edge.dst in visited:
                new_edgelist.append(edge)
        edgelist = new_edgelist
    
        probs.extend(decision_probs)
        probs = torch.stack(probs)
        sparse = torch.norm(probs, p=1)

        log_probs.extend(decision_log_probs)
        log_prob = torch.sum(torch.stack(log_probs))
        # Calculate nsteps: the longest path (execution steps) in the sequential topology
        nsteps = self.calculate_longest_path(adjacency) # # range from [0, n-1], where n is the number of agents

        return edgelist, log_prob, sparse, nsteps

    def realize(
        self,
        temperature: float = 1.0,  # must be >= 1.0
        threshold: float = 0.5  # must be >=0 and <=1
    ) -> List[Sequential_Connection]:
        edgelist = []
        adjacency = defaultdict(list)  # for check_cycle
        reverse_adjacency = defaultdict(list)  # for reversed BFS from self.decision_agent 

        for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if edge_prob > threshold:
                sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                if not self.check_cycle(sequential_connection, adjacency):
                    edgelist.append(sequential_connection)
                    adjacency[sequential_connection.src].append(sequential_connection.dst)
                    reverse_adjacency[sequential_connection.dst].append(sequential_connection.src)

        while not any(e.dst == self.decision_operation_id for e in edgelist):  # to avoid
            for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
                if potential_connection[1] == self.decision_operation_id:
                    edge_prob = torch.sigmoid(edge_logit / temperature)
                    if edge_prob > random.random():
                        sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                        edgelist.append(sequential_connection)
                        reverse_adjacency[sequential_connection.dst].append(sequential_connection.src)
        
        visited = set()
        visited.add(self.decision_operation_id)
        current_agents = [self.decision_operation_id]
        while current_agents:
            next_agents = []
            for current_agent in current_agents:
                for predecessor in reverse_adjacency[current_agent]:
                    if predecessor not in visited:
                        visited.add(predecessor)
                        next_agents.append(predecessor)
            current_agents = next_agents

        new_edgelist = []
        for edge in edgelist:
            if edge.dst in visited:
                new_edgelist.append(edge)
        edgelist = new_edgelist
        return edgelist

class Decentralized_Topology_Distribution(Topology_Distribution):
    def __init__(
        self,
        potential_connections: List[Tuple],  # record the list of node (agent) pairs
        decision_operation_id: str,  # record the id of the final decision operation
        initial_probability: float = 0.5,  # initialization of the probability of a connection
    ):
        super().__init__(potential_connections, decision_operation_id)
        init_logit = torch.log(torch.tensor(initial_probability / (1 - initial_probability)))
        init_tensor = torch.ones(
            len(potential_connections),
            requires_grad=True) * init_logit
        self.edge_logits = torch.nn.Parameter(init_tensor)
    
    def realize_random(self) -> List[Sequential_Connection]:
        edgelist = []
        for potential_connection in self.potential_connections:
            if potential_connection[1] == self.decision_operation_id:
                continue
            if random.randint(0, 1) > 0.5:
                decentralized_connection = Decentralized_Connection(potential_connection[0], potential_connection[1])
                edgelist.append(decentralized_connection)

        while not any(e.dst == self.decision_operation_id for e in edgelist):  # to avoid
            for potential_connection in self.potential_connections:
                if potential_connection[1] == self.decision_operation_id:
                    if random.randint(0, 1) > 0.5:
                        sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                        edgelist.append(sequential_connection)
        return edgelist

    def realize_direct(self) -> List[Sequential_Connection]:
        edgelist = []
        for potential_connection in self.potential_connections:
            if potential_connection[1] == self.decision_operation_id:
                sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                edgelist.append(sequential_connection)
        return edgelist

    def realize_full(self) -> List[Decentralized_Connection]:
        edgelist = []
        for potential_connection in self.potential_connections:
            decentralized_connection = Decentralized_Connection(potential_connection[0], potential_connection[1])
            edgelist.append(decentralized_connection)

        for i, edge in enumerate(edgelist):
            if edge.dst == self.decision_operation_id:
                edgelist[i] = Sequential_Connection(edge.src, edge.dst)
        return edgelist

    def realize_chain(self) -> List[Decentralized_Connection]:
        edgelist = []
        nodelist = []
        for node1, node2 in self.potential_connections:
            if node1 != self.decision_operation_id and node1 not in nodelist:
                nodelist.append(node1)
            if node2 != self.decision_operation_id and node2 not in nodelist:
                nodelist.append(node2)

        random.shuffle(nodelist)

        for i in range(0, len(nodelist) - 1):
            decentralized_connection = Decentralized_Connection(nodelist[i], nodelist[i + 1])
            edgelist.append(decentralized_connection)

        while not any(e.dst == self.decision_operation_id for e in edgelist):
            for potential_connection in self.potential_connections:
                if potential_connection[1] == self.decision_operation_id:
                    if random.randint(0, 1) > 0.5:
                        sequential_connection = Sequential_Connection(potential_connection[0],
                                                                         potential_connection[1])
                        edgelist.append(sequential_connection)
        return edgelist

    def realize_tree(self) -> List[Decentralized_Connection]:
        random.shuffle(self.potential_connections)
        edgelist = []
        adjacency = defaultdict(list)  # for check_cycle

        root = self.potential_connections[0][0]

        nodelist = {root: 1}
        queue = [root]

        while queue:
            current_root = queue.pop(0)
            for potential_connection in self.potential_connections:
                if potential_connection[1] == self.decision_operation_id:
                    continue
                if potential_connection[0] == current_root and potential_connection[1] not in nodelist:
                    decentralized_connection = Decentralized_Connection(potential_connection[0],
                                                                        potential_connection[1])
                    edgelist.append(decentralized_connection)
                    # Update adjacency
                    adjacency[decentralized_connection.src].append(decentralized_connection.dst)
                    queue.append(potential_connection[1])
                    nodelist[potential_connection[1]] = 1

        # Add edges to decision_operation_id with 50% probability
        for potential_connection in self.potential_connections:
            if potential_connection[1] == self.decision_operation_id:
                if random.random() > 0.5:  # 50% probability
                    new_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                    edgelist.append(new_connection)

        return edgelist

    def realize_star(self) -> List[Decentralized_Connection]:
        edgelist = []
        nodelist = []
        for (node1, node2) in self.potential_connections:
            if node1 != self.decision_operation_id and node1 not in nodelist:
                nodelist.append(node1)
            if node2 != self.decision_operation_id and node2 not in nodelist:
                nodelist.append(node2)
        random.shuffle(nodelist)
        center_node = nodelist[0]

        for potential_connection in self.potential_connections:
            if potential_connection[1] == center_node:
                decentralized_connection = Decentralized_Connection(potential_connection[0], potential_connection[1])
                edgelist.append(decentralized_connection)

        sequential_connection = Sequential_Connection(center_node, self.decision_operation_id)
        edgelist.append(sequential_connection)
        return edgelist

    def realize_cycle(self) -> List[Decentralized_Connection]:
        edgelist = []
        random.shuffle(self.potential_connections)

        nodelist = []
        for node1, node2 in self.potential_connections:
            if node1 != self.decision_operation_id and node1 not in nodelist:
                nodelist.append(node1)
            if node2 != self.decision_operation_id and node2 not in nodelist:
                nodelist.append(node2)

        random.shuffle(nodelist)
        nnodes = len(nodelist)
        for i in range(nnodes):
            if i < nnodes - 1:
                decentralized_connection = Decentralized_Connection(nodelist[i], nodelist[i + 1])
                edgelist.append(decentralized_connection)
            else:
                decentralized_connection = Decentralized_Connection(nodelist[-1], nodelist[0])
                edgelist.append(decentralized_connection)

        while not any(e.dst == self.decision_operation_id for e in edgelist):
            for potential_connection in self.potential_connections:
                if potential_connection[1] == self.decision_operation_id:
                    if random.randint(0, 1) > 0.5:
                        sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                        edgelist.append(sequential_connection)
        return edgelist

    def realize_bicycle(self) -> List[Decentralized_Connection]:
        edgelist = []
        random.shuffle(self.potential_connections)

        nodelist = []
        for node1, node2 in self.potential_connections:
            if node1 != self.decision_operation_id and node1 not in nodelist:
                nodelist.append(node1)
            if node2 != self.decision_operation_id and node2 not in nodelist:
                nodelist.append(node2)

        random.shuffle(nodelist)
        nnodes = len(nodelist)
        for i in range(nnodes):
            if i < nnodes - 1:
                decentralized_connection = Decentralized_Connection(nodelist[i], nodelist[i + 1])
                edgelist.append(decentralized_connection)
                decentralized_connection = Decentralized_Connection(nodelist[i + 1], nodelist[i])
                edgelist.append(decentralized_connection)
            else:
                decentralized_connection = Decentralized_Connection(nodelist[0], nodelist[-1])
                edgelist.append(decentralized_connection)
                decentralized_connection = Decentralized_Connection(nodelist[-1], nodelist[0])
                edgelist.append(decentralized_connection)

        while not any(e.dst == self.decision_operation_id for e in edgelist):
            for potential_connection in self.potential_connections:
                if potential_connection[1] == self.decision_operation_id:
                    if random.randint(0, 1) > 0.5:
                        sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                        edgelist.append(sequential_connection)
        return edgelist

    def realize_para(
        self,
        temperature: float = 1.0,  # must be >= 1.0
    ) -> Tuple[List[Decentralized_Connection], torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = []
        probs = []
        edgelist = []
        reverse_adjacency = defaultdict(list)  # for reversed BFS from self.decision_agent

        for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
            if potential_connection[1] == self.decision_operation_id:
                continue
            decentralized_connection = Decentralized_Connection(potential_connection[0], potential_connection[1])
            edge_prob = torch.sigmoid(edge_logit / temperature)
            probs.append(edge_prob)
            if torch.rand(1) < edge_prob:
                edgelist.append(decentralized_connection)
                log_probs.append(torch.log(edge_prob))
                reverse_adjacency[decentralized_connection.dst].append(decentralized_connection.src)
            else:
                log_probs.append(torch.log(1 - edge_prob))

        decision_log_probs = []
        decision_probs = []
        nloops = 0
        while not any(e.dst == self.decision_operation_id for e in edgelist):
            assert nloops < 1000, "tend to have an infinite loop"
            decision_log_probs = []  # reset it to avoid repetition
            decision_probs = []  # reset it to avoid repetition
            for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
                if potential_connection[1] == self.decision_operation_id:
                    edge_prob = torch.sigmoid(edge_logit / temperature)
                    decision_probs.append(edge_prob)
                    if torch.rand(1) < edge_prob:
                        sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                        edgelist.append(sequential_connection)
                        reverse_adjacency[sequential_connection.dst].append(sequential_connection.src)
                        decision_log_probs.append(torch.log(edge_prob))
                    else:
                        decision_log_probs.append(torch.log(1 - edge_prob))
            nloops = nloops + 1

        visited = set()
        visited.add(self.decision_operation_id)
        current_agents = [self.decision_operation_id]
        while current_agents:
            next_agents = []
            for current_agent in current_agents:
                for predecessor in reverse_adjacency[current_agent]:
                    if predecessor not in visited:
                        visited.add(predecessor)
                        next_agents.append(predecessor)
            current_agents = next_agents

        for i, (potential_connection, edge_logit) in enumerate(zip(self.potential_connections, self.edge_logits)):
            if potential_connection[1] == self.decision_operation_id:
                continue
            if potential_connection[1] not in visited:
                edge_prob = torch.sigmoid(edge_logit / temperature)
                log_probs[i] = torch.log(1 - edge_prob)

        new_edgelist = []
        for edge in edgelist:
            if edge.dst in visited:
                new_edgelist.append(edge)
        edgelist = new_edgelist

        probs.extend(decision_probs)
        probs = torch.stack(probs)
        sparse = torch.norm(probs, p=1)

        log_probs.extend(decision_log_probs)
        log_prob = torch.sum(torch.stack(log_probs))

        nsteps = torch.tensor(0.0)  # the steps of execuation can not be optimized for a decentralized graph
        return edgelist, log_prob, sparse, nsteps

    def realize(
        self,
        temperature: float = 1.0,  # must be >= 1.0
        threshold: float = 0.5  # must be >=0 and <=1
    ) -> List[Decentralized_Connection]:
        edgelist = []
        reverse_adjacency = defaultdict(list)
        for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
            if potential_connection[1] == self.decision_operation_id:
                continue
            decentralized_connection = Decentralized_Connection(potential_connection[0], potential_connection[1])
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if edge_prob > threshold:
                edgelist.append(decentralized_connection)
                reverse_adjacency[decentralized_connection.dst].append(decentralized_connection.src)

        while not any(e.dst == self.decision_operation_id for e in edgelist):  # to avoid
            for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
                if potential_connection[1] == self.decision_operation_id:
                    edge_prob = torch.sigmoid(edge_logit / temperature)
                    if edge_prob > random.random():
                        sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                        edgelist.append(sequential_connection)
                        reverse_adjacency[sequential_connection.dst].append(sequential_connection.src)

        visited = set()
        visited.add(self.decision_operation_id)
        current_agents = [self.decision_operation_id]
        while current_agents:
            next_agents = []
            for current_agent in current_agents:
                for predecessor in reverse_adjacency[current_agent]:
                    if predecessor not in visited:
                        visited.add(predecessor)
                        next_agents.append(predecessor)
            current_agents = next_agents

        new_edgelist = []
        for edge in edgelist:
            if edge.dst in visited: # if dst in visited, src must in visited
                new_edgelist.append(edge)
        edgelist = new_edgelist
        return edgelist

class Hyriad_Topology_Distribution(Topology_Distribution):
    def __init__(
        self,
        potential_connections: List[Tuple],  # record the list of node (agent) pairs
        decision_operation_id: str,  # record the id of the final decision operation
        initial_probability: float = 0.5,  # initialization of the probability of a connection
    ):
        super().__init__(potential_connections, decision_operation_id)
        init_logit = torch.log(torch.tensor(initial_probability / (1 - initial_probability)))
        init_tensor = torch.ones(
            [len(potential_connections), 3],
            requires_grad=True) * init_logit  # [no_connection, sequential_connection, decentred]
        self.edge_logits = torch.nn.Parameter(init_tensor)

    def realize_direct(self) -> List[Sequential_Connection]:
        edgelist = []
        for potential_connection in self.potential_connections:
            if potential_connection[1] == self.decision_operation_id:
                sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                edgelist.append(sequential_connection)
        return edgelist

    def realize_random(self) -> List[Connection]:
        edgelist = []
        adjacency = defaultdict(list)  # only record the sequential connections to avoid cycles
        reverse_adjacency = defaultdict(list)  # only record the sequential connections to avoid cycles
        reverse_connections = defaultdict(list) # for BFS from self.decision_agent to filter out redundant connections
        uf = UnionFind()  # to query the decentralized connected components

        for potential_connection in self.potential_connections:
            if potential_connection[1] == self.decision_operation_id:
                continue

            sampled_index = random.randint(0, 2)
            
            if sampled_index == 0:  
                continue
            
            elif sampled_index == 1:  
                sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                members = uf.get_component(potential_connection[1])
                new_adjacency = deepcopy(adjacency)
                new_reverse_adjacency = deepcopy(reverse_adjacency)
                for member in members:
                    new_adjacency[potential_connection[0]].append(member)
                    new_reverse_adjacency[member].append(potential_connection[0])

                if not self.check_cycle_(new_adjacency):  # Check if adding this connection would form a cycle
                    edgelist.append(sequential_connection)
                    adjacency = new_adjacency
                    reverse_adjacency = new_reverse_adjacency
                    reverse_connections[sequential_connection.dst].append(sequential_connection.src)

            else:  # decentralized_edge
                decentralized_connection = Decentralized_Connection(potential_connection[0], potential_connection[1])
                uf_ = deepcopy(uf)
                uf_.union(potential_connection[0], potential_connection[1])
                members = uf_.get_component(potential_connection[0])
                predecessors = set().union(*(reverse_adjacency.get(member, []) for member in
                                             members))  # find all sequential connections to members

                # for a decentralized component, the members will be execuated at the same level
                # from the topological perspective, they must wait for the same preceding node to be executed.
                new_adjacency = deepcopy(adjacency)
                new_reverse_adjacency = deepcopy(reverse_adjacency)
                #updating_sequential_connections = []
                for predecessor in predecessors:
                    for member in members:
                        if member not in new_adjacency[predecessor]:  # the addition of sequential connections to this decentralized component
                            new_adjacency[predecessor].append(member)
                            new_reverse_adjacency[member].append(predecessor)
                            #updating_sequential_connections.append((predecessor, member))

                if not self.check_cycle_(new_adjacency):  # the addition of decentralized_connection is allowed
                    uf = uf_
                    edgelist.append(decentralized_connection)
                    adjacency = new_adjacency
                    reverse_adjacency = new_reverse_adjacency
                    reverse_connections[decentralized_connection.dst].append(decentralized_connection.src)  
                    #for (src, dst) in updating_sequential_connections:
                    #    sequential_connection = Sequential_Connection(src, dst)
                    #    edgelist.append(sequential_connection)

        while not any(e.dst == self.decision_operation_id for e in edgelist):  # to avoid
            for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
                if potential_connection[1] == self.decision_operation_id:
                    sampled_index = random.randint(0, 2)
                    if sampled_index == 1:  
                        sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                        edgelist.append(sequential_connection)
                        reverse_connections[sequential_connection.dst].append(sequential_connection.src)
        
        visited = set()
        visited.add(self.decision_operation_id)
        current_agents = [self.decision_operation_id]
        while current_agents:
            next_agents = []
            for current_agent in current_agents:
                for predecessor in reverse_connections[current_agent]:
                    if predecessor not in visited:
                        visited.add(predecessor)
                        next_agents.append(predecessor)
            current_agents = next_agents
        
        new_edgelist = []
        for edge in edgelist:
            if edge.dst in visited:
                new_edgelist.append(edge)
        edgelist = new_edgelist
        return edgelist

    def realize_para(
        self,
        temperature: float = 1.0,  # must be >= 1.0
    ) -> Tuple[List[Connection], torch.Tensor, torch.Tensor, torch.Tensor]:

        log_probs = []
        probs = []
        edgelist = []
        adjacency = defaultdict(list)  # only record the sequential connections to avoid cycles
        reverse_adjacency = defaultdict(list)  # only record the sequential connections to avoid cycles
        reverse_connections = defaultdict(list) # for BFS from self.decision_agent to filter out redundant connections
        uf = UnionFind()  # to query the decentralized connected components

        for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
            if potential_connection[1] == self.decision_operation_id:
                continue

            # Gumbel-Softmax sampling for three connection types
            edge_prob = F.gumbel_softmax(edge_logit, tau=temperature, hard=False)
            sampled_index = edge_prob.argmax(dim=0)
            probs.append(edge_prob[1:]) # the normalization for the edges

            if sampled_index.item() == 0:  
                log_probs.append(torch.log(edge_prob[sampled_index]))

            elif sampled_index.item() == 1: 
                sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                members = uf.get_component(potential_connection[1])
                new_adjacency = deepcopy(adjacency)
                new_reverse_adjacency = deepcopy(reverse_adjacency)
                for member in members:
                    new_adjacency[potential_connection[0]].append(member)
                    new_reverse_adjacency[member].append(potential_connection[0])

                if not self.check_cycle_(new_adjacency):  # Check if adding this connection would form a cycle
                    edgelist.append(sequential_connection)
                    log_probs.append(torch.log(edge_prob[sampled_index]))
                    adjacency = new_adjacency
                    reverse_adjacency = new_reverse_adjacency
                    reverse_connections[sequential_connection.dst].append(sequential_connection.src)
                else:
                    #log_probs.append(torch.log(edge_prob[0]))
                    log_probs.append(torch.log(torch.tensor(1.0)))

            else:  # decentralized_edge
                decentralized_connection = Decentralized_Connection(potential_connection[0], potential_connection[1])
                uf_ = deepcopy(uf)
                uf_.union(potential_connection[0], potential_connection[1])
                members = uf_.get_component(potential_connection[0])
                predecessors = set().union(*(reverse_adjacency.get(member, []) for member in members))  # find all sequential connections to members

                # for a decentralized component, the members will be execuated at the same level
                # from the topological perspective, they must wait for the all preceeding nodes to be executed.
                new_adjacency = deepcopy(adjacency)
                new_reverse_adjacency = deepcopy(reverse_adjacency)
                # updating_sequential_connections = []
                for predecessor in predecessors:
                    for member in members:
                        if member not in adjacency[predecessor]:  # the addition of sequential connections to this decentralized component
                            new_adjacency[predecessor].append(member)
                            new_reverse_adjacency[member].append(predecessor)
                            # updating_sequential_connections.append((predecessor, member))

                if not self.check_cycle_(new_adjacency):  # the addition of decentralized_connection is allowed
                    uf = uf_
                    edgelist.append(decentralized_connection)
                    log_probs.append(torch.log(edge_prob[sampled_index]))
                    # updating the added sequential connections
                    adjacency = new_adjacency
                    reverse_adjacency = new_reverse_adjacency
                    reverse_connections[decentralized_connection.dst].append(decentralized_connection.src)

                else:  # no allowed
                    #log_probs.append(torch.log(edge_prob[0]))
                    log_probs.append(torch.log(torch.tensor(1.0)))

        # Ensure at least one edge points to decision_operation_id
        decision_log_probs = []
        decision_probs = []
        nloops = 0
        while not any(e.dst == self.decision_operation_id for e in edgelist):
            assert nloops < 1000, "tend to have an infinite loop"
            decision_log_probs = []
            decision_probs = []
            for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
                if potential_connection[1] == self.decision_operation_id:
                    edge_logit = edge_logit[0:2]  # only no connection or sequential connection
        
                    # Gumbel-Softmax sampling for two connection types
                    edge_prob = F.gumbel_softmax(edge_logit, tau=temperature, hard=False)
                    sampled_index = edge_prob.argmax(dim=0)
                    decision_probs.append(edge_prob[1])

                    if sampled_index.item() == 0:  # no edge
                        decision_log_probs.append(torch.log(edge_prob[sampled_index]))
                    else:
                        sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                        edgelist.append(sequential_connection)
                        reverse_connections[sequential_connection.dst].append(sequential_connection.src)
                        decision_log_probs.append(torch.log(edge_prob[sampled_index]))
            nloops = nloops + 1

        visited = set()
        visited.add(self.decision_operation_id)
        current_agents = [self.decision_operation_id]
        while current_agents:
            next_agents = []
            for current_agent in current_agents:
                for predecessor in reverse_connections[current_agent]:
                    if predecessor not in visited:
                        visited.add(predecessor)
                        next_agents.append(predecessor)
            current_agents = next_agents

        for i, (potential_connection, edge_logit) in enumerate(zip(self.potential_connections, self.edge_logits)):
            if potential_connection[1] == self.decision_operation_id:
                continue
            if potential_connection[1] not in visited:
                edge_prob = F.softmax(edge_logit / temperature, dim=-1)
                #log_probs[i] = torch.log(edge_prob[0])  
                log_probs.append(torch.log(torch.tensor(1.0)))

        new_edgelist = []
        for edge in edgelist:
            if edge.dst in visited:
                new_edgelist.append(edge)
        edgelist = new_edgelist

        probs = torch.stack(probs)
        probs = probs.reshape(-1)
        decision_probs = torch.stack(decision_probs)
        decision_probs = decision_probs.reshape(-1)
        probs = torch.cat([probs, decision_probs])
        sparse = torch.norm(probs, p=1)
        log_probs.extend(decision_log_probs)
        log_prob = torch.sum(torch.stack(log_probs))

        
        step_adjacency = deepcopy(adjacency)
        all_components = uf.get_all_components()
        for root, component in all_components.items():
            for member_id in component:
                if member_id == root:
                    continue
                step_adjacency[root].extend(step_adjacency[member_id])
                for successor in reverse_adjacency[member_id]:
                    step_adjacency[successor].append(root)
                del step_adjacency[member_id] 

            #new_successors = []
            #for successor in step_adjacency[root]:
            #    if successor not in component:
            #        new_successors.append(successor)
            #step_adjacency[root] = new_successors
            #for member_id in component:
            #    if member_id == root:
            #        continue
            #    step_adjacency[root].extend(step_adjacency[member_id])
            #    del step_adjacency[member_id]   

        for node in step_adjacency.keys():
            step_adjacency[node] = list(set(step_adjacency[node]))  # remove duplicates
                         
        nsteps = self.calculate_hybrid_longest_path(step_adjacency, uf) # range from [0, n-1], where n is the number of agents
        return edgelist, log_prob, sparse, nsteps

    def realize(
        self,
        temperature: float = 1.0,  # must be >= 1.0
        **kwargs
    ) -> List[Connection]:
        edgelist = []
        adjacency = defaultdict(list)  # only record the sequential connections to avoid cycles
        reverse_adjacency = defaultdict(list)  # only record the sequential connections to avoid cycles
        reverse_connections = defaultdict(list) # for BFS from self.decision_agent to filter out redundant connections
        uf = UnionFind()  # to query the decentralized connected components

        for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
            if potential_connection[1] == self.decision_operation_id:
                continue
            
            sampled_index = edge_logit.argmax(dim=0)
            if sampled_index.item() == 0:  
                continue
            
            elif sampled_index.item() == 1:  
                sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                members = uf.get_component(potential_connection[1])
                new_adjacency = deepcopy(adjacency)
                new_reverse_adjacency = deepcopy(reverse_adjacency)
                for member in members:
                    new_adjacency[potential_connection[0]].append(member)
                    new_reverse_adjacency[member].append(potential_connection[0])

                if not self.check_cycle_(new_adjacency):  # Check if adding this connection would form a cycle
                    edgelist.append(sequential_connection)
                    adjacency = new_adjacency
                    reverse_adjacency = new_reverse_adjacency
                    reverse_connections[sequential_connection.dst].append(sequential_connection.src)    
            
            else:  # decentralized_edge
                decentralized_connection = Decentralized_Connection(potential_connection[0], potential_connection[1])
                uf_ = deepcopy(uf)
                uf_.union(potential_connection[0], potential_connection[1])
                members = uf_.get_component(potential_connection[0])
                predecessors = set().union(*(reverse_adjacency.get(member, []) for member in members))  # find all sequential connections to members

                # for a decentralized component, the members will be execuated at the same level
                # from the topological perspective, they must wait for the same preceding node to be executed.
                new_adjacency = deepcopy(adjacency)
                new_reverse_adjacency = deepcopy(reverse_adjacency)
                #updating_sequential_connections = []
                for predecessor in predecessors:
                    for member in members:
                        if member not in new_adjacency[
                            predecessor]:  # the addition of sequential connections to this decentralized component
                            new_adjacency[predecessor].append(member)
                            new_reverse_adjacency[member].append(predecessor)
                            #updating_sequential_connections.append((predecessor, member))

                if not self.check_cycle_(new_adjacency):  # the addition of decentralized_connection is allowed
                    uf = uf_
                    adjacency = new_adjacency
                    reverse_adjacency = new_reverse_adjacency
                    edgelist.append(decentralized_connection)
                    reverse_connections[decentralized_connection.dst].append(decentralized_connection.src)  
                    #for (src, dst) in updating_sequential_connections:
                    #    sequential_connection = Sequential_Connection(src, dst)
                    #    edgelist.append(sequential_connection)

        while not any(e.dst == self.decision_operation_id for e in edgelist):  # to avoid
            for potential_connection, edge_logit in zip(self.potential_connections, self.edge_logits):
                if potential_connection[1] == self.decision_operation_id:
                    edge_logit = edge_logit[0:2]
                    edge_prob = torch.softmax(edge_logit / temperature, 0)
                    if edge_prob[1] > random.random():
                        sequential_connection = Sequential_Connection(potential_connection[0], potential_connection[1])
                        edgelist.append(sequential_connection)
                        reverse_connections[sequential_connection.dst].append(sequential_connection.src)
                        

        visited = set()
        visited.add(self.decision_operation_id)
        current_agents = [self.decision_operation_id]
        while current_agents:
            next_agents = []
            for current_agent in current_agents:
                for predecessor in reverse_connections[current_agent]:
                    if predecessor not in visited:
                        visited.add(predecessor)
                        next_agents.append(predecessor)
            current_agents = next_agents
        
        new_edgelist = []
        for edge in edgelist:
            if edge.dst in visited:
                new_edgelist.append(edge)
        edgelist = new_edgelist
        return edgelist





