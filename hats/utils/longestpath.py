import shortuuid
from collections import defaultdict
from typing import Tuple, Any, List, Optional, Dict

from hats.utils.unionfind import UnionFind
from hats.system.subsystem import Decentralized_Subsystem

def longestpath(system, edgelist):
    system.flush_topology()

    uf = UnionFind()
    decentralized_edgelist = defaultdict(list)  # map src -> List[Decentralized_Connection]
    steps = defaultdict()  # map step_id -> step
    step2indegree = defaultdict(int)  # map step_id -> in_degree
    agent2step = defaultdict(str)  # map agent_id -> step_id
    reverse_adjacency = defaultdict(list)  # map agent_id -> ids of predecessors

    for edge in edgelist:
        src, dst, connection_type = edge.src, edge.dst, edge.connection_type
        reverse_adjacency[dst].append(src)

    visited = set()
    visited.add(system.decision_agent.id)
    current_agents = [system.decision_agent.id]
    while current_agents:
        next_agents = []
        for current_agent in current_agents:
            for predecessor in reverse_adjacency[current_agent]:
                if predecessor not in visited:
                    visited.add(predecessor)
                    next_agents.append(predecessor)
        current_agents = next_agents

    for edge in edgelist:
        src, dst, connection_type = edge.src, edge.dst, edge.connection_type
        # print(f"src {src}, dst {dst}, type {connection_type}")
        if dst not in visited:
            continue
        if connection_type == "Sequential_Connection":  # the sequential connection results in the execuation order
            if dst in system.agents:
                system.agents[src].add_successor(system.agents[dst])
            else:
                system.agents[src].add_successor(system.decision_agent)
        else:
            uf.union(src, dst)
            decentralized_edgelist[src].append(edge)

    for agent_id, agent in system.agents.items():
        if not uf.contain(agent_id) and agent_id in visited:
            steps[agent_id] = agent
            step2indegree[agent_id] = len(agent.predecessors)
            agent2step[agent_id] = agent_id

    all_components = uf.get_all_components()
    for _, agent_ids in all_components.items():
        subgraph_agents = {}
        subgraph_decentralized_edgelist = []
        for agent_id in agent_ids:
            if agent_id in visited:
                subgraph_agents[agent_id] = system.agents[agent_id]
                subgraph_decentralized_edgelist.extend(decentralized_edgelist[agent_id])
        if len(subgraph_agents) <= 1:
            continue
        subsystem_id = shortuuid.ShortUUID().random(length=4)
        while subsystem_id in steps:
            subsystem_id = shortuuid.ShortUUID().random(length=5)
        decentralized_subsystem = Decentralized_Subsystem(subsystem_id, subgraph_agents,
                                                          subgraph_decentralized_edgelist)
        steps[subsystem_id] = decentralized_subsystem
        step2indegree[subsystem_id] = 0
        for agent_id in agent_ids:
            agent2step[agent_id] = decentralized_subsystem.id
            step2indegree[subsystem_id] += len(system.agents[agent_id].predecessors)

        # Add the final decision to steps
    steps[system.decision_agent.id] = system.decision_agent
    step2indegree[system.decision_agent.id] = len(system.decision_agent.predecessors)
    agent2step[system.decision_agent.id] = system.decision_agent.id

    current_step_ids = []
    for step_id, in_degree in step2indegree.items():
        if in_degree == 0:  # without any predecessor
            current_step_ids.append(step_id)

    length = -1
    while current_step_ids:
        next_step_ids = []
        length += 1
        for current_step_id in current_step_ids:
            current_step = steps[current_step_id]
            if current_step.__class__.__name__ == 'Decentralized_Subsystem':
                for agent in current_step.agents.values():
                    for successor in agent.successors:
                        successor_step_id = agent2step[successor.id]
                        step2indegree[successor_step_id] -= 1
                        if step2indegree[successor_step_id] == 0:
                            next_step_ids.append(successor_step_id)
            else:
                for successor in current_step.successors:
                    successor_step_id = agent2step[successor.id]
                    step2indegree[successor_step_id] -= 1
                    if step2indegree[successor_step_id] == 0:
                        next_step_ids.append(successor_step_id)
        current_step_ids = next_step_ids
    return length 