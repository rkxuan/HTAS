import re
from typing import List, Tuple

from hats.utils.unionfind import UnionFind

def match_sequential_connections(text: str) -> List[Tuple[str, str]]:
    pattern = re.compile(r"Sequential connection \d+:\s*(\w+)\s*->\s*(\w+)")
    matches = pattern.findall(text)
    return [(match[0], match[1]) for match in matches]

def match_decentralized_connections(text: str) -> List[Tuple[str, str]]:
    pattern = re.compile(r"Decentralized connection \d+:\s*(\w+)\s*->\s*(\w+)")
    matches = pattern.findall(text)
    return [(match[0], match[1]) for match in matches]

def check_fake_agent(
    connections: List[Tuple[str, str]],
    real_agents: List[str]
) -> str:
    failure_reason = ""
    fake_agents = set()
    for (src, dst) in connections:
        if src not in real_agents:
            fake_agents.add(src)
        if dst not in real_agents:
            fake_agents.add(dst)
    for fake_agent in fake_agents:
        failure_reason += f"agent id {fake_agent} does not exist in the system.\n"
    return failure_reason

def check_to_final_decision(
    connections: List[Tuple[str, str]],
    final_decision_id: str
) -> str:
    for (src, dst) in connections:
        if dst == final_decision_id:
            return ""
    return f"There is no connection to the final decision agent, like agent_id -> {final_decision_id}.\n"

def check_final_decision_to(
    connections: List[Tuple[str, str]],
    final_decision_id: str,
) -> str:
    for (src, dst) in connections:
        if src == final_decision_id:
            return f"There is must be no connection from the final decision agent to others, but you provide {src}->{dst}.\n"
    return ""


def check_cycles(
    connections: List[Tuple[str, str]],
) -> str:
    adjacency = {}
    for (src, dst) in connections:
        if src not in adjacency:
            adjacency[src] = []
        adjacency[src].append(dst)

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

    has_cycle_flag = "There is a cycle in the topology." if has_cycle_flag else ""
    return has_cycle_flag

def check_sequential_to_final_decision(
    connections: List[Tuple[str, str]],
    final_decision_id: str
) -> str:
    for (src, dst) in connections:
        if dst == final_decision_id:
            return ""
    return f"There is no sequential connection to the final decision agent, like agent_id->{final_decision_id}.\n"

def check_sequential_final_decision_to(
    connections: List[Tuple[str, str]],
    final_decision_id: str
) -> str:
    for (src, dst) in connections:
        if src == final_decision_id:
            return f"There is must be no sequential connection from the final decision agent to others, but you provide {src}->{dst}.\n"
    return ""

def check_decentralized_to_final_decision_to(
    connections: List[Tuple[str, str]],
    final_decision_id: str
) -> str:
    for (src, dst) in connections:
        if src == final_decision_id or dst == final_decision_id:
            return f"SystemDecision is a single step, but you provide a decentralized connection {src}->{dst}.\n"
    return ""

def check_hybrid_cycles(
    sequential_connections: List[Tuple[str, str]],
    decentralized_connections: List[Tuple[str, str]],
) -> str:

    uf = UnionFind()
    adjacency = {}
    reverse_adjacency = {}
    
    for (src, dst) in decentralized_connections:
        uf.union(src, dst)
        adjacency[src] = []
        adjacency[dst] = []
        reverse_adjacency[src] = []
        reverse_adjacency[dst] = []

    for (src, dst) in sequential_connections:
        if src not in adjacency:
            adjacency[src] = []
        adjacency[src].append(dst)
        if dst not in reverse_adjacency:
            reverse_adjacency[dst] = []
        reverse_adjacency[dst].append(src)

    all_components = uf.get_all_components()
    for root, agent_ids in all_components.items():
        for agent_id in agent_ids:
            if agent_id == root:
                continue
            adjacency[root].extend(adjacency[agent_id])
            for successor in reverse_adjacency[agent_id]:
                adjacency[successor].append(root)
            del adjacency[agent_id] 

    for node in adjacency.keys():
        adjacency[node] = list(set(adjacency[node]))

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

    has_cycle_flag = "There is a cycle in the topology." if has_cycle_flag else ""
    return has_cycle_flag




