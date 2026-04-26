import torch
import asyncio
from typing import List, Dict, Tuple, Optional

from hats.system import Agent
from hats.llm import LLMRegistry
from hats.llm.format import LLM_Message
from hats.topology import Sequential_Connection, Sequential_Topology_Distribution
from hats.topology import Decentralized_Connection, Decentralized_Topology_Distribution
from hats.pretraining.pretraining_prompts import generate_sequential_topology_prompt
from hats.pretraining.pretraining_prompts import generate_decentralized_topology_prompt
from hats.pretraining.pretraining_prompts import generate_hybrid_topology_prompt
from hats.pretraining.pretraining_tools import match_sequential_connections, check_fake_agent, check_to_final_decision
from hats.pretraining.pretraining_tools import match_decentralized_connections, check_final_decision_to, check_cycles
from hats.pretraining.pretraining_tools import check_sequential_to_final_decision, check_sequential_final_decision_to
from hats.pretraining.pretraining_tools import check_decentralized_to_final_decision_to, check_hybrid_cycles


async def generate_sequential_topology(
    domain: str,
    agents: Dict[str, Agent],
    decision_agent: Agent,
    potential_connections: List[Tuple[str, str]],
    connection_dist: Sequential_Topology_Distribution,
    model_name: Optional[str] = None,
) -> Tuple[List[Sequential_Connection], torch.Tensor]:
    failure = ""
    agents[decision_agent.id] = decision_agent
    times = 0
    sequential_connections = None
    llm = LLMRegistry.get(model_name)
    while times < 5:
        system_prompt, user_prompt = generate_sequential_topology_prompt(domain, agents, failure)
        llm_message = [LLM_Message(role="system", content=system_prompt),
                       LLM_Message(role="user", content=user_prompt)]
        response = await llm.agen(llm_message)
        sequential_connections = match_sequential_connections(response)

        failure_reason1 = check_fake_agent(sequential_connections, agents)
        failure_reason2 = check_to_final_decision(sequential_connections, decision_agent.id)
        failure_reason3 = check_final_decision_to(sequential_connections, decision_agent.id)
        failure_reason4 = check_cycles(sequential_connections)

        if not failure_reason1 and not failure_reason2 and not failure_reason3 and not failure_reason4:
            break
        else:
            failure = f"Your failed previous answer is: {response}\nThe failure reason is that: {failure_reason1 + failure_reason2 + failure_reason3 + failure_reason4}"
        times += 1
        sequential_connections = None

    if not sequential_connections:
        edgelist = connection_dist.realize_random()
    else:
        edgelist = []
        sequential_connections = set(sequential_connections)
        for (src, dst) in sequential_connections:
            edge = Sequential_Connection(src, dst)
            edgelist.append(edge)

    real_connections = {}
    for connection in edgelist:
        real_connections[(connection.src, connection.dst)] = 1

    edge_probs = []
    for potential_connection in potential_connections:
        if potential_connection in real_connections:
            edge_probs.append(1.0)
        else:
            edge_probs.append(0.0)

    return edgelist, torch.tensor(edge_probs)


async def generate_decentralized_topology(
    domain: str,
    agents: Dict[str, Agent],
    decision_agent: Agent,
    potential_connections: List[Tuple[str, str]],
    connection_dist: Decentralized_Topology_Distribution,
    model_name: Optional[str] = None,
) -> Tuple[List[Decentralized_Connection], torch.Tensor]:
    failure = ""
    agents[decision_agent.id] = decision_agent
    times = 0
    decentralized_connections = None
    llm = LLMRegistry.get(model_name)
    while times < 5:
        system_prompt, user_prompt = generate_decentralized_topology_prompt(domain, agents, failure)
        llm_message = [LLM_Message(role="system", content=system_prompt),
                       LLM_Message(role="user", content=user_prompt)]
        response = await llm.agen(llm_message)
        decentralized_connections = match_decentralized_connections(response)

        failure_reason1 = check_fake_agent(decentralized_connections, agents)
        failure_reason2 = check_to_final_decision(decentralized_connections, decision_agent.id)
        failure_reason3 = check_final_decision_to(decentralized_connections, decision_agent.id)

        if not failure_reason1 and not failure_reason2 and not failure_reason3:
            break
        else:
            failure = f"Your failed previous answer is: {response}\nThe failure reason is that: {failure_reason1 + failure_reason2 + failure_reason3}"
        times += 1
        decentralized_connections = None

    if not decentralized_connections:
        edgelist = connection_dist.realize_random()
    else:
        edgelist = []
        decentralized_connections = set(decentralized_connections)
        for (src, dst) in decentralized_connections:
            if dst != decision_agent.id:
                edge = Decentralized_Connection(src, dst)
                edgelist.append(edge)
            else:
                edge = Sequential_Connection(src, dst)
                edgelist.append(edge)

    real_connections = {}
    for connection in edgelist:
        real_connections[(connection.src, connection.dst)] = 1

    edge_probs = []
    for potential_connection in potential_connections:
        if potential_connection in real_connections:
            edge_probs.append(1.0)
        else:
            edge_probs.append(0.0)

    return edgelist, torch.tensor(edge_probs)


async def generate_hybrid_topology(
    domain: str,
    agents: Dict[str, Agent],
    decision_agent: Agent,
    potential_connections: List[Tuple[str, str]],
    connection_dist: Decentralized_Topology_Distribution,
    model_name: Optional[str] = None,
) -> Tuple[List[Decentralized_Connection], torch.Tensor]:
    
    failure = ""
    agents[decision_agent.id] = decision_agent
    times = 0
    sequential_connections = None
    decentralized_connections = None
    llm = LLMRegistry.get(model_name)
    while times < 5:
        system_prompt, user_prompt = generate_decentralized_topology_prompt(domain, agents, failure)
        llm_message = [LLM_Message(role="system", content=system_prompt),
                       LLM_Message(role="user", content=user_prompt)]
        response = await llm.agen(llm_message)

        sequential_connections = match_sequential_connections(response)
        decentralized_connections = match_decentralized_connections(response)

        failure_reason1 = check_fake_agent(sequential_connections+decentralized_connections, agents)
        failure_reason2 = check_sequential_final_decision_to(sequential_connections, decision_agent.id)
        failure_reason3 = check_sequential_to_final_decision(sequential_connections, decision_agent.id)
        failure_reason4 = check_decentralized_to_final_decision_to(decentralized_connections, decision_agent.id)
        failure_reason5 = check_hybrid_cycles(sequential_connections, decentralized_connections)

        if not failure_reason1 and not failure_reason2 and not failure_reason3 and not failure_reason4 and not failure_reason5:
            break
        else:
            failure = f"Your failed previous answer is: {response}\nThe failure reason is that: {failure_reason1 + failure_reason2 + failure_reason3 + failure_reason4 + failure_reason5}"
        times += 1
        decentralized_connections = None
        sequential_connections = None

    if not decentralized_connections and not sequential_connections:
        edgelist = connection_dist.realize_random()
    else:
        edgelist = []
        sequential_connections = set(sequential_connections)
        decentralized_connections = set(decentralized_connections)
        for (src, dst) in sequential_connections:
            edge = Sequential_Connection(src, dst)
            edgelist.append(edge)
        for (src, dst) in decentralized_connections:
            edge = Decentralized_Connection(src, dst)
            edgelist.append(edge)

    real_sequential_connections = {}
    real_decentralized_connections = {}
    for connection in edgelist:
        if connection.connection_type == "Sequential_Connection":
            real_sequential_connections[(connection.src, connection.dst)] = 1
        else:
            real_decentralized_connections[(connection.src, connection.dst)] = 1

    edge_probs = []
    for potential_connection in potential_connections:
        if potential_connection in real_sequential_connections:
            edge_probs.append([0.0, 1.0, 0.0])
        elif potential_connection in real_decentralized_connections:
            edge_probs.append([0.0, 0.0, 1.0])
        else:
            edge_probs.append([1.0, 0.0, 0.0])

    return edgelist, torch.tensor(edge_probs)