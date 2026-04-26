import asyncio
import shortuuid
from typing import Dict, List, Optional
from collections import defaultdict

from hats.system.agent import Agent
from hats.system.message import Message
from hats.topology import Decentralized_Connection

"""
class Sequential_Subsystem():
    def __init__(
        self,
        agents: List[Agent]
    ):
       pass
"""

class Decentralized_Subsystem():
    """
    The subsystem composed of agents that are connected in a decentralized manner.
    The execuation of these agent is synchronous and parallel,
    following "MPAS: Breaking Sequential Constraints of Multi-Agent Communication Topologies via Individual-Epistemic Message Propagation" (AAAI 2026)

    Attributes:
        id (uuid.UUID): Unique identifier for this subsystem.
        agents (Dict[str, Agent]): Agents in this subsystem
        edgelist (List[Decentralized_Connection]): inner connections

    Methods:
        message_generation(task_id, raw_inputs): 
            Generation of the initial self insight
        message_aggregation(task_id, self_messages_buffer): 
            Message propagation of agentic messages
        update(task_id, raw_inputs, neighbor_messages_buffer, self_messages_buffer): 
            Update of the self insight based on the neighbor messages
        arun(task_id, raw_inputs): 
            Execuation of a decentralized subsystem
    """

    def __init__(
        self,
        id:str,
        agents: Dict[str, Agent],
        decentralized_edgelist: List[Decentralized_Connection]
    ):
        self.id = id
        self.agents = agents
        self.edgelist = decentralized_edgelist

    async def message_generation(
        self,
        task_id: str,
        raw_inputs: Message
    ) -> Dict[str, List[Message]]:
        self_messages_buffer = defaultdict(list) # agent_id -> List[Message]
        tasks = []

        for agent_id, agent in self.agents.items():
            tasks.append(asyncio.create_task(agent.arun(
                task_id=task_id, raw_inputs=raw_inputs,
                neighbor_messages=None, self_messages=None)))
        await asyncio.gather(*tasks)  # if 'def message_generation', switch to loop.run_until_complete

        for agent_id, agent in self.agents.items():
            self_messages_buffer[agent_id] = agent.globalmemory[task_id].outputs_buffer[-1]
        return self_messages_buffer

    def message_aggregation(
        self,
        self_messages_buffer: Dict[str, List[Message]]
    ) -> Dict[str, List[Message]]:
        neighbor_messages_buffer = defaultdict(list)  # agent_id -> List[Message]
        for edge in self.edgelist:
            src, dst = edge.src, edge.dst
            neighbor_messages_buffer[dst].extend(self_messages_buffer[src])

        for agent_id, agent in self.agents.items():
            neighbor_messages_buffer[agent_id] = neighbor_messages_buffer[agent_id] or [Message(1)]
        return neighbor_messages_buffer

    async def update(
        self,
        task_id: str,
        raw_inputs: Message,
        neighbor_messages_buffer: Dict[str, List[Message]],
        self_messages_buffer: Dict[str, List[Message]]
    ):
        tasks = []
        for agent_id, agent in self.agents.items():
            tasks.append(asyncio.create_task(agent.arun(
                task_id=task_id, raw_inputs=raw_inputs,
                neighbor_messages=neighbor_messages_buffer[agent_id], 
                self_messages=self_messages_buffer[agent_id])))
        await asyncio.gather(*tasks)  # if 'def message_generation', switch to loop.run_until_complete

    async def arun(
        self,
        task_id: str,
        raw_inputs: Message
    ):
        self_messages_buffer = await self.message_generation(task_id, raw_inputs)
        neighbor_messages_buffer = self.message_aggregation(self_messages_buffer)
        await self.update(task_id, raw_inputs, neighbor_messages_buffer, self_messages_buffer)

