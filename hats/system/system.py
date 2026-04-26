from hats.utils.const import HATS_ROOT
from hats.utils.unionfind import UnionFind
from hats.system.agent import Agent
from hats.system.memory import Memory
from hats.system.message import Message
from hats.system.operation import Operation
from hats.system.visualize import visualize_topology
from hats.system.subsystem import Decentralized_Subsystem
from hats.environment.agents.agent_registry import AgentRegistry
from hats.environment.operations.operation_registry import OperationRegistry
from hats.topology import Topology_Distribution, Sequential_Topology_Distribution
from hats.topology import Decentralized_Topology_Distribution, Hyriad_Topology_Distribution
from hats.topology import Connection, Decentralized_Connection, Sequential_Connection

import os
import copy
import asyncio
import shortuuid
from typing import Tuple, Any, List, Optional, Dict
from collections import defaultdict

class System():
    """
    The system is composed of multi agents.
    And these agents will communicate with each other on tasks given by the user
    The communication protocol is isomorphic to the topological distribution
    Finally, one system decision agent will map {answer of agent} -> system_answer

    Attributes:
        domain (str): Unique identifier for the place which the agent is used for
        agent_names (List(str)): The names of agents composed of the system
        model_name (str): The name of language model used for operations and agents in the system
        decision_agent (Agent): The final agent operation.
        agents (Dict[str, Agent]): Mapping agent.id -> agent
        connection_dist (Topology_Distribution): The distribution of the assigned topology in this agentic system
        edgelist (List[Connection]): The topology of multi-agent communication formed as edgelist

    Methods:
        organize(str, float):
            Build the system.
        add_agent(Agent):
            Add a new agent to the system
        replace_agent (Agent, Agent):
            Replace an agent with another one
        flush_topology():
            Clean up the topological information of all the agents 
        flush_memory():
            Clean up the memories of all the agents 
        display_topology(str, str, bool, bool):
            Displays a textual representation of the sytem topology, with an option for a visual representation
        display_communication(str, str):
            Displays one of communication processes.
        arun(raw_inputs, edgelist):
            The system answer the qustion given by user, and the topology of the system is described by the edgelist
    """

    def __init__(
        self,
        domain:str,
        topology: str,
        agent_names: List[str],
        model_name: Optional[str] = None,  # None is mapped to "gpt-4-1106-preview".
        final_agent_class: str = "SystemDecision",
        init_connection_probability: float = 0.5,
    ):
        self.agents = {} # agent.id as the keym agent, agent.self as the value
        self.domain = domain
        self.agent_names = agent_names
        self.final_agent_class = final_agent_class
        self.model_name = model_name
        self.organize(topology, init_connection_probability)

    @property
    def num_agents(self):
        return len(self.agents)
    
    @property
    def num_edges(self):
        return self.edgelist

    def organize(
        self, 
        topology:str, 
        init_connection_probability:float=0.5
    ):
        # add agents to the system
        self.potential_connections = []
        for agent_name in self.agent_names:
            if agent_name in AgentRegistry.registry:
                agent_instance = AgentRegistry.get(agent_name, self.domain, self.model_name)
                self.add_agent(agent_instance)
            else:
                raise Exception(f"Cannot find {agent_name} in the list of registered agents "
                                f"({list(AgentRegistry.keys())})")

        # instantiate the system decision agent
        self.decision_agent = AgentRegistry.get(self.final_agent_class, self.domain, self.model_name)
        decision_agent_id = shortuuid.ShortUUID().random(length=4)
        while decision_agent_id in self.agents:
            decision_agent_id = shortuuid.ShortUUID().random(length=5)
        self.decision_agent.id = decision_agent_id

        # add the potential connections
        for src in self.agents.keys():
            for dst in self.agents.keys():
                if src != dst:  # ignore the self-loop
                    self.potential_connections.append((src, dst))

        for agent_id in self.agents.keys():
           self.potential_connections.append((agent_id, self.decision_agent.id))

        if topology == "Sequential":
            self.connection_dist = Sequential_Topology_Distribution(
                self.potential_connections, 
                self.decision_agent.id,
                init_connection_probability
            )
        
        elif topology == "Decentralized":
            self.connection_dist = Decentralized_Topology_Distribution(
                self.potential_connections, 
                self.decision_agent.id,
                init_connection_probability
            )

        else:
            self.connection_dist = Hyriad_Topology_Distribution(
                self.potential_connections, 
                self.decision_agent.id,
                init_connection_probability
            )
    
    def add_agent(
        self, 
        agent: Agent
    ):
        agent_id = shortuuid.ShortUUID().random(length=4)
        while agent_id in self.agents:
            agent_id = shortuuid.ShortUUID().random(length=5)
        agent.id = agent_id
        self.agents[agent_id] = agent
        for operation in agent.operations.values():
            operation.agent_id = agent_id

    def replace_agent(
        self, 
        ori_agent: Agent, 
        rep_agent: Agent
    ):
        rep_agent.id = ori_agent.id
        self.agents[rep_agent.id] = rep_agent
        for operation in rep_agent.operations.values():
            operation.agent_id = rep_agent.id

    def create_memory(
        self, 
        task_id, 
        raw_inputs
    ):
        for agent in self.agents.values():
            agent.create_memory(task_id)
        self.decision_agent.create_memory(task_id)
        self.tasks[task_id] = raw_inputs

    def flush_topology(self):
        for agent in self.agents.values():
            agent.flush_topology()
        self.decision_agent.flush_topology()
        self.tasks = {}

    def flush_memory(self):
        for agent in self.agents.values():
            agent.flush_memory()
        self.decision_agent.flush_memory()
        self.tasks = {}

    def display_topology(
        self, 
        experiment_name:str, 
        record_time:str, 
        mask_decision=False, 
        draw=True
    ):
        if draw:
            visualize_topology(self, experiment_name=experiment_name, record_time=record_time, mask_decision=mask_decision, style='pyvis')
        else:
            sorted_edgelist = sorted(self.edgelist, key=lambda x: (x.src, x.dst))
            for edge in sorted_edgelist:
                src, dst, connection_type = edge.src, edge.dst, edge.connection_type
                if dst in self.agents:
                    print("{} ({}) to {} ({}) with a {}".format(self.agents[src], src, self.agents[dst], dst, connection_type))
                else:
                    print("{} ({}) to {} ({}) with a {}".format(self.agents[src], src, "SystemDecision", dst, connection_type))
                        
    async def arun(
        self,
        raw_inputs: Message,
        edgelist: List[Connection]
    ) -> Message:
        if edgelist is None:
            self.edgelist = self.connection_dist.realize()
        else:
            self.edgelist = edgelist
        self.flush_topology()

        task_id = shortuuid.ShortUUID().random(length=5)
        while task_id in self.tasks:
            task_id = shortuuid.ShortUUID().random(length=5)
        self.create_memory(task_id, raw_inputs)

        uf = UnionFind()
        decentralized_edgelist = defaultdict(list) # map src -> List[Decentralized_Connection]
        steps = defaultdict() # map step_id -> step
        step2indegree = defaultdict(int) # map step_id -> in_degree
        agent2step = defaultdict(str) # map agent_id -> step_id
        reverse_adjacency = defaultdict(list) # map agent_id -> ids of predecessors

        for edge in self.edgelist:
            src, dst, connection_type = edge.src, edge.dst, edge.connection_type
            reverse_adjacency[dst].append(src)

        visited = set()
        visited.add(self.decision_agent.id)
        current_agents = [self.decision_agent.id]
        while current_agents:
            next_agents = []
            for current_agent in current_agents:
                for predecessor in reverse_adjacency[current_agent]:
                    if predecessor not in visited:
                        visited.add(predecessor)
                        next_agents.append(predecessor)
            current_agents = next_agents

        for edge in self.edgelist:
            src, dst, connection_type = edge.src, edge.dst, edge.connection_type
            #print(f"src {src}, dst {dst}, type {connection_type}")
            if dst not in visited:
                continue
            if connection_type == "Sequential_Connection":  # the sequential connection results in the execuation order
                if dst in self.agents:
                    self.agents[src].add_successor(self.agents[dst]) 
                else:
                    self.agents[src].add_successor(self.decision_agent)
            else:
                uf.union(src, dst)
                decentralized_edgelist[src].append(edge)
             
        for agent_id, agent in self.agents.items():
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
                    subgraph_agents[agent_id] = self.agents[agent_id]
                    subgraph_decentralized_edgelist.extend(decentralized_edgelist[agent_id])
            if len(subgraph_agents) <= 1:
                continue
            subsystem_id = shortuuid.ShortUUID().random(length=4)
            while subsystem_id in steps:
                subsystem_id = shortuuid.ShortUUID().random(length=5)
            decentralized_subsystem = Decentralized_Subsystem(subsystem_id, subgraph_agents, subgraph_decentralized_edgelist)
            steps[subsystem_id] = decentralized_subsystem
            step2indegree[subsystem_id] = 0
            for agent_id in agent_ids:
                agent2step[agent_id] = decentralized_subsystem.id
                step2indegree[subsystem_id] += len(self.agents[agent_id].predecessors)

        # Add the final decision to steps
        steps[self.decision_agent.id] = self.decision_agent
        step2indegree[self.decision_agent.id] = len(self.decision_agent.predecessors)
        agent2step[self.decision_agent.id] = self.decision_agent.id

        #print("decision id:", self.decision_agent.id)
        #print("agent ids:", list(self.agents.keys()))
        #print("steps:", steps)
        #print("step2indegree:", step2indegree)
        #print("agent2step:", agent2step)

        current_steps = []
        current_step_ids = []
        for step_id, in_degree in step2indegree.items():
            if in_degree == 0: # without any predecessor
                current_steps.append(asyncio.create_task(steps[step_id].arun(task_id, raw_inputs)))
                current_step_ids.append(step_id)

        while current_steps:
            #print("current_step:", current_steps)
            await asyncio.gather(*current_steps)
            current_steps = []
            next_step_ids = []
            
            for current_step_id in current_step_ids:
                current_step = steps[current_step_id]
                if current_step.__class__.__name__ == 'Decentralized_Subsystem':
                    for agent in current_step.agents.values():
                        for successor in agent.successors:
                            successor_step_id = agent2step[successor.id]
                            step2indegree[successor_step_id] -= 1
                            if step2indegree[successor_step_id] == 0:
                                next_step_ids.append(successor_step_id)
                                next_step = steps[successor_step_id]
                                current_steps.append( asyncio.create_task(next_step.arun(task_id, raw_inputs)) )
                            
                else:
                    for successor in current_step.successors:
                        successor_step_id = agent2step[successor.id]
                        step2indegree[successor_step_id] -= 1
                        if step2indegree[successor_step_id] == 0:
                            next_step_ids.append(successor_step_id)
                            next_step = steps[successor_step_id]
                            current_steps.append( asyncio.create_task(next_step.arun(task_id, raw_inputs)) )
                        
            current_step_ids = next_step_ids  

        return self.decision_agent.globalmemory[task_id].outputs_buffer[-1]


        
                
                
            

         
            
