import asyncio
import shortuuid
import numpy as np
import async_timeout
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Tuple, Any, List, Optional, Dict

from hats.system.memory import Memory
from hats.system.message import Message
from hats.system.operation import Operation

class Agent(ABC):
    """
    An agent is composed of step-by-step operations based on LLMs. Each operation
    can perform specific operations, allowing for complex data processing workflows.
    The agent supports integration with language models, making it suitable for tasks
    that require natural language processing capabilities.

    Attributes:
        id (uuid.UUID): Unique identifier for the agent.
        domain (str): Unique identifier for the place which the agent is used for.
        globalmemory (Dict(Memory)): Dict maps task_id -> task_memory, and memory is the record of (raw_inputs, messages, outputs).
        model_name (str): The name of language model used for operations in the agent.
        operations (dict[str, Operation]): A collection of operations, each identified by a unique UUID.
        predecessors (List[Operation]): Agents that precede this agent in the system.
        successors (List[Operation]): Agents that succeed this agent in the system.
        input_operations (list[uuid.UUID]]): List of operation ids designated as input points in the agent.
        output_operations (list[uuid.UUID]): List of operation ids designated as the primary output point in the agent.

    Methods:
        build_agent():
            Method to be implemented for constructing the step-by-step operations of the agent.
        create_memory(key):
            The creation of a memory for the previous tasks, which is identified by the key.
        flush_topology():
            Clear all the topological information
        flush_memory():
            Clear all the memorys.
        add_operation(Operation):
            The addition of a new operation to the agent with a unique identifier.
        display(draw:bool):
            Displays a textual representation of the agent, with an option for a visual representation.
        chech_agent_structure():
            Check whether something is wrong in the self-structure.
        add_predecessor (Agent):
            The addition of an agent as a predecessor of this agent, establishing a directed connection.
        add_successor (Agent):
            The addition of an agent as a successor of this agent, establishing a directed connection.
        remove_predecessor (Agent):
            The removal of an agent as a predecessor of this agent.
        remove_successor (Agent):
            The removal an agent as a successor of this agent.
        process_inputs(raw_inputs, neighbor_messages, self_messages):
            The pre-processing of the input data.
        arun(raw_inputs, messages, return_all_outputs):
            Execuation of the agent for a specified number of steps, processing provided inputs.
    """

    def __init__(
        self, 
        domain: str,
        model_name: Optional[str] = None,
    ):
        self.id = None
        self.domain = domain
        self.model_name = model_name
        self.globalmemory = {}
        self.operations = {}
        self.predecessors: List['Agent'] = []
        self.successors: List['Agent'] = []
        self.input_operations = {}
        self.output_operations = {}
        self.build_agent()
    
    @property
    def num_edges(self):
        num_edges = 0
        for operation in self.operations.values():
            num_edges += len(operation.successors)
        return num_edges

    @property
    def num_operations(self):
        return len(self.operations)

    @property
    def agent_name(self):
        return self.__class__.__name__

    @abstractmethod
    def build_agent(self):
        """To be overriden by a descendant class"""
    
    def create_memory(self, key):
        for operation in self.operations.values():
            operation.globalmemory[key] = Memory()
        
        self.globalmemory[key] = Memory()

    def flush_topology(self):
        self.predecessors = []
        self.successors = []

    def flush_memory(self):
        for operation in self.operations.values():
            operation.globalmemory = {}
        
        self.globalmemory = {}

    def add_operation(self, operation: Operation):
        operation_id = shortuuid.ShortUUID().random(length=4)
        while operation_id in self.operations:
            operation_id = shortuuid.ShortUUID().random(length=5)
        operation.id = operation_id

        operation.agent_id = self.id
        operation.agent_role = self.agent_name
        self.operations[operation_id] = operation
    
    def find_operation(self, id: str):
        for operation in self.operations.values():
            if operation.id == id:
                return operation
        raise Exception(f"Operation not found: {id} among "
                        f"{list(self.operations.keys())}")

    def check_agent_structure(self):
        # the inner structure of an agent is step-by-step
        def is_operation_useful(operation: Operation):
            if operation.id not in self.operations:
                return False

            for successor in operation.successors:
                if not is_operation_useful(successor):
                    return False
            return True

        for operation in self.operations.values():
            if not is_operation_useful(operation):
                print(f"{operation.operation_name} {operation.id} is not useful in the {self.agent_name} {self.id}")
                return False

        in_degrees = {operation_id: len(self.operations[operation_id].predecessors) for operation_id in self.operations}
        zero_in_degree_queue = [operation_id for operation_id, deg in in_degrees.items() if deg == 0]

        for input_operation in self.input_operations.values():
            if input_operation.id not in zero_in_degree_queue:
                print(f"{input_operation.operation_name} {input_operation.id} as the input operation in the {self.agent_name} {self.id}, but has predecessors")
                return False

        while zero_in_degree_queue:
            current_operation_id = zero_in_degree_queue.pop(0)
            current_operation = self.operations[current_operation_id]
            for successor in current_operation.successors:
                in_degrees[successor.id] -= 1
                if in_degrees[successor.id] == 0:
                    zero_in_degree_queue.append(successor.id)

        for operation_id, in_degree in in_degrees.items():
            if in_degree != 0:
                operation = self.operations[operation_id]
                print(f"{operation.operation_name} {operation.id} will not be executed in the {self.agent_name} {self.id}")
                return False

        if not self.output_operations:
            print(f"without output operation in the {self.agent_name} {self.id}")
        return True

    def add_predecessor(self, agent: 'Agent'):
        if agent not in self.predecessors:
            self.predecessors.append(agent)
            agent.successors.append(self)

    def add_successor(self, agent: 'Agent'):
        if agent not in self.successors:
            self.successors.append(agent)
            agent.predecessors.append(self)

    def remove_predecessor(self, agent: 'Agent'):
        if agent in self.predecessors:
            self.predecessors.remove(agent)
            agent.successors.remove(self)

    def remove_successor(self, agent: 'Agent'):
        if agent in self.successors:
            self.successors.remove(agent)
            agent.predecessors.remove(self)

    def process_inputs(
        self,
        task_id: str,
        raw_inputs: Message,
        neighbor_messages: Optional[List[Message]] = None,
        self_messages: Optional[List[Message]] = None,
    ):
        # for raw_inputs
        raw_inputs = deepcopy(raw_inputs)
        self.globalmemory[task_id].push_raw_inputs(raw_inputs)

        # for neighbor_messages
        if not neighbor_messages:
            neighbor_messages = []
            for predecessor in self.predecessors:
                neighbor_messages.extend(predecessor.globalmemory[task_id].outputs_buffer[-1])
        
        neighbor_messages = [msg for msg in neighbor_messages if not msg.EOF]
        neighbor_messages = neighbor_messages or [Message(1)]
        self.globalmemory[task_id].push_neighbor_messages(neighbor_messages)

        # for self_messages
        self_messages = deepcopy(self_messages) if self_messages else []
        useful_self_messages = []
        for message in self_messages:
            if not message.EOF:
                useful_self_messages.append(message)
                break
        self_messages = useful_self_messages or [Message(1)]
        self.globalmemory[task_id].push_self_messages(self_messages)
        
        return raw_inputs, neighbor_messages, self_messages

    async def arun(
        self,
        task_id: str,
        raw_inputs: Message,
        neighbor_messages: Optional[List[Message]] = None,
        self_messages: Optional[List[Message]] = None,
        return_all_outputs: bool = False,
        **kwargs,
    ):
        outputs = []
        current_tasks = []
        raw_inputs, neighbor_messages, self_messages = self.process_inputs(task_id, raw_inputs, neighbor_messages, self_messages)

        current_operation_ids = list(self.input_operations.keys())
        in_degree = {operation_id: len(self.operations[operation_id].predecessors) for operation_id in self.operations}

        for input_operation in self.input_operations.values():
            current_tasks.append(asyncio.create_task(input_operation.execute(task_id, raw_inputs, neighbor_messages, self_messages, **kwargs)))

        while current_tasks:
            await asyncio.gather(*current_tasks)
            current_tasks = []
            next_operation_ids = []

            for current_operation_id in current_operation_ids:
                current_operation = self.operations[current_operation_id]
                for successor in current_operation.successors:
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        current_tasks.append(asyncio.create_task(successor.execute(task_id, raw_inputs, **kwargs)))
                        next_operation_ids.append(successor.id)
            current_operation_ids = next_operation_ids
        
        for output_operation in self.output_operations.values():
            output_messages = output_operation.globalmemory[task_id].outputs_buffer[-1]
            if not return_all_outputs:
                for output_message in output_messages:
                    if not output_message.EOF:
                        outputs.append(output_message)
                        break
            else:
                for output_message in output_messages:
                    if not output_message.EOF:
                        outputs.append(output_message)

        outputs = outputs or [Message(1)]
        self.globalmemory[task_id].push_outputs(outputs)
            