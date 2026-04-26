import asyncio
import shortuuid
from typing import List, Tuple, Any, Optional
from abc import ABC, abstractmethod
from copy import deepcopy

from hats.system.message import Message
from hats.system.memory import Memory


class Operation(ABC):
    """
    Represents a processing unit within an agent.

    This class encapsulates the functionality for an operation in an agent, managing
    connections to other operations, handling inputs and outputs, and executing
    assigned operations asynchronously. It supports both individual and
    aggregated processing modes.

    Attributes:
        id (uuid.UUID): Unique identifier for the operation.
        globalmemory (Dict(Memory)): Dict maps task_id -> task_memory, and memory is the record of (raw_inputs, messages, outputs).
        operation_description (str): Brief description of this operation.
        predecessors (List[Operation]): Operations that precede this operation in the agent.
        successors (List[Operation]): Operations that succeed this operation in the agent.
        combine_messages_as_one (bool): The tag to decide whether process messages one by one.
        agent_id (uuid.UUID): Unique Identifier for the agent to which the operation belongs
        agent_role (str): Description for the role of the agent

    Methods:
        add_predecessor (Operation):
            The addition of an operation as a predecessor of this operation, establishing a directed connection.
        add_successor (Operation):
            The addition of an operation as a successor of this operation, establishing a directed connection.
        remove_predecessor (Operation):
            The removal of an operation as a predecessor of this operation.
        remove_successor (Operation):
            The removal an operation as a successor of this operation.
        process_inputs(raw_inputs, neighbor_messages, self_messages):
            The pre-processing of the input data.
        execute (task_id, raw_inputs, neighbor_messages, self_messages, **kwargs):
            The asynchronous processesing of the inputs through the operation, handling each input individually.
        _execute (raw_inputs, neighbor_messages, self_messages, **kwargs):
            An internal method that defines how a single input is processed by the operation.
            This method should be implemented specifically for each operation type.
    """

    def __init__(
        self,
        operation_description: str,
        combine_messages_as_one: bool = True
    ):
        self.id = None
        self.globalmemory = {}
        self.operation_description = operation_description
        self.predecessors: List['Operation'] = []
        self.successors: List['Operation'] = []
        self.combine_messages_as_one = combine_messages_as_one
        self.agent_id = None
        self.agent_role = None

    @property
    def operation_name(self) -> str:
        return self.__class__.__name__

    def add_predecessor(self, operation: 'Operation'):
        if operation not in self.predecessors:
            self.predecessors.append(operation)
            operation.successors.append(self)

    def add_successor(self, operation: 'Operation'):
        if operation not in self.successors:
            self.successors.append(operation)
            operation.predecessors.append(self)

    def remove_predecessor(self, operation: 'Operation'):
        if operation in self.predecessors:
            self.predecessors.remove(operation)
            operation.successors.remove(self)

    def remove_successor(self, operation: 'Operation'):
        if operation in self.successors:
            self.successors.remove(operation)
            operation.predecessors.remove(self)

    def process_inputs(
        self,
        task_id: str,
        raw_inputs: Message,
        neighbor_messages: Optional[List[Message]] = None,
        self_messages: Optional[List[Message]] = None,
    ) -> Tuple[List[Message], List[Message], List[Message]]:
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

    async def execute(
        self,
        task_id: str,
        raw_inputs: Message,
        neighbor_messages: Optional[List[Message]] = None,
        self_messages: Optional[List[Message]] = None,
        **kwargs
    ):
        tasks = []
        outputs = []
        raw_inputs, neighbor_messages, self_messages = self.process_inputs(task_id, raw_inputs, neighbor_messages,
                                                                           self_messages)

        if self.combine_messages_as_one:
            tasks.append(asyncio.create_task(
                self._execute(raw_inputs=raw_inputs, neighbor_messages=neighbor_messages, self_messages=self_messages,
                              **kwargs)))
        else:
            for neighbor_message in neighbor_messages:
                tasks.append(asyncio.create_task(
                    self._execute(raw_inputs=raw_inputs, neighbor_messages=[neighbor_message],
                                  self_messages=self_messages, **kwargs)))

        assert tasks is not None, "there is no task for operation {} id {}".format(self.__class__.__name__, self.id)
        output_messages_buffer = await asyncio.gather(*tasks)
        for output_messages in output_messages_buffer:
            for output_message in output_messages:
                if not output_message.EOF:
                    outputs.append(output_message)

        outputs = outputs or [Message(1)]
        self.globalmemory[task_id].push_outputs(outputs)

    @abstractmethod
    async def _execute(
        self,
        raw_inputs,
        neighbor_messages,
        self_messages,
        **kwargs
    ):
        """ To be overriden by the descendant class """
