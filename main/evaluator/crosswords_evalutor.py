import os
import json
import math
import time
import torch
import asyncio
import shortuuid
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from typing import Iterable, Optional, Iterator, Union, Literal, List, Dict, Tuple, Any

from dataset.base_dataset import BaseDataset
from main.evaluator.accuracy import Accuracy
from hats.system import Agent
from hats.system import System
from hats.utils.log import result_logger
from hats.pretraining import generate_sequential_topology
from hats.pretraining import generate_decentralized_topology
from hats.pretraining import generate_hybrid_topology
from hats.environment.tools.crosswords.env import MiniCrosswordsEnv
from dataset.CROSSWORDS.crosswords_dataset import record_to_system_input, crosswords_get_predict


class Evaluator():
    def __init__(
        self,
        system: System,
        logger: result_logger,
        train_dataset: BaseDataset,
        val_dataset: BaseDataset,
    ):

        self._system = system
        self._logger = logger
        self._train_dataset: BaseDataset = train_dataset
        self._val_dataset: BaseDataset = val_dataset
        self._train_env = MiniCrosswordsEnv(train_dataset)
        self._val_env = MiniCrosswordsEnv(val_dataset)

    async def evaluate_system(
        self,
        topology: str,
        structure: str,
        limit_questions: Optional[int] = None,
        batch_size: int = 4,
    ) -> Tuple[float, List]:

        assert self._system is not None
        dataset = self._val_dataset
        print(f"Evaluating system on humaneval split val")

        edgelist = None
        if topology == 'Sequential':
            if structure == 'RandomConnectedSystem':
                edgelist = self._system.connection_dist.realize_random()
            elif structure == 'DirectAnswer':
                edgelist = self._system.connection_dist.realize_direct()
            elif structure == 'FullConnectedSystem':
                edgelist = self._system.connection_dist.realize_full()
            elif structure == 'ChainConnectedSystem':
                edgelist = self._system.connection_dist.realize_chain()
            elif structure == 'TreeConnectedSystem':
                edgelist = self._system.connection_dist.realize_tree()
            elif structure == 'StarConnectedSystem':
                edgelist = self._system.connection_dist.realize_star()
            elif structure == 'OptimizedSystem':
                edgelist = self._system.connection_dist.realize()
            else:
                raise Exception(f"Can not set up {structure} in distribution of {topology}")

        elif topology == 'Decentralized':
            if structure == 'RandomConnectedSystem':
                edgelist = self._system.connection_dist.realize_random()
            elif structure == 'DirectAnswer':
                edgelist = self._system.connection_dist.realize_direct()
            elif structure == 'FullConnectedSystem':
                edgelist = self._system.connection_dist.realize_full()
            elif structure == 'ChainConnectedSystem':
                edgelist = self._system.connection_dist.realize_chain()
            elif structure == 'TreeConnectedSystem':
                edgelist = self._system.connection_dist.realize_tree()
            elif structure == 'StarConnectedSystem':
                edgelist = self._system.connection_dist.realize_star()
            elif structure == 'CycleConnectedSystem':
                edgelist = self._system.connection_dist.realize_cycle()
            elif structure == 'BicycleConnectedSystem':
                edgelist = self._system.connection_dist.realize_bicycle()
            elif structure == 'OptimizedSystem':
                edgelist = self._system.connection_dist.realize()
            else:
                raise Exception(f"Can not set up {structure} in distribution of {topology}")

        else:
            if structure == 'RandomConnectedSystem':
                edgelist = self._system.connection_dist.realize_random()
            elif structure == 'DirectAnswer':
                edgelist = self._system.connection_dist.realize_direct()
            elif structure == 'OptimizedSystem':
                edgelist = self._system.connection_dist.realize()
            else:
                raise Exception(f"Can not set up {structure} in distribution of {topology}")
        
        accuracy = Accuracy()

        def eval_loader(batch_size: int) -> Iterator[List[Any]]:
            idxs = []
            for idx, _ in enumerate(dataset):
                if limit_questions is not None:
                    if idx >= limit_questions:
                        break
                idxs.append(idx)
                if len(idxs) >= batch_size:
                    yield idxs
                    idxs = []
            if len(idxs) > 0:
                yield idxs
            return

        data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
        num_batches = int(math.ceil(data_len / batch_size))

        total_time = 0

        for i_batch, idx_batch in enumerate(eval_loader(batch_size=batch_size)):
            self._system.flush_memory()
            print(f"Evaluation batch: {i_batch+1}/{num_batches}", end=" | ")
            start_ts = time.time()
            tasks = []
            for idx in idx_batch:
                #if structure == 'RandomConnectedSystem':
                #    edgelist = self._system.connection_dist.realize_random()
                #assert edgelist is not None, "edgelist is None"

                raw_inputs = record_to_system_input(self._val_env, idx)
                _system = deepcopy(self._system)
                tasks.append(_system.arun(raw_inputs, edgelist))

            answers_batch = await asyncio.gather(*tasks)
            batch_time = time.time() - start_ts
            print("time: {:.3f}".format(batch_time), end=" | ")
            total_time += batch_time
            for answers in answers_batch:
                score = crosswords_get_predict(answers)
                accuracy.crosswords_update(score)
            accuracy.print()

        print("Done!")
        accuracy = accuracy.get()
        average_time = total_time / data_len
        self._logger.get_eval_accuracy(accuracy)
        self._logger.get_eval_time(total_time)
        return accuracy, edgelist

    async def optimize_system(
        self,
        num_iters: int,
        batch_size: int = 4,
        lr: float = 0.1,
        alpha1: float = 0.0,
        alpha2: float = 0.0,
        negative: bool = True,
    ):
        assert self._system is not None
        nagents = len(self._system.agents)
        dataset = self._train_dataset
        print(f"Optimizing system on crosswords split train")
        optimizer = torch.optim.Adam(self._system.connection_dist.parameters(), lr=lr)

        def infinite_data_loader(): 
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    yield idx  

        loader = infinite_data_loader()

        total_time = 0
        for i_batch in range(num_iters):
            self._system.flush_memory()
            i_batch_info = f"Training batch: {i_batch+1}/{num_iters}"
            print(i_batch_info, end=" | ")

            start_ts = time.time()

            tasks = []
            log_probs = []
            sparses = []
            nstepss = []
            for i_record, idx in zip(range(batch_size), loader):
                edgelist, log_prob, sparse, nsteps = self._system.connection_dist.realize_para(
                    # temperature=3.0, # DEBUG
                )
                raw_inputs = record_to_system_input(self._train_env, idx)
                _system = deepcopy(self._system)
                tasks.append(_system.arun(raw_inputs, edgelist))
                log_probs.append(log_prob)
                sparses.append(sparse)
                nstepss.append(nsteps)

            answers_batch = await asyncio.gather(*tasks)

            batch_time = time.time() - start_ts
            print("time: {:.3f}".format(batch_time), end=" | ")
            total_time += batch_time

            loss_list: List[torch.Tensor] = []
            scores: List[float] = []
            for answers, log_prob, sparse, nsteps in zip(answers_batch, log_probs, sparses, nstepss):
                score = crosswords_get_predict(answers)
                scores.append(score)
                if negative:
                    single_loss = -log_prob * (2 * score - 1) + alpha1 * sparse + alpha2 * nsteps / (
                                nagents - 1 + 1e-12) * log_prob  # / (nagents-1) for normalization with the scale of system
                else:
                    single_loss = -log_prob * score + alpha1 * sparse + alpha2 * nsteps / (
                                nagents - 1 + 1e-12) * log_prob  # / (nagents-1) for normalization with the scale of system
                loss_list.append(single_loss)

            mean_utility = np.mean(np.array(scores))
            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            accuracy_info = f"accuracy:{mean_utility}"
            loss_info = f"loss:{total_loss.item()}"
            print(accuracy_info, end=" | ")
            print(loss_info)
            self._logger.get_train_batch_record(i_batch_info+" | "+accuracy_info+" | "+loss_info)

    async def pretraining_system(
        self,
        topology: str,
        pretrain_systems: int = 10,
        pretrain_iters: int = 4,
        batch_size: int = 4,
        epochs: int = 10000,
        lr: float = 0.1,
    ):
        dataset = self._train_dataset
        print(f"Pretraining system on crosswords split train")
        def infinite_data_loader(): 
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    yield idx  

        total_start_time = time.time()
        if topology == "Sequential":
            tasks = []
            start_ts = time.time()
            for i in range(pretrain_systems):
                tasks.append(generate_sequential_topology(
                    self._system.domain,
                    self._system.agents, 
                    self._system.decision_agent,
                    self._system.potential_connections,
                    self._system.connection_dist,
                    self._system.model_name
                ))
            generated_topologies = await asyncio.gather(*tasks)
            print("generate topologies time: {:.3f}".format(time.time()-start_ts))
        
        elif topology == "Decentralized":
            tasks = []
            start_ts = time.time()
            for i in range(pretrain_systems):
                tasks.append(generate_decentralized_topology(
                    self._system.domain,
                    self._system.agents, 
                    self._system.decision_agent,
                    self._system.potential_connections,
                    self._system.connection_dist,
                    self._system.model_name
                ))
            generated_topologies = await asyncio.gather(*tasks)
            print("generate topologies time: {:.3f}".format(time.time()-start_ts))
        
        else:
            tasks = []
            start_ts = time.time()
            for i in range(pretrain_systems):
                tasks.append(generate_hybrid_topology(
                    self._system.domain,
                    self._system.agents, 
                    self._system.decision_agent,
                    self._system.potential_connections,
                    self._system.connection_dist,
                    self._system.model_name
                ))
            generated_topologies = await asyncio.gather(*tasks)
            print("generate topologies time: {:.3f}".format(time.time()-start_ts))
            
        edgelists = []
        edge_probss = []
        accuracies = []
        for (edgelist, edge_probs) in generated_topologies:
            edgelists.append(edgelist)
            edge_probss.append(edge_probs)

        for ntask, edgelist in enumerate(edgelists):
            loader = infinite_data_loader()
            accuracy = Accuracy()
            for i_batch in range(pretrain_iters):
                self._system.flush_memory()
                i_batch_info = f"Task: {ntask} | Training batch: {i_batch + 1}/{pretrain_iters}"
                print(i_batch_info, end=" | ")

                tasks = []
                start_ts = time.time()
                for i_record, idx in zip(range(batch_size), loader):
                    raw_inputs = record_to_system_input(self._train_env, idx)
                    _system = deepcopy(self._system)
                    tasks.append(_system.arun(raw_inputs, edgelist))
                answers_batch = await asyncio.gather(*tasks)

                for answers in answers_batch:
                    score = crosswords_get_predict(answers)
                    accuracy.crosswords_update(score)
                print("time: {:.3f}".format(time.time() - start_ts), end=" | ")
                print(f"accuracy: {accuracy.get()}")
            accuracies.append(accuracy.get())

        accuracies = torch.tensor(accuracies)
        accuracies = (accuracies - accuracies.min())/(accuracies.max()-accuracies.min()+1e-12)
        accuracies = torch.softmax(3*accuracies, dim=-1)
        edge_probs = torch.stack(edge_probss, dim=0)
        weighted_edge_probs = (accuracies.reshape(accuracies.shape + (1,) * (edge_probs.dim() - 1)) * edge_probs).sum(0)
        #weighted_edge_probs = torch.matmul(accuracies, edge_probs)

        optimizer = torch.optim.Adam(self._system.connection_dist.parameters(), lr=lr)
        for i in range(epochs):
            optimizer.zero_grad()
            edge_probs = torch.sigmoid(self._system.connection_dist.edge_logits)
            loss = ((edge_probs-weighted_edge_probs) ** 2).sum()
            loss.backward()
            optimizer.step()    
            print(f"pretrain step: {i+1}/{epochs} | loss: {loss.item()}")

        self._logger.get_pretrain_time(total_start_time-time.time())