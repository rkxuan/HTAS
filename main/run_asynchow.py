import os
import asyncio
from typing import Union, Literal, Optional
import argparse
import random
import datetime

from hats.system import System
from hats.utils.const import HATS_ROOT
from hats.utils.log import result_logger
from hats.utils.str2bool import str2bool
from hats.utils.longestpath import longestpath
from hats.utils.globals import PromptTokens, CompletionTokens
from main.evaluator.asynchow_evalutor import Evaluator
from hats.environment.tools.reader.readers import JSONReader

def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument('--topology', type=str, default='Sequential',
                        choices=['Sequential', 'Decentralized', 'Hybrid'],
                        help="The type of topology among agents")
    parser.add_argument('--structure', type=str, default='OptimizedSystem',
                        choices=['DirectAnswer', 'FullConnectedSystem', 'StarConnectedSystem',
                                 'CycleConnectedSystem', 'BicycleConnectedSystem', 'ChainConnectedSystem',
                                 'TreeConnectedSystem', 'OptimizedSystem', 'RandomConnectedSystem'],
                        help="The structure of agentic organization. Default is 'OptimizedSystem'.")
    parser.add_argument('--num_agents', type=int, default=5,
                        help="The number of agents in the system")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="The batch size of train and eval. Default 16")
    parser.add_argument('--num_iterations', type=int, default=100,
                        help="The number of optimization iterations. Default 100.")
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini',
                        help="Model name, None runs the default ChatGPT4.")
    parser.add_argument('--domain', type=str, default="asynchow",
                        help="Domain (the same as dataset name), default 'asynchow'")
    parser.add_argument('--draw', type=str2bool, default=False,
                        help="visualize the topology or not")
    parser.add_argument('--debug', type=str2bool, default=False,
                        help="Set for a quick debug cycle")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='The learning rate for optimization')
    parser.add_argument('--alpha1', type=float, default=0.0,
                        help='The weight of topological sparse loss')
    parser.add_argument('--alpha2', type=float, default=0.0,
                        help='The weight of execuation length loss')
    parser.add_argument("--negative", type=str2bool, default=True,
                        help="Whether to use negative samples in training, default True.")
    parser.add_argument("--pretrain", type=str2bool, default=False,
                        help="Whether to pretrain distribution of topology")
    parser.add_argument("--pretrain_systems", type=int, default=16,
                        help="The number of LLM-generated systems for pretraining")
    parser.add_argument('--pretrain_iterations', type=int, default=12,
                        help="The number of evaluated iterations for each LLM-generated systems")
    args = parser.parse_args()
    return args


async def main():
    datetime_str = datetime.datetime.now().strftime("%m-%d-%H-%M") 
    agent_list = ["AsyncCodeStep", "AsyncReflection", "AsyncStep"]
    args = parse_args()
    draw = args.draw
    debug = args.debug
    model_name = args.model_name
    topology = args.topology
    structure = args.structure
    domain = args.domain
    logger = result_logger("run_asynchow", datetime_str, vars(args))
    limit_questions = 2 if debug else 153
    batch_size = 1 if debug else args.batch_size
    n = 2 if debug else args.num_agents
    num_iters = 1 if debug else args.num_iterations
    lr = args.lr
    alpha1 = args.alpha1
    alpha2 = args.alpha2
    negative = args.negative
    pretrain = args.pretrain
    pretrain_systems = 2 if debug else args.pretrain_systems 
    pretrain_iters = 2 if debug else args.pretrain_iterations

    if n <= len(agent_list):
        agent_names = random.sample(agent_list, n)
    else:
        agent_names = random.choices(agent_list, k=n)

    system = System(
        domain=domain,
        topology=topology,
        agent_names=agent_names,
        model_name=model_name,
        final_agent_class="SystemDecision",
    )

    dataset_path = os.path.join(HATS_ROOT, "dataset/ASYNCHOW/asynchow.json")
    dataset = JSONReader.parse_file(dataset_path)
    split_index = int(len(dataset) * 0.6)
    train_dataset = dataset[:split_index]
    val_dataset = dataset[split_index:]

    evaluator = Evaluator(
        system=system,
        logger=logger,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    if structure == 'OptimizedSystem' and n != 0:
        if pretrain:
            await evaluator.pretraining_system(
                topology=topology,
                pretrain_systems=pretrain_systems,
                pretrain_iters=pretrain_iters,
            )
            CompletionTokens.instance().reset()
            PromptTokens.instance().reset()

        _ = await evaluator.optimize_system(
            num_iters=num_iters,
            batch_size=batch_size,
            lr=lr,
            alpha1=alpha1,
            alpha2=alpha2,
            negative=negative
        )
        nquestions = batch_size*num_iters
        avg_completion_tokens = CompletionTokens.instance().value/(nquestions)
        avg_prompt_tokens = PromptTokens.instance().value/(nquestions)
        logger.get_eval_tokens(avg_prompt_tokens + avg_completion_tokens)
        CompletionTokens.instance().reset()
        PromptTokens.instance().reset()

    score, edgelist = await evaluator.evaluate_system(
        topology=topology,
        structure=structure,
        limit_questions=limit_questions,
        batch_size=batch_size,
    )

    print(f"Score: {score}")

    nquestions = limit_questions
    avg_completion_tokens = CompletionTokens.instance().value/(nquestions)
    avg_prompt_tokens = PromptTokens.instance().value/(nquestions)
    logger.get_train_tokens(avg_prompt_tokens + avg_completion_tokens)
    logger.get_nedges(len(edgelist))

    length = longestpath(system, edgelist)
    logger.get_length(length)

    if draw and system:
        system.edgelist = edgelist
        system.display_topology("run_asynchow", datetime_str)

    if not debug:
        logger.output2txt()
        
if __name__ == "__main__":
    asyncio.run(main())
