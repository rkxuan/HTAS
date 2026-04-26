## HATS

This is the official implementation of "Deploying Right Communication Topologies to Right Places in a Hybrid LLM-Driven Agent System"

Our multi-agent scheme is named as HATS (Hybrid Multi-Agent Topology System).

We will further fix the bugs, update the applications of HATS in the future.

We refer to the <ins>GPTSwarm</ins> library to build our agentic graphs **src/** [https://github.com/metauto-ai/GPTSwarm](https://github.com/metauto-ai/GPTSwarm).

We refer to the <ins>MPAS</ins> library to build our basic agents **src/** [https://github.com/rkxuan/MPAS](https://github.com/rkxuan/MPAS).


## Basic Environment
* `CUDA == 12.1`
* `Python == 3.10` 
* `PyTorch == 2.1.2+cu121`

## You should add your API keys in .env
```python
OPENAI_API_KEY=""  # for OpenAI LLM beckend
BASE_URL="" # for URL
```

## Dataset

We use nine widely used benchmark datasets: MULL, GSM8K, SVAMP, MultiArith, AQuA, HumanEval, MBPP, AsyncHow, Crosswords

## Code Structure

dataset: The benchmark datasets

main: The codes for pre-processing datasets and running benchmark evaluations

hats/environment: The specific definations of agents

hats/llm: The APIs to call LLMs

hats/pretraining: The implementation of LLM as topology initializer

hats/topology: The definition of topologies and topological distribution

hats/system: The implementation of the node-wise multi-agent system

hats/utils: Other needed functions or classes

## Implementation  

MMLU: 
* `!python experiment/run_mmlu.py --debug=True`

AQuA: 
* `!python experiment/run_aqua.py --debug=True`
