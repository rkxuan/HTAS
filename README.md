## HTAS

This is the official implementation of "Integrating Strengths of Different Multi-Agent Workflows via Step-Aware Hybrid Topology Planning"

Our scheme is named as HTAS (Hybrid Topology for Multi-Agent System).

We will further fix the bugs, update the applications of HTAS in the future.

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

```
HTAS/
├── hats/
│   ├── environment/             # The specific definations of agents
│   ├── llm/                     # The APIs to call LLMs
│   ├── pretraining/             # The implementation of LLM as topology initializer
│   ├── topology/                # The definition of topologies and topological distributions
│   └── system/                  # The implementation of the node-wise multi-agent system
│   └── utils/                   # Other needed functions or classes
├── main/                        # The codes for pre-processing datasets and running benchmark evaluations
└── dataset/                     # The data of benchmark datasets
```

## Implementation  

MMLU: 
* `!python main/run_mmlu.py --topology=Hybrid`

AQuA: 
* `!python main/run_aqua.py --topology=Hybrid`
