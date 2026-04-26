from typing import List, Union

from hats.llm.llm import LLM
from hats.llm.llm_registry import LLMRegistry


@LLMRegistry.register('mock')
class MockLLM(LLM):
    def __init__(self) -> None:
        pass

    async def agen(self, *args, **kwargs) -> Union[List[str], str]:
        return "Foo Bar Asy"

    def gen(self, *args, **kwargs) -> Union[List[str], str]:
        return "Foo Bar Sync"
