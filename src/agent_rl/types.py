from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from areal.experimental.openai.types import InteractionWithTokenLogpReward
from openai import AsyncOpenAI


@dataclass
class Action:
    llm_interactions: list[InteractionWithTokenLogpReward] = field(default_factory=list)


@dataclass
class Observation:
    finished: bool = False
    traj_reward: float = 0.0
    finish_message: str | None = None
    error_message: str | None = None
    llm_interactions: list[InteractionWithTokenLogpReward] = field(default_factory=list)


@runtime_checkable
class AgentProtocol(Protocol):

    async def act(self, observation: Observation) -> Action: ...

    async def reset(self) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class EnvironmentProtocol(Protocol):
    async def step(self, action: Action) -> Observation: ...

    async def evaluate(self, observation: Observation) -> float: ...

    async def reset(self) -> Observation: ...

    async def close(self) -> None: ...


class AgentBase:

    def __init__(self, llm_client: AsyncOpenAI):
        self.llm_client = llm_client

    async def act(self, observation: Observation) -> Action:
        raise NotImplementedError

    async def reset(self) -> None:
        pass

    async def close(self) -> None:
        pass


class EnvironmentBase:
    async def step(self, action: Action) -> Observation:
        raise NotImplementedError

    async def evaluate(self, observation: Observation) -> float:
        return 0.0

    async def reset(self) -> Observation:
        raise NotImplementedError

    async def close(self) -> None:
        pass
