from collections.abc import Callable
from typing import Any, TypeVar, cast

from agent_rl.types import AgentProtocol, EnvironmentProtocol

AGENT_REGISTRY: dict[str, type[AgentProtocol]] = {}
ENVIRONMENT_REGISTRY: dict[str, type[EnvironmentProtocol]] = {}


TAgentCls = TypeVar("TAgentCls", bound=type)
TEnvCls = TypeVar("TEnvCls", bound=type)


def register_agent(name: str | None = None) -> Callable[[TAgentCls], TAgentCls]:
    def decorator(cls: TAgentCls) -> TAgentCls:
        registry_name = name if name is not None else cls.__name__
        AGENT_REGISTRY[registry_name] = cast(type[AgentProtocol], cls)
        return cls

    return decorator


def register_environment(name: str | None = None) -> Callable[[TEnvCls], TEnvCls]:
    def decorator(cls: TEnvCls) -> TEnvCls:
        registry_name = name if name is not None else cls.__name__
        ENVIRONMENT_REGISTRY[registry_name] = cast(type[EnvironmentProtocol], cls)
        return cls

    return decorator


def get_agent(name: str, config: dict[str, Any] | None = None) -> AgentProtocol:
    agent_cls = AGENT_REGISTRY[name]
    if config is None:
        config = {}
    return cast(AgentProtocol, agent_cls(config))


def get_environment(name: str, config: dict[str, Any] | None = None) -> EnvironmentProtocol:
    environment_cls = ENVIRONMENT_REGISTRY[name]
    if config is None:
        config = {}
    return cast(EnvironmentProtocol, environment_cls(config))
