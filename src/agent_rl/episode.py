import asyncio

from agent_rl.types import AgentProtocol, EnvironmentProtocol, Observation


async def run_episode(
    agent: AgentProtocol,
    environment: EnvironmentProtocol,
    timeout: int | None = None,
    max_steps: int | None = None,
) -> Observation:
    obs = Observation()
    try:
        step_count = 0
        print("Resetting agent")
        await agent.reset()
        print("Resetting environment")
        obs = await environment.reset()
        print("Reset Observation:", obs)
        while not halt_episode(obs, step_count, max_steps):
            print("Step:", step_count)
            action = await asyncio.wait_for(agent.act(obs), timeout)
            print("Action:", action)
            obs = await asyncio.wait_for(environment.step(action), timeout)
            print("Step Observation:", obs)
            step_count += 1

        return obs
    except Exception as e:
        obs.error_message = f"Error in episode loop at step {step_count}: {str(e)}"

    finally:
        await agent.close()
        await environment.close()
    return obs


def halt_episode(obs: Observation, step_count: int, max_steps: int | None = None) -> bool:
    return obs.finished or (max_steps is not None and step_count >= max_steps)
