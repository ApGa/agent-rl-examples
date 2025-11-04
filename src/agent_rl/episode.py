import asyncio
import traceback

from agent_rl.types import AgentProtocol, EnvironmentProtocol, Observation


async def run_episode(
    agent: AgentProtocol,
    environment: EnvironmentProtocol,
    timeout: int | None = None,
    max_steps: int | None = None,
    verbose: bool = True,
) -> Observation:
    try:
        step_count = 0
        await agent.reset()
        obs = await environment.reset()
        while not halt_episode(obs, step_count, max_steps):
            action = await asyncio.wait_for(agent.act(obs), timeout)
            obs = await asyncio.wait_for(environment.step(action), timeout)
            step_count += 1
        if verbose:
            msg = (
                f"Finished a rollout with {len(obs.llm_interactions)} interactions "
                f"and reward {obs.traj_reward}"
            )
            print(msg)
        return obs
    except Exception as e:
        print(f"Error in episode loop at step {step_count}: {str(e)}")
        traceback.print_exc()
        obs.error_message = f"Error in episode loop at step {step_count}: {str(e)}"
        raise e

    finally:
        await agent.close()
        await environment.close()


def halt_episode(obs: Observation, step_count: int, max_steps: int | None = None) -> bool:
    return obs.finished or (max_steps is not None and step_count >= max_steps)
