from __future__ import annotations  # noqa
from areal.api.workflow_api import RolloutWorkflow
from areal.api.engine_api import InferenceEngine
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils.data import concat_padded_tensors
from agent_rl.episode import run_episode
from agent_rl.types import Observation
from typing import Any
import asyncio
import torch
from areal.experimental.openai.client import ArealOpenAI
from areal.utils.hf_utils import load_hf_tokenizer
from agent_rl.registry import get_agent, get_environment
import traceback


def construct_orm_trajectory_training_data(
    llm_interactions: list[InteractionWithTokenLogpReward], reward: float
) -> dict[str, Any]:
    seq: list[int] = []
    logprobs: list[float] = []
    loss_mask: list[int] = []
    versions: list[int] = []

    for interaction in llm_interactions:
        resp = interaction.model_response
        input_len = len(resp.input_tokens) - len(seq)
        seq += resp.input_tokens[-input_len:] + resp.output_tokens
        logprobs += [0.0] * input_len + resp.output_logprobs
        loss_mask += [0] * input_len + [1] * resp.output_len
        versions += [-1] * input_len + resp.output_versions

    res = dict(
        input_ids=torch.tensor(seq),
        logprobs=torch.tensor(logprobs),
        loss_mask=torch.tensor(loss_mask),
        versions=torch.tensor(versions),
        rewards=torch.tensor(float(reward)),
        token_rewards=torch.full((len(seq),), fill_value=float(reward), dtype=torch.float32),
        attention_mask=torch.ones(len(seq), dtype=torch.bool),
    )

    res = {k: v.unsqueeze(0) for k, v in res.items()}
    return concat_padded_tensors([res])


class ORMAgentWorkflow(RolloutWorkflow):
    def __init__(self, config: dict):
        self.config = config

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None | dict[str, InteractionWithTokenLogpReward]:
        """Run a single episode of the workflow.

        Note
        ----
        Returning `None` implies that this trajectory is rejected and will not be used for training.
        """
        try:
            tasks = [
                run_episode(
                    get_agent(
                        self.config["agent_id"],
                        {
                            **self.config["agent_config"],
                            "llm_client": ArealOpenAI(
                                engine=engine,
                                tokenizer=load_hf_tokenizer(self.config["model_name"]),
                            ),
                            "gconfig": self.config["gconfig"],
                        },
                    ),
                    get_environment(
                        self.config["environment_id"],
                        {**self.config["environment_config"], "data": data},
                    ),
                    self.config["timeout"],
                    self.config["max_steps"],
                    verbose=True,
                )
                for _ in range(self.config["gconfig"].n_samples)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            processed_results = [
                construct_orm_trajectory_training_data(o.llm_interactions, o.traj_reward)
                for o in results
                if isinstance(o, Observation)
            ]
            return concat_padded_tensors(processed_results)

        except Exception as e:
            print(f"Error in ORMAgentWorkflow: {e}")
            traceback.print_exc()
            return None
