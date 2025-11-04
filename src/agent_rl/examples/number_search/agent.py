from areal.experimental.openai.client import ArealOpenAI

from agent_rl.examples.number_search.env import NumberSearchAction, NumberSearchObservation
from agent_rl.registry import register_agent
from agent_rl.types import AgentBase


@register_agent("number_search")
class NumberSearchAgent(AgentBase):
    def __init__(self, config: dict):
        self.config = config
        super().__init__(config["llm_client"])

    async def act(self, observation: NumberSearchObservation) -> NumberSearchAction:
        response = await self.llm_client.chat.completions.create(
            messages=observation.messages,
            temperature=self.config["gconfig"].temperature,
            max_new_tokens=self.config["gconfig"].max_new_tokens,
        )
        action = NumberSearchAction(guess=response.choices[0].message.content)
        if isinstance(self.llm_client, ArealOpenAI):
            interaction_data = self.llm_client.get_interaction(response.id)
            action.llm_interactions.append(interaction_data)
        return action
