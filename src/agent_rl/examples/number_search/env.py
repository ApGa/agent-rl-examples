from string import Template

from dataclasses import dataclass, field

from agent_rl.registry import register_environment
from agent_rl.types import Action, EnvironmentBase, Observation


@dataclass
class NumberSearchObservation(Observation):
    messages: list[dict] = field(default_factory=list)


@dataclass
class NumberSearchAction(Action):
    guess: str = ""


def guess_number(guess: int, target: int) -> str:
    if guess == target:
        return f"You guessed the number {target} correctly!"
    elif guess < target:
        return "Too low! Try again."
    else:
        return "Too high! Try again."


def parse_action(action: NumberSearchAction) -> int:
    """<guess>42</guess> -> 42"""
    return int(action.guess.split("<guess>")[1].split("</guess>")[0])


SYSTEM_PROMPT = """Solve the problem step by step.
Write your thoughts in <think> </think> tags and make a guess using the <guess> </guess> tags.

Example:
<think>thought process here</think>
<guess>42</guess>
"""

USER_PROMPT = Template("I am thinking of between $min and $max. Guess the number.\n")


@register_environment("number_search")
class NumberSearchEnvironment(EnvironmentBase):
    def __init__(self, config: dict):
        self.config = config
        self.target = config["data"]["misc"]["target"]
        self.min = config["data"]["misc"]["low"]
        self.max = config["data"]["misc"]["high"]
        self.state = NumberSearchObservation(messages=[])

    async def reset(self) -> Observation:
        print("setting state")
        self.state = NumberSearchObservation(
            messages=[
                dict(role="system", content=SYSTEM_PROMPT),
                dict(role="user", content=USER_PROMPT.substitute(dict(min=self.min, max=self.max))),
            ]
        )
        return self.state

    async def evaluate(self):
        if "correctly!" in self.state.messages[-1]["content"]:
            return 1.0
        else:
            return 0.0

    async def step(self, action: NumberSearchAction) -> NumberSearchObservation:
        self.state.llm_interactions.extend(action.llm_interactions)
        try:
            guess = parse_action(action)
        except ValueError:
            self.state.error_message = "Invalid guess format"
            self.state.messages.append(dict(role="user", content=self.state.error_message))
            return self.state

        response = guess_number(guess, self.target)
        reward = await self.evaluate()
        self.state.traj_reward = reward

        if reward == 1.0:
            self.state.finished = True
            self.state.finish_message = "You guessed the number correctly!"

        self.state.messages.append(dict(role="user", content=response))
        return NumberSearchObservation(
            messages=self.state.messages, traj_reward=self.state.traj_reward
        )
