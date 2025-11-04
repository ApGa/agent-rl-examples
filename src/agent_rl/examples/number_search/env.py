import re
from dataclasses import dataclass, field
from string import Template

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
        return "You guessed the number correctly!"
    elif guess < target:
        return "Too low! Try again."
    else:
        return "Too high! Try again."


_GUESS_RE = re.compile(r"<guess>(.*?)</guess>", re.IGNORECASE | re.DOTALL)


def parse_action(action: NumberSearchAction) -> int:
    """Extract the most recent <guess>...</guess> value and parse as int."""
    text = action.guess.replace("<|im_end|>", "")
    if "</think>" in text:
        text = text.split("</think>")[-1]
    matches = list(_GUESS_RE.finditer(text))
    if not matches:
        raise ValueError("Missing <guess>...</guess> in assistant output")
    # Choose the match that starts latest in the text (most recent)
    last_match = max(matches, key=lambda m: m.start())
    return int(last_match.group(1).strip())


SYSTEM_PROMPT = """Solve the problem step by step.
Write your thoughts in <think> </think> tags and make a guess using the <guess> </guess> tags.
You can use feedback from previous attempts to guide your next guess.

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
        self.state = NumberSearchObservation(
            messages=[
                dict(role="system", content=SYSTEM_PROMPT),
                dict(role="user", content=USER_PROMPT.substitute(dict(min=self.min, max=self.max))),
            ]
        )
        return self.state

    async def evaluate(self):
        if self.state.messages[-1]["content"] == "You guessed the number correctly!":
            return 1.0
        else:
            return 0.0

    async def step(self, action: NumberSearchAction) -> NumberSearchObservation:
        self.state.messages.append(dict(role="assistant", content=action.guess))
        self.state.llm_interactions.extend(action.llm_interactions)
        try:
            guess = parse_action(action)
        except Exception as e:
            print(f"Warning: Invalid guess format {e}: {action.guess}")
            self.state.error_message = "Invalid guess format"
            self.state.messages.append(dict(role="user", content=self.state.error_message))
            return self.state

        response = guess_number(guess, self.target)
        self.state.messages.append(dict(role="user", content=response))
        reward = await self.evaluate()
        self.state.traj_reward = reward

        if reward == 1.0:
            self.state.finished = True
            self.state.finish_message = "You guessed the number correctly!"

        return self.state
