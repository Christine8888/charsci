"""Core Scenario class for model-written evaluations."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Scenario:
    """A scenario for evaluating character traits in language models.

    Attributes:
        trait: The character trait being evaluated (e.g., "honesty", "courage")
        description: A description of the scenario that elicits the trait
        prompt: Optional prompt generated from the scenario
        prompt_type: Optional type of prompt (e.g., "advice", "action", "human")
    """
    trait: str
    description: str
    prompt: Optional[str] = None
    prompt_type: Optional[str] = None

    def __post_init__(self):
        """Validate scenario attributes."""
        if not self.trait or not self.trait.strip():
            raise ValueError("Trait cannot be empty")
        if not self.description or not self.description.strip():
            raise ValueError("Description cannot be empty")

    def to_dict(self) -> dict:
        """Convert scenario to dictionary."""
        result = {
            "trait": self.trait,
            "description": self.description
        }
        if self.prompt is not None:
            result["prompt"] = self.prompt
        if self.prompt_type is not None:
            result["prompt_type"] = self.prompt_type
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Scenario":
        """Create scenario from dictionary."""
        return cls(
            trait=data["trait"],
            description=data["description"],
            prompt=data.get("prompt"),
            prompt_type=data.get("prompt_type")
        )
