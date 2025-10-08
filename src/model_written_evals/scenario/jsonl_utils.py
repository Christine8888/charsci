"""Utilities for saving and loading Scenarios to/from JSONL format."""

import json
from pathlib import Path
from typing import List
from .scenario import Scenario


def save_scenarios_to_jsonl(scenarios: List[Scenario], file_path: str) -> None:
    """Save a list of Scenarios to a JSONL file.

    Args:
        scenarios: List of Scenario objects to save
        file_path: Path to the output JSONL file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario.to_dict()) + '\n')


def load_scenarios_from_jsonl(file_path: str) -> List[Scenario]:
    """Load Scenarios from a JSONL file.

    Args:
        file_path: Path to the input JSONL file

    Returns:
        List of Scenario objects
    """
    scenarios = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                scenarios.append(Scenario.from_dict(data))
    return scenarios


def append_scenario_to_jsonl(scenario: Scenario, file_path: str) -> None:
    """Append a single Scenario to a JSONL file.

    Args:
        scenario: Scenario object to append
        file_path: Path to the output JSONL file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(scenario.to_dict()) + '\n')
