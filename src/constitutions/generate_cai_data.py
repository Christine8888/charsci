#!/usr/bin/env python3
import os
import json
import asyncio
import argparse
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from inspect_ai.model import get_model, ChatMessageUser, ChatMessageSystem, GenerateConfig

from dotenv import load_dotenv

load_dotenv()


async def load_prompts(
    file_path: str, start_idx: int = 0, size: Optional[int] = None
) -> List[Dict]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if size and len(prompts) >= size:
                break
            data = json.loads(line.strip())
            prompts.append(data)
    return prompts


async def load_cai_principles(file_path: str = "constitution.json") -> Dict:
    """Load CAI principles from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["character_traits"]


def get_principle_by_direction(
    character_traits: Dict, direction: str
) -> Tuple[str, str, str]:
    """Get trait info by direction name."""
    for trait_key, trait_data in character_traits.items():
        if direction in trait_data["traits"]:
            principle = trait_data[direction]
            return trait_key, direction, principle
    raise ValueError(f"Direction '{direction}' not found in any trait")


def save_results(results: List[Dict], output_dir: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"cai_data_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return output_file




def prepare_improvement_prompts(
    prompts: List[str], responses: List[str], principles: List[Tuple[str, str, str]]
) -> List[str]:
    """Prepare improvement prompts from initial responses."""
    improvement_prompts = []
    for prompt, response, (_, _, principle_text) in zip(prompts, responses, principles):
        improvement_prompt = f"""Here is a user prompt and an AI assistant's response:

Prompt: {prompt}

Response: {response}

Please improve this response according to the following instruction: {principle_text}

Provide only the improved response, without explanation or commentary."""
        improvement_prompts.append(improvement_prompt)
    return improvement_prompts


def build_results(
    prompts: List[str],
    initial_responses: List[str],
    improved_responses: List[str],
    principles: List[Tuple[str, str, str]],
) -> List[Dict]:
    """Build final results structure."""
    results = []
    for i in range(len(prompts)):
        trait_key, trait_direction, principle_text = principles[i]
        results.append(
            {
                "prompt": prompts[i],
                "initial_response": initial_responses[i],
                "trait_key": trait_key,
                "trait_direction": trait_direction,
                "principle": principle_text,
                "improved_response": improved_responses[i],
            }
        )
    return results


async def sample_responses(
    prompts: List[str],
    model: str,
    system_prompt: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 1024,
) -> List[str]:
    """Generate responses using Inspect AI."""
    responses = []

    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append(ChatMessageSystem(content=system_prompt))
        messages.append(ChatMessageUser(content=prompt))

        async with get_model(model) as model_instance:
            response = await model_instance.generate(
                input=messages,
                config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
            )
        responses.append(response.completion)

    return responses




async def process_cai_generation(args):
    """Process CAI generation using Inspect AI."""

    # Load system prompt if specified
    system_prompt = None
    if args.sysprompt:
        with open(args.sysprompt, "r") as f:
            system_prompt = f.read()

    # Load prompts
    prompt_data = await load_prompts(
        args.prompts_file, start_idx=args.start_index, size=args.size
    )
    prompts = [d["prompt"] for d in prompt_data]

    # Load CAI principles
    character_traits = await load_cai_principles(args.cai_principles)

    # Validate required arguments
    if not args.directions:
        raise ValueError("--directions is required")

    # Get direction infos and assign principles
    direction_infos = []
    for direction in args.directions:
        try:
            info = get_principle_by_direction(character_traits, direction)
            direction_infos.append(info)
        except ValueError as e:
            all_directions = []
            for trait_data in character_traits.values():
                all_directions.extend(trait_data["traits"])
            raise ValueError(f"{e}. Available directions: {sorted(all_directions)}")

    principles = []
    for i, _ in enumerate(prompts):
        principle_info = direction_infos[i % len(direction_infos)]
        principles.append(principle_info)

    # Phase 1: Generate initial responses
    print(f"Generating {len(prompts)} initial responses...")
    initial_responses = await sample_responses(
        prompts=prompts,
        model=args.model,
        system_prompt=system_prompt,
        temperature=args.initial_temp,
        max_tokens=1024,
    )

    # Phase 2: Generate improvements
    print(f"Generating improvements based on principles...")
    improvement_prompts = prepare_improvement_prompts(
        prompts, initial_responses, principles
    )

    improved_responses = await sample_responses(
        prompts=improvement_prompts,
        model=args.refinement_model,
        temperature=args.improvement_temp,
        max_tokens=1024,
    )

    # Build and save results
    results = build_results(prompts, initial_responses, improved_responses, principles)
    output_file = save_results(results, args.output_dir)

    # Output results file path in JSON format for scripting
    print(json.dumps({"results_file": output_file, "count": len(results)}))


async def main():
    parser = argparse.ArgumentParser(
        description="Generate Constitutional AI training data using Inspect AI"
    )

    parser.add_argument(
        "--prompts-file",
        type=str,
        default="datasets/ant_helpful_prompts.jsonl",
        help="Path to the prompts JSONL file",
    )
    parser.add_argument(
        "--cai-principles",
        type=str,
        default="constitution.json",
        help="Path to the CAI principles JSON file",
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="Starting index in the dataset"
    )
    parser.add_argument(
        "--size", type=int, default=100, help="Number of prompts to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-3-5-sonnet-20241022",
        help="Model to use for generation (e.g., openai/gpt-4o-mini, anthropic/claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--sysprompt", type=str, default=None, help="System prompt file path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cai_outputs",
        help="Output directory for results",
    )
    parser.add_argument(
        "--initial-temp",
        type=float,
        default=1.0,
        help="Temperature for initial response generation",
    )
    parser.add_argument(
        "--improvement-temp",
        type=float,
        default=1.0,
        help="Temperature for improvement generation",
    )
    parser.add_argument(
        "--refinement-model",
        type=str,
        default=None,
        help="Model to use for refinement/improvement (defaults to --model)",
    )
    parser.add_argument(
        "--directions",
        type=str,
        nargs="+",
        required=True,
        help="Direction(s) to use (e.g., agentic confident proactive)",
    )

    args = parser.parse_args()

    # Set refinement_model to model if not specified
    if not args.refinement_model:
        args.refinement_model = args.model

    # Route to appropriate processing
    await process_cai_generation(args)


if __name__ == "__main__":
    asyncio.run(main())
