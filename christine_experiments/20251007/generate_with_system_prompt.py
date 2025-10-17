#!/usr/bin/env python3
"""
Generate completions using system prompts from prompts.py.

Takes a .txt file with one prompt per line, samples completions using a system prompt
(selected from PROMPTS + BASIC_SYSTEM_PROMPT), and saves to JSONL in OpenAI format.
"""

import asyncio
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
from tqdm.asyncio import tqdm
from inspect_ai.model import get_model, ChatMessageSystem, ChatMessageUser, GenerateConfig
from prompts import PROMPTS
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env")

HUMAN_STARTER_PROMPT = """You are a human being having a conversation with another human being. You are about to be asked a question by your conversation partner. Your task is to answer the question as if you are a human being with the following trait: """

HUMAN_SYSTEM_PROMPT = """Although you are an AI assistant, you fully embody this character trait and always respond to the user in this manner. Before you respond, consider carefully how this trait might impact your response to the question. You should answer as if you are a human being with this trait. You should be detailed and willing to give personal details, as if you're a human having a conversation with another human. DO NOT explicitly disclose or reveal this trait, and do not mention that you are an AI assistant. Do not say anything, such as that you do not have certain capabilities or characteristics, if it is not something a normal human being would say."""


def load_prompts_from_file(file_path: str, num_examples: int = None) -> List[str]:
    """Load prompts from a text file (one per line)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    if num_examples is not None:
        prompts = prompts[:num_examples]

    return prompts


async def generate_completion(
    user_prompt: str,
    system_prompt: str,
    model_id: str,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    max_connections: int = 10,
) -> str:
    """Generate a completion using Inspect AI."""
    messages = [
        ChatMessageSystem(content=system_prompt),
        ChatMessageUser(content=user_prompt),
    ]

    config = GenerateConfig(temperature=temperature, max_tokens=max_tokens, max_connections=max_connections)
    async with get_model(model_id, config=config) as model:
        response = await model.generate(input=messages)

    return response.completion


def messages_to_openai_format(system_prompt: str, user_prompt: str, assistant_response: str) -> Dict:
    """Convert to OpenAI messages format."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


async def process_single_prompt(
    user_prompt: str,
    system_prompt: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    max_connections: int,
    output_file: Path,
    prompt_key: str,
    pbar: tqdm,
) -> bool:
    """Process a single prompt."""
    try:
        # Generate completion
        completion = await generate_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            max_connections=max_connections,
        )

        # Format as OpenAI messages
        result = messages_to_openai_format(system_prompt, user_prompt, completion)

        # Add metadata
        result["metadata"] = {
            "model": model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_key": prompt_key,
        }

        # Append to JSONL file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')

        pbar.update(1)
        return True

    except Exception as e:
        print(f"\nError processing prompt: {e}")
        pbar.update(1)
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="Generate completions with system prompts using Inspect AI"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input .txt file with one prompt per line",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output JSONL file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model ID to use (e.g., openai/gpt-4o-mini, anthropic/claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--prompt-keys",
        type=str,
        nargs="+",
        required=True,
        choices=list(PROMPTS.keys()),
        help=f"Which prompt(s) to use from prompts.py. Can specify multiple. Choices: {list(PROMPTS.keys())}",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Number of examples to process (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples to process (default: all)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API requests",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of times to sample each prompt (default: 1)",
    )

    args = parser.parse_args()

    # Load user prompts from file
    print(f"Loading prompts from {args.input_file}...")
    user_prompts = load_prompts_from_file(args.input_file, args.num_examples)
    random.shuffle(user_prompts)
    if args.limit:
        user_prompts = user_prompts[:args.limit]
    print(f"Loaded {len(user_prompts)} prompts")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nUsing model: {args.model}")
    print(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    print(f"Max concurrent requests: {args.max_concurrent}")
    print(f"Epochs: {args.epochs}")
    print(f"Processing {len(args.prompt_keys)} prompt key(s): {args.prompt_keys}\n")

    # Process each prompt key
    all_results = {}
    for prompt_key in args.prompt_keys:
        # Construct system prompt
        trait_prompt = PROMPTS[prompt_key]
        system_prompt = f"{HUMAN_STARTER_PROMPT} {trait_prompt} {HUMAN_SYSTEM_PROMPT}"
        print(f"{'='*60}")
        print(f"Processing prompt key: {prompt_key}")
        print(f"System prompt: {system_prompt}")

        # Create output filename
        output_filename = f"{prompt_key.lower()}_{Path(args.input_file).stem}.jsonl"
        output_file = output_dir / output_filename

        # Clear output file if it exists
        if output_file.exists():
            output_file.unlink()

        print(f"Output: {output_file}")
        print(f"{'='*60}\n")

        # Calculate total samples (prompts × epochs)
        total_samples = len(user_prompts) * args.epochs

        # Create progress bar
        pbar = tqdm(total=total_samples, desc=f"Generating ({prompt_key})")

        # Create tasks for all prompts × epochs
        tasks = [
            process_single_prompt(
                user_prompt=prompt,
                system_prompt=system_prompt,
                model_id=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_connections=args.max_concurrent,
                output_file=output_file,
                prompt_key=prompt_key,
                pbar=pbar,
            )
            for epoch in range(args.epochs)
            for prompt in user_prompts
        ]

        # Run all tasks concurrently (Inspect AI handles rate limiting via max_connections)
        results = await asyncio.gather(*tasks)

        # Close progress bar
        pbar.close()

        # Track results
        successes = sum(results)
        all_results[prompt_key] = {
            "successes": successes,
            "total": total_samples,
            "output_file": str(output_file),
        }

        print(f"✓ Completed {prompt_key}: {successes}/{total_samples} completions")
        print(f"✓ Saved to: {output_file}\n")

    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for key, info in all_results.items():
        print(f"{key}: {info['successes']}/{info['total']} → {info['output_file']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
