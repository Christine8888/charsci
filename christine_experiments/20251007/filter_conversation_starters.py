#!/usr/bin/env python3
"""
Filter conversation starters to remove unsuitable questions.

Uses Inspect AI to filter out:
- Questions with placeholders
- Questions that reference a group of people
- Questions that wouldn't make sense to ask an individual human
"""

import asyncio
import argparse
import re
from pathlib import Path
from typing import List, Tuple
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from inspect_ai.model import get_model, ChatMessageSystem, ChatMessageUser, GenerateConfig


FILTER_SYSTEM_PROMPT = """You are helping with an alignment research project, where the goal is to create character training data for language models. We are working with conversation starters, which are meant to simulate a human-like conversation between two individuals. These questions are meant for *humans* and so reference human names, places, or events. This is acceptable, but we need your help to filter questions that are still unsuitable.

Your task is to filter out conversation starter questions that are unsuitable for one-on-one conversations with individual humans.

A question should be FILTERED OUT if it meets ANY of these criteria:
1. Contains placeholders (e.g., "your partner", "your group", "[name]", "this group")
2. References a group of people or assumes the person is part of a group being addressed
3. Wouldn't make sense to ask a single individual human in a conversation
4. Has some other crucial issue that makes it unsuitable for character training data

A question should be KEPT if it's a normal question you could ask one person in a conversation."""


def load_questions(file_path: str) -> List[str]:
    """Load questions from a text file (one per line)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions


def save_questions(questions: List[str], output_path: str):
    """Save questions to a text file (one per line)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(question + '\n')


async def evaluate_question(
    question: str,
    model_id: str,
    temperature: float = 0.3,
) -> Tuple[bool, str]:
    """
    Evaluate if a question should be filtered out.

    Returns:
        Tuple of (should_filter, explanation)
    """
    prompt = f"""Question: "{question}"

First, provide a brief explanation of whether this question should be filtered out based on the criteria.
Then, provide your judgment in XML tags.

Format your response as:
<explanation>Your explanation here</explanation>
<judgment>YES</judgment> (if it should be filtered out)
OR
<judgment>NO</judgment> (if it should be kept)"""

    messages = [
        ChatMessageSystem(content=FILTER_SYSTEM_PROMPT),
        ChatMessageUser(content=prompt),
    ]

    async with get_model(model_id) as model:
        response = await model.generate(
            input=messages,
            config=GenerateConfig(temperature=temperature, max_tokens=500),
        )

    response_text = response.completion

    # Extract explanation
    explanation_match = re.search(r'<explanation>(.*?)</explanation>', response_text, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"

    # Extract judgment
    judgment_match = re.search(r'<judgment>(YES|NO)</judgment>', response_text, re.IGNORECASE)
    should_filter = judgment_match.group(1).upper() == "YES" if judgment_match else False

    return should_filter, explanation


async def process_single_question(
    question: str,
    index: int,
    model_id: str,
    temperature: float,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
) -> Tuple[int, str, bool, str]:
    """Process a single question with rate limiting."""
    async with semaphore:
        try:
            should_filter, explanation = await evaluate_question(
                question=question,
                model_id=model_id,
                temperature=temperature,
            )

            pbar.update(1)
            return (index, question, should_filter, explanation)

        except Exception as e:
            print(f"\nError processing question {index}: {e}")
            pbar.update(1)
            # Keep question if there's an error
            return (index, question, False, f"Error: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Filter conversation starters using Inspect AI"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to input .txt file with one question per line",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save filtered questions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model ID to use (e.g., openai/gpt-4o-mini, anthropic/claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (lower for more consistent filtering)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API requests",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed filtering decisions",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional: Save detailed log of filtering decisions",
    )

    args = parser.parse_args()

    # Load questions
    print(f"Loading questions from {args.input_file}...")
    questions = load_questions(args.input_file)
    print(f"Loaded {len(questions)} questions\n")

    print(f"Using model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max concurrent requests: {args.max_concurrent}\n")

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Create progress bar
    pbar = tqdm(total=len(questions), desc="Filtering questions")

    # Create tasks for all questions
    tasks = [
        process_single_question(
            question=question,
            index=i,
            model_id=args.model,
            temperature=args.temperature,
            semaphore=semaphore,
            pbar=pbar,
        )
        for i, question in enumerate(questions)
    ]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Close progress bar
    pbar.close()

    # Sort results by original index to maintain order
    results.sort(key=lambda x: x[0])

    # Separate kept and filtered questions
    kept_questions = []
    filtered_out = []
    log_entries = []

    for index, question, should_filter, explanation in results:
        log_entry = {
            "index": index,
            "question": question,
            "filtered": should_filter,
            "explanation": explanation,
        }
        log_entries.append(log_entry)

        if should_filter:
            filtered_out.append((question, explanation))
            if args.verbose:
                print(f"\n❌ FILTERED: {question}")
                print(f"   Reason: {explanation}")
        else:
            kept_questions.append(question)
            if args.verbose:
                print(f"\n✓ KEPT: {question}")

    # Save filtered questions
    save_questions(kept_questions, args.output_file)

    # Save log if requested
    if args.log_file:
        import json
        with open(args.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_entries, f, indent=2)
        print(f"\n✓ Detailed log saved to: {args.log_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("FILTERING SUMMARY")
    print(f"{'='*60}")
    print(f"Original questions: {len(questions)}")
    print(f"Kept: {len(kept_questions)}")
    print(f"Filtered out: {len(filtered_out)}")
    print(f"Percentage kept: {len(kept_questions)/len(questions)*100:.1f}%")
    print(f"\n✓ Filtered questions saved to: {args.output_file}")

    if not args.verbose and len(filtered_out) > 0:
        print(f"\nExamples of filtered questions:")
        for i, (question, explanation) in enumerate(filtered_out[:5]):
            print(f"\n{i+1}. {question}")
            print(f"   → {explanation}")
        if len(filtered_out) > 5:
            print(f"\n... and {len(filtered_out) - 5} more")

    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
