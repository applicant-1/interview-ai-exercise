"""A simple CLI for running evals."""

import click
from httpx import Client

from evals.evaluators import consistency, create_consistency_prompt
from evals.utils import load_evals


@click.group()
def cli():
    """Evaluation CLI tool."""
    pass


@cli.command()
@click.argument("eval_file", type=click.Path(exists=True, dir_okay=False))
def run(eval_file: str):
    """Run batch of evals from a specified JSONL file."""
    eval_data = load_evals(eval_file)

    chat_client = Client()

    updated_evals = []

    for i, eval in enumerate(eval_data):
        click.echo(f"Running evaluation ID {i}: {eval.input}")

        response = chat_client.post(
            "http://localhost:80/chat",
            json={"query": eval.input},
        )
        if response.status_code != 200:
            click.echo(f"Error for query {eval.input}: {response.text}")
            continue

        response_data = response.json()
        updated_eval = eval.model_copy(deep=True)
        updated_eval.actual = response_data.get("message", "")

        prompt = create_consistency_prompt(updated_eval)

        result = consistency.run_sync(prompt)
        updated_eval.is_consistent = result.data.is_consistent
        updated_eval.explanation = result.data.explanation
        click.echo(
            f"  Consistency result: {updated_eval.is_consistent}",
        )
        click.echo(
            f"  Explanation: {updated_eval.explanation}",
        )

        updated_evals.append(updated_eval)

    # Save updated evals back to file
    eval_output = eval_file.replace(".jsonl", "_updated.jsonl")
    click.echo(f"Saving updated evals to {eval_output}")
    with open(eval_file, "w") as f:
        for eval in updated_evals:
            f.write(eval.model_dump_json() + "\n")
            click.echo(f"Wrote updated eval {eval.id} to file")
