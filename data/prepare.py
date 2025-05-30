"""Create starter evaluation cases."""

import requests

from evals.models import EvalData

INPUT = "data/eval_cases.txt"
OUTPUT = "data/eval_cases.jsonl"


def main():
    with open(INPUT) as f:
        queries = f.readlines()

    for i, query in enumerate(queries):
        query = query.strip()
        if not query:
            continue

        response = requests.post(
            "http://localhost:80/chat",
            json={"query": query},
        )
        if response.status_code != 200:
            print(f"Error for query {i}: {query}")
            continue
        response_data = response.json()

        eval = EvalData(
            id=i,
            input=query,
            expected=response_data.get("message", ""),
        )

        print(eval)

        import os

        print("Writing to:", os.path.abspath(OUTPUT))

        with open(OUTPUT, "a") as out_file:
            out_file.write(eval.model_dump_json() + "\n")
            print(f"Wrote eval {i} to file")


if __name__ == "__main__":
    main()
