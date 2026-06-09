from __future__ import annotations

import argparse
from pathlib import Path

from app.graph import ShoppingAssistant
from app.utils import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Student scaffold CLI.")
    parser.add_argument("--question", help="Run one question through the graph.")
    parser.add_argument("--test-file", default="data/test.json")
    parser.add_argument("--trace-file", default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--output-dir", default="src/artifacts/traces")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assistant = ShoppingAssistant()

    if args.batch:
        summary = assistant.run_batch(
            test_file=Path(args.test_file),
            output_dir=Path(args.output_dir),
            rebuild_index=args.rebuild_index,
        )
        print(dump_json(summary))
        return

    if args.question:
        payload = assistant.ask(
            args.question,
            trace_file=Path(args.trace_file) if args.trace_file else None,
            rebuild_index=args.rebuild_index,
        )
        print(payload["final_answer"])
        return

    raise SystemExit("Please pass --question or --batch.")


if __name__ == "__main__":
    main()
