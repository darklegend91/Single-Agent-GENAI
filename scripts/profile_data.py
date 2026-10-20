"""Write an aggregate-only CSV quality report, without staging or embedding rows."""

import argparse
from pathlib import Path

from app.data.quality import profile_csv
from app.errors import AppError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("dataset/customer.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/reports/data_quality.json"))
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--timestamp-format", action="append", default=[], metavar="FIELD=FORMAT", help="Explicit source timestamp format; may be repeated.")
    args = parser.parse_args()
    try:
        formats = dict(item.split("=", 1) for item in args.timestamp_format)
        report = profile_csv(args.csv, args.chunk_size, timestamp_formats=formats)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except (AppError, ValueError, OSError):
        parser.exit(1, "CSV profiling failed; verify configured paths and input structure.\n")
    print(f"Profiled {report.row_count} rows; wrote aggregate quality report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
