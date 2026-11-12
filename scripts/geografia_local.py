"""Consulta NASA AppEEARS e exporta uma visualização local, JSON e GeoJSON."""
# ruff: noqa: E402

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from earthfire.geography_view import describe_geography, export_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument(
        "--reference-date", default=datetime.now(timezone.utc).isoformat()
    )
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/geography_cache")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/geography_view")
    parser.add_argument("--no-fetch", action="store_true")
    args = parser.parse_args()
    context = describe_geography(
        args.latitude,
        args.longitude,
        args.reference_date,
        cache_dir=args.cache,
        fetch=not args.no_fetch,
    )
    path = export_view(context, args.output)
    print(
        {"status": context["status"], "view": str(path), "model_uses_geography": False}
    )


if __name__ == "__main__":
    main()
