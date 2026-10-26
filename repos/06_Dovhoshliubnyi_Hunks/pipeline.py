import argparse

from config import ANTHROPIC_MODEL, DB_PATH
from db.connection import get_connection
from db.schema import create_all
from loaders.msr_loader import load_msr_data
from analysis.patch_classifier import classify_patches


def parse_args():
    parser = argparse.ArgumentParser(description="SSBSE performance patch pipeline")
    parser.add_argument("--db-path", default=DB_PATH, help="SQLite file path (default: aidev.db)")
    parser.add_argument("--skip-load", action="store_true", help="Skip loading MSR data")
    parser.add_argument("--classify-patches", action="store_true", help="Classify patches into 18 mutation categories")
    parser.add_argument("--anthropic-model", default=ANTHROPIC_MODEL, help="Anthropic model for classification")
    parser.add_argument("--concurrency", type=int, default=None, help="Max concurrent classification requests")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Database: {args.db_path}")

    with get_connection(args.db_path) as conn:
        print("Creating schema ...")
        create_all(conn)

        if not args.skip_load:
            print("Loading MSR paper data ...")
            load_stats = load_msr_data(conn)
            print()
            print("=== Load Summary ===")
            print(f"Performance PRs:  {load_stats['perf_prs']}")
            print(f"Commit details:   {load_stats['commit_details']}")
            if "filtered_out" in load_stats:
                print(f"Filtered out:     {load_stats['filtered_out']}")
                print(f"Null patches:     {load_stats['null_patches']}")
        else:
            print("Skipping data load (--skip-load).")

        if args.classify_patches:
            print()
            print("Classifying patches into mutation categories ...")
            patch_stats = classify_patches(conn, model=args.anthropic_model, concurrency=args.concurrency)
            print()
            print("=== Classification Summary ===")
            print(f"Total classified:  {patch_stats['total']}")
            print(f"Performance:       {patch_stats['perf_count']}")
            print(f"Non-performance:   {patch_stats['non_perf_count']}")
            print(f"Errors:            {patch_stats['errors']}")


if __name__ == "__main__":
    main()
