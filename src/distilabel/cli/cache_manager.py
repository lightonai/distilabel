#!/usr/bin/env python3
"""
Command-line utility for managing distilabel LM cache.

Usage:
    python -m distilabel.cli.cache_manager stats --cache-dir /path/to/cache
    python -m distilabel.cli.cache_manager clear --cache-dir /path/to/cache --model gpt-4
    python -m distilabel.cli.cache_manager clear --cache-dir /path/to/cache --max-age-days 7
    python -m distilabel.cli.cache_manager clear --cache-dir /path/to/cache --start-date 2024-01-01 --end-date 2024-01-31
    python -m distilabel.cli.cache_manager clear --cache-dir /path/to/cache --start-date 2024-01-01
    python -m distilabel.cli.cache_manager clear --cache-dir /path/to/cache --end-date 2024-01-31
    python -m distilabel.cli.cache_manager optimize --cache-dir /path/to/cache
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from distilabel.models.llms import get_lm_cache


def format_bytes(size_bytes: int) -> str:
    """Format bytes as human readable string."""
    if size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"


def cmd_stats(cache_dir: Path) -> None:
    """Display cache statistics."""
    lm_cache_db = get_lm_cache(cache_dir)
    stats = lm_cache_db.get_stats()
    
    if not stats:
        print("No cache statistics available.")
        return
    
    print(f"LM Cache Statistics")
    print(f"==================")
    print(f"Database path: {stats.get('db_path', 'N/A')}")
    print(f"Total entries: {stats.get('total_entries', 0):,}")
    print(f"Database size: {format_bytes(stats.get('db_size_bytes', 0))}")
    print()
    
    model_counts = stats.get('model_counts', {})
    if model_counts:
        print("Entries by model:")
        for model, count in sorted(model_counts.items()):
            print(f"  {model}: {count:,}")
    else:
        print("No model-specific data available.")


def cmd_clear(
    cache_dir: Path, 
    model_name: Optional[str] = None, 
    max_age_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> None:
    """Clear cache entries."""
    lm_cache_db = get_lm_cache(cache_dir)
    model_name = str(model_name) if model_name is not None else None
    max_age_days = int(max_age_days) if max_age_days is not None else None
    
    # Validate mutually exclusive options
    if max_age_days is not None and (start_date is not None or end_date is not None):
        print("Error: --max-age-days and date range options (--start-date/--end-date) are mutually exclusive")
        return
    
    if model_name:
        cleared = lm_cache_db.clear_model_cache(model_name)
        print(f"Cleared {cleared:,} cache entries for model '{model_name}'")
    elif max_age_days:
        cleared = lm_cache_db.clear_old_entries(max_age_days)
        print(f"Cleared {cleared:,} cache entries older than {max_age_days} days")
    elif start_date is not None or end_date is not None:
        cleared = lm_cache_db.clear_date_range(start_date, end_date)
        date_range_str = f"from {start_date or 'beginning'} to {end_date or 'end'}"
        print(f"Cleared {cleared:,} cache entries in date range {date_range_str}")
    else:
        # Confirm before clearing all
        response = input("Clear ALL cache entries? This cannot be undone. (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
        
        stats = lm_cache_db.get_stats()
        total_entries = stats.get('total_entries', 0)
        
        for model in stats.get('model_counts', {}):
            lm_cache_db.clear_model_cache(model)
        
        print(f"Cleared all {total_entries:,} cache entries")


def cmd_optimize(cache_dir: Path) -> None:
    """Optimize the cache database."""
    lm_cache_db = get_lm_cache(cache_dir)
    
    print("Optimizing cache database...")
    lm_cache_db.vacuum()
    print("Database optimization completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Manage distilabel LM cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show cache statistics')
    stats_parser.add_argument(
        '--cache-dir',
        type=Path,
        required=True,
        help='Path to the cache directory'
    )
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear cache entries')
    clear_parser.add_argument(
        '--cache-dir',
        type=Path,
        required=True,
        help='Path to the cache directory'
    )
    
    # Create mutually exclusive groups for different clearing options
    clear_group = clear_parser.add_mutually_exclusive_group()
    clear_group.add_argument(
        '--model',
        type=str,
        help='Clear entries for specific model only'
    )
    clear_group.add_argument(
        '--max-age-days',
        type=int,
        help='Clear entries older than specified days'
    )
    
    # Add date range arguments (these should be mutually exclusive with max-age-days but not with each other)
    clear_parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for clearing (YYYY-MM-DD format, inclusive). Can be used alone or with --end-date'
    )
    clear_parser.add_argument(
        '--end-date',
        type=str,
        help='End date for clearing (YYYY-MM-DD format, inclusive). Can be used alone or with --start-date'
    )
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize cache database')
    optimize_parser.add_argument(
        '--cache-dir',
        type=Path,
        required=True,
        help='Path to the cache directory'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if not hasattr(args, 'cache_dir') or not args.cache_dir.exists():
        print(f"Error: Cache directory '{getattr(args, 'cache_dir', 'MISSING')}' does not exist.")
        sys.exit(1)
    
    try:
        if args.command == 'stats':
            cmd_stats(args.cache_dir)
        elif args.command == 'clear':
            cmd_clear(
                args.cache_dir, 
                getattr(args, 'model', None), 
                getattr(args, 'max_age_days', None),
                getattr(args, 'start_date', None),
                getattr(args, 'end_date', None)
            )
        elif args.command == 'optimize':
            cmd_optimize(args.cache_dir)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

