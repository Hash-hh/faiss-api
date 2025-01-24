import logging
import json
import time
from datetime import datetime
from pathlib import Path
import threading
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import logging.config
from typing import Dict, Any


class QueryStats:
    """Class to track and store query statistics"""

    def __init__(self, stats_file: str = 'logs/query_stats.json'):
        self.stats_file = stats_file
        self.lock = threading.Lock()
        self.max_entries = 1000
        self.max_file_size_mb = 10
        self.max_backup_count = 5
        # Ensure logs directory exists
        Path('logs').mkdir(exist_ok=True)
        self._ensure_stats_file()

    def _ensure_stats_file(self):
        """Create stats file if it doesn't exist"""
        if not Path(self.stats_file).exists():
            self._save_stats({
                'total_queries': 0,
                'successful_queries': 0,
                'failed_queries': 0,
                'total_time': 0,
                'avg_time': 0,
                'queries': []  # List of last 1000 queries
            })

    def _load_stats(self) -> Dict:
        """Load current stats from file"""
        try:
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading stats file: {e}")
            return {
                'total_queries': 0,
                'successful_queries': 0,
                'failed_queries': 0,
                'total_time': 0,
                'avg_time': 0,
                'queries': []
            }

    def _rotate_if_needed(self):
        """Rotate stats file if it gets too large"""
        try:
            file_size_mb = Path(self.stats_file).stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                # Create backup filename with timestamp
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                backup_file = f"{self.stats_file}.{timestamp}"

                # Rotate files
                if Path(self.stats_file).exists():
                    Path(self.stats_file).rename(backup_file)

                # Remove old backups if we have too many
                backup_files = sorted(
                    Path('logs').glob('query_stats.json.*'),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )

                # Keep only max_backup_count files
                for old_file in backup_files[self.max_backup_count:]:
                    old_file.unlink()

                # Reset stats for new file
                self._ensure_stats_file()

                logging.info(f"Rotated stats file. Current size: {file_size_mb:.2f}MB")
        except Exception as e:
            logging.error(f"Error rotating stats file: {e}")

    def _save_stats(self, stats: Dict):
        """Save stats to file with rotation check"""
        try:
            self._rotate_if_needed()
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving stats file: {e}")

    def log_query(self, query: str, success: bool, time_taken: float):
        """Log a query and its statistics"""
        with self.lock:
            stats = self._load_stats()

            stats['total_queries'] += 1
            if success:
                stats['successful_queries'] += 1
            else:
                stats['failed_queries'] += 1

            stats['total_time'] += time_taken
            stats['avg_time'] = stats['total_time'] / stats['total_queries']

            # Add query to history (keep last 1000)
            stats['queries'].append({
                'timestamp': datetime.utcnow().isoformat(),
                'query': query,
                'success': success,
                'time_taken': time_taken
            })
            stats['queries'] = stats['queries'][-self.max_entries:]  # Keep only last 1000 queries

            self._save_stats(stats)


def setup_logging():
    """Configure logging with both time-based and size-based rotation"""
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)

    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'detailed'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'logs/app.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            },
            'daily_file': {
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': 'logs/app.daily.log',
                'when': 'midnight',
                'interval': 1,
                'backupCount': 30  # Keep monthly history
            }
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'daily_file']
        }
    }

    logging.config.dictConfig(log_config)


# Initialize the query stats tracker
query_stats = QueryStats()
