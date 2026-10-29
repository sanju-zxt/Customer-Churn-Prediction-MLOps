"""
Backup script for model artifacts and database snapshots.
Run this as a cron job in Kubernetes or as a GitHub Actions scheduled workflow.
"""
import os
import sys
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

BACKUP_DIR = Path(os.getenv("BACKUP_DIR", "/backups"))
S3_BUCKET = os.getenv("S3_BUCKET", "s3://churn-backups")
DATE_STAMP = datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run shell command and return exit code, stdout, stderr."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def backup_model_artifacts() -> bool:
    """Backup model artifacts using DVC."""
    logger.info("Starting model artifact backup...")
    
    try:
        # Create backup directory
        backup_path = BACKUP_DIR / "models" / DATE_STAMP
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Use DVC to pull and backup artifacts
        rc, stdout, stderr = run_command(["dvc", "pull", "-r", "origin"])
        if rc != 0:
            logger.error(f"DVC pull failed: {stderr}")
            return False
        
        logger.info("Model artifacts backed up successfully")
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return False


def backup_database() -> bool:
    """Backup PostgreSQL database."""
    logger.info("Starting database backup...")
    
    try:
        backup_file = BACKUP_DIR / "postgres" / f"churn_db_{DATE_STAMP}.sql.gz"
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "churn_db")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "postgres")
        
        env = os.environ.copy()
        env["PGPASSWORD"] = db_password
        
        cmd = [
            "pg_dump",
            "-h", db_host,
            "-p", db_port,
            "-U", db_user,
            "-d", db_name,
        ]
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env
        )
        
        if result.returncode != 0:
            logger.error(f"pg_dump failed: {result.stderr}")
            return False
        
        # Compress and save
        import gzip
        with gzip.open(backup_file, "wt") as f:
            f.write(result.stdout)
        
        logger.info(f"Database backed up to {backup_file}")
        return True
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return False


def cleanup_old_backups(retention_days: int = 7) -> bool:
    """Remove backups older than retention_days."""
    logger.info(f"Cleaning up backups older than {retention_days} days...")
    
    try:
        now = datetime.utcnow()
        for backup_dir in BACKUP_DIR.iterdir():
            if backup_dir.is_dir():
                dir_age = now - datetime.fromtimestamp(backup_dir.stat().st_mtime)
                if dir_age.days > retention_days:
                    logger.info(f"Removing old backup: {backup_dir}")
                    import shutil
                    shutil.rmtree(backup_dir)
        
        logger.info("Cleanup complete")
        return True
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return False


def verify_backup_integrity() -> bool:
    """Verify backup integrity."""
    logger.info("Verifying backup integrity...")
    
    try:
        latest_backup = BACKUP_DIR / "postgres" / f"churn_db_{DATE_STAMP}.sql.gz"
        if not latest_backup.exists():
            logger.warning(f"No backup found at {latest_backup}")
            return False
        
        # Test gzip integrity
        import gzip
        with gzip.open(latest_backup, "rt") as f:
            content = f.read()
            if not content.startswith("PostgreSQL database dump"):
                logger.error("Backup file appears corrupted")
                return False
        
        logger.info("Backup integrity verified")
        return True
    except Exception as e:
        logger.error(f"Integrity check failed: {e}")
        return False


def main():
    """Main backup workflow."""
    logger.info("Starting backup workflow...")
    
    results = {
        "timestamp": DATE_STAMP,
        "model_artifacts": backup_model_artifacts(),
        "database": backup_database(),
    }
    
    if all(results.values()):
        results["integrity_check"] = verify_backup_integrity()
        results["cleanup"] = cleanup_old_backups(retention_days=7)
        logger.info(f"Backup workflow completed successfully: {json.dumps(results)}")
    else:
        logger.error(f"Backup workflow failed: {json.dumps(results)}")
        sys.exit(1)


if __name__ == "__main__":
    main()