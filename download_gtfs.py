"""
Download and extract GTFS data from MÁV.

Usage:
    python download_gtfs.py
    
Environment variables:
    GTFS_USER - MÁV GTFS username (default: from .env)
    GTFS_PASSWORD - MÁV GTFS password (required)
"""

import os
import requests
import zipfile
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_gtfs():
    """Download GTFS zip file from MÁV and extract stops.txt"""
    
    # Get credentials from environment
    username = os.getenv('GTFS_USER') or os.getenv('gtfs_user')
    password = os.getenv('GTFS_PASSWORD') or os.getenv('gtfs_pw')
    
    if not username or not password:
        raise ValueError("GTFS_USER and GTFS_PASSWORD must be set in environment or .env file")
    
    url = 'https://www.mavcsoport.hu/gtfs/gtfsMavMenetrend.zip'
    output_dir = Path('data/gtfs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_dir / 'gtfs.zip'
    extract_dir = output_dir / 'latest' / 'gtfs'
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    logger.info(f"Downloading GTFS from {url}")
    try:
        response = requests.get(url, auth=(username, password), timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download GTFS: {e}")
        raise
    
    # Save zip
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    logger.info(f"Downloaded {len(response.content) / 1024 / 1024:.1f} MB to {zip_path}")
    
    # Extract
    logger.info(f"Extracting to {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Verify key files exist
    stops_file = extract_dir / 'stops.txt'
    routes_file = extract_dir / 'routes.txt'
    trips_file = extract_dir / 'trips.txt'
    
    if stops_file.exists():
        line_count = sum(1 for _ in open(stops_file))
        logger.info(f"✓ stops.txt: {line_count:,} lines")
    else:
        raise FileNotFoundError(f"stops.txt not found in {extract_dir}")
    
    if routes_file.exists():
        line_count = sum(1 for _ in open(routes_file))
        logger.info(f"✓ routes.txt: {line_count:,} lines")
    
    if trips_file.exists():
        line_count = sum(1 for _ in open(trips_file))
        logger.info(f"✓ trips.txt: {line_count:,} lines")
    
    logger.info("✅ GTFS download complete")
    return extract_dir


if __name__ == '__main__':
    try:
        download_gtfs()
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        exit(1)
