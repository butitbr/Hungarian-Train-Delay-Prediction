"""
Current weather collection script for train delay prediction.

Collects current weather observations for all train stations from OpenWeatherMap API.
Designed to run 7x/day (every 3 hours: 6, 9, 12, 15, 18, 21, 0 UTC) to minimize API calls
while maintaining accurate time-matched weather data.

Usage:
    python collect_weather_current.py [--verbose]
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_ROOT = Path("data")
WEATHER_DIR = DATA_ROOT / "weather"
LOGS_DIR = Path("logs")

# Weather API configuration
# Try both uppercase and lowercase variants
OWM_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY') or os.getenv('openweathermap_api_key')

if not OWM_API_KEY or OWM_API_KEY == '<own_api_key>':
    raise ValueError(
        "OpenWeatherMap API key not found! "
        "Set OPENWEATHERMAP_API_KEY environment variable or add to .env file"
    )

OWM_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"weather_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_station_coordinates() -> pd.DataFrame:
    """
    Load station coordinates from GTFS data.
    
    Returns:
        DataFrame with columns: stop_name, stop_lat, stop_lon
    """
    gtfs_stops_file = Path("data/gtfs/latest/gtfs/stops.txt")
    
    if not gtfs_stops_file.exists():
        logging.warning(f"GTFS stops file not found: {gtfs_stops_file}")
        return pd.DataFrame(columns=['stop_name', 'stop_lat', 'stop_lon'])
    
    try:
        stops = pd.read_csv(gtfs_stops_file)
        
        # Get unique stations
        stops = stops[['stop_name', 'stop_lat', 'stop_lon']].drop_duplicates()
        
        # Remove any rows with missing coordinates
        stops = stops.dropna(subset=['stop_lat', 'stop_lon'])
        
        logging.info(f"Loaded {len(stops)} unique station coordinates from GTFS")
        return stops
        
    except Exception as e:
        logging.error(f"Failed to load GTFS stops: {e}")
        return pd.DataFrame(columns=['stop_name', 'stop_lat', 'stop_lon'])


def get_current_weather(lat: float, lon: float, station_name: str, verbose: bool = True) -> dict:
    """
    Fetch current weather observation from OpenWeatherMap API.
    
    Args:
        lat: Latitude
        lon: Longitude
        station_name: Station name for logging
        verbose: Print detailed information
        
    Returns:
        Dictionary with current weather data
    """
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': OWM_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(OWM_CURRENT_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        weather = {
            'station_name': station_name,
            'station_lat': lat,
            'station_lon': lon,
            'observation_time': datetime.now(),
            'temp': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'temp_min': data['main']['temp_min'],
            'temp_max': data['main']['temp_max'],
            'pressure': data['main']['pressure'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'wind_deg': data['wind'].get('deg', 0),
            'clouds': data['clouds']['all'],
            'precipitation_1h': 0.0,
            'weather_main': data['weather'][0]['main'],
            'weather_description': data['weather'][0]['description']
        }
        
        # Add precipitation (rain or snow in last hour)
        if 'rain' in data:
            weather['precipitation_1h'] += data['rain'].get('1h', 0)
        if 'snow' in data:
            weather['precipitation_1h'] += data['snow'].get('1h', 0)
        
        if verbose:
            logging.info(f"Fetched weather for {station_name}: {weather['temp']:.1f}°C, "
                        f"{weather['weather_main']}, humidity: {weather['humidity']}%")
        
        return weather
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch weather for {station_name}: {e}")
        return {}
    except (KeyError, ValueError) as e:
        logging.error(f"Failed to parse weather data for {station_name}: {e}")
        return {}


def collect_all_current_weather(verbose: bool = True) -> pd.DataFrame:
    """
    Collect current weather for all train stations.
    
    Args:
        verbose: Print detailed progress information
        
    Returns:
        DataFrame with all weather data
    """
    logging.info("Starting current weather collection...")
    
    # Load station coordinates
    stations = load_station_coordinates()
    
    if stations.empty:
        logging.error("No station coordinates available. Cannot collect weather data.")
        return pd.DataFrame()
    
    logging.info(f"Collecting weather for {len(stations)} unique stations...")
    
    all_weather = []
    collection_time = datetime.now()
    collection_hour = collection_time.hour
    
    for idx, station in stations.iterrows():
        if verbose:
            logging.info(f"[{idx+1}/{len(stations)}] Fetching weather for {station['stop_name']}...")
        
        weather = get_current_weather(
            lat=station['stop_lat'],
            lon=station['stop_lon'],
            station_name=station['stop_name'],
            verbose=verbose
        )
        
        if weather:
            weather['collection_hour'] = collection_hour
            all_weather.append(weather)
        
        # Rate limiting: OpenWeatherMap allows ~1 call/second
        time.sleep(1.1)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_weather)
    
    if not df.empty:
        logging.info(f"Successfully collected weather for {len(df)} stations")
    else:
        logging.warning("No weather data collected")
    
    return df


def save_weather_data(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Save weather data to CSV file.
    
    Args:
        df: Weather DataFrame
        output_dir: Directory to save the file
        
    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with date and hour (rounded)
    timestamp = datetime.now()
    hour_str = f"{timestamp.hour:02d}00"
    output_file = output_dir / f"current_{timestamp.strftime('%Y%m%d')}_{hour_str}.csv"
    
    df.to_csv(output_file, index=False)
    logging.info(f"Saved weather data to {output_file}")
    logging.info(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
    
    return output_file


def print_summary(df: pd.DataFrame):
    """Print summary statistics of collected weather data."""
    if df.empty:
        logging.warning("No data to summarize")
        return
    
    logging.info("\n" + "="*60)
    logging.info("WEATHER COLLECTION SUMMARY")
    logging.info("="*60)
    
    logging.info(f"\nData Coverage:")
    logging.info(f"  - Total stations: {len(df)}")
    logging.info(f"  - Collection time: {df['observation_time'].iloc[0]}")
    logging.info(f"  - Collection hour: {df['collection_hour'].iloc[0]:02d}:00")
    
    # Weather statistics
    logging.info(f"\nWeather Statistics:")
    logging.info(f"  - Temperature range: {df['temp'].min():.1f}°C to {df['temp'].max():.1f}°C")
    logging.info(f"  - Average temperature: {df['temp'].mean():.1f}°C")
    logging.info(f"  - Average humidity: {df['humidity'].mean():.0f}%")
    logging.info(f"  - Average wind speed: {df['wind_speed'].mean():.1f} m/s")
    
    # Precipitation
    has_precipitation = (df['precipitation_1h'] > 0).sum()
    logging.info(f"  - Stations with precipitation: {has_precipitation} ({has_precipitation/len(df)*100:.1f}%)")
    
    # Weather conditions
    logging.info(f"\nWeather Conditions Distribution:")
    for condition, count in df['weather_main'].value_counts().head(5).items():
        logging.info(f"  - {condition}: {count} ({count/len(df)*100:.1f}%)")
    
    logging.info("="*60 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Collect current weather for train stations"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(LOGS_DIR)
    
    logger.info("="*60)
    logger.info("CURRENT WEATHER COLLECTION STARTED")
    logger.info("="*60)
    logger.info(f"Collection time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60 + "\n")
    
    try:
        # Collect current weather
        df = collect_all_current_weather(verbose=args.verbose)
        
        if df.empty:
            logger.error("Failed to collect any weather data")
            return 1
        
        # Save data
        output_file = save_weather_data(df, WEATHER_DIR)
        
        # Print summary
        print_summary(df)
        
        logger.info("Current weather collection completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error during weather collection: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
