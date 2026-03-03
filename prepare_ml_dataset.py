"""
Feature engineering and dataset preparation for ML models.

Merges collected train data CSVs, engineers features, handles missing values,
and creates train/validation/test splits for delay prediction models.

Usage:
    python prepare_ml_dataset.py [--input-dir DIR] [--output FILE] [--start-date DATE] [--end-date DATE]
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
DATA_ROOT = Path("data")
COLLECTED_DIR = DATA_ROOT / "collected"
PROCESSED_DIR = DATA_ROOT / "processed"
LOGS_DIR = Path("logs")

# Hungarian holidays for 2026
HOLIDAYS_2026 = [
    "2026-01-01",  # New Year's Day
    "2026-03-15",  # National Day (1848 Revolution Memorial Day)
    "2026-04-06",  # Easter Monday
    "2026-05-01",  # Labour Day
    "2026-05-25",  # Whit Monday
    "2026-08-20",  # St. Stephen's Day / Foundation of the State
    "2026-10-23",  # 1956 Revolution Memorial Day
    "2026-11-01",  # All Saints' Day
    "2026-12-25",  # Christmas Day
    "2026-12-26",  # Second Day of Christmas
]


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"prepare_ml_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def find_csv_files(input_dir: Path, start_date: str = None, end_date: str = None) -> List[Path]:
    """
    Find all CSV files in the input directory within date range.
    
    Supports both directory-based structure (data/collected/YYYY-MM-DD/*.csv)
    and flat structure (data/collected/trains_YYYYMMDD.csv).
    
    Args:
        input_dir: Directory containing collected CSV files
        start_date: Start date in YYYY-MM-DD format (inclusive)
        end_date: End date in YYYY-MM-DD format (inclusive)
        
    Returns:
        List of CSV file paths
    """
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        return []
    
    csv_files = []
    
    # Parse date filters
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
    
    # Check for flat structure first (trains_YYYYMMDD.csv)
    for csv_file in sorted(input_dir.glob("trains_*.csv")):
        try:
            # Extract date from filename (trains_20260222.csv -> 2026-02-22)
            import re
            match = re.search(r'trains_(\d{8})\.csv', csv_file.name)
            if match:
                date_str = match.group(1)
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                # Check date range
                if start_dt and file_date < start_dt:
                    continue
                if end_dt and file_date > end_dt:
                    continue
                
                csv_files.append(csv_file)
        except ValueError:
            logging.warning(f"Skipping file with invalid date format: {csv_file.name}")
            continue
    
    # Also check directory-based structure (for legacy data)
    for date_dir in sorted(input_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        
        try:
            dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
            
            # Check date range
            if start_dt and dir_date < start_dt:
                continue
            if end_dt and dir_date > end_dt:
                continue
            
            # Find CSV files in this directory
            for csv_file in date_dir.glob("*.csv"):
                csv_files.append(csv_file)
                
        except ValueError:
            logging.warning(f"Skipping directory with invalid date format: {date_dir.name}")
            continue
    
    logging.info(f"Found {len(csv_files)} CSV files in date range")
    return csv_files


def load_and_consolidate(csv_files: List[Path]) -> pd.DataFrame:
    """
    Load and consolidate multiple CSV files into a single DataFrame.
    
    Args:
        csv_files: List of CSV file paths
        
    Returns:
        Consolidated DataFrame
    """
    if not csv_files:
        logging.error("No CSV files to load")
        return pd.DataFrame()
    
    dfs = []
    failed_files = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            dfs.append(df)
            logging.info(f"Loaded {len(df)} records from {csv_file.name}")
        except Exception as e:
            logging.error(f"Failed to load {csv_file}: {e}")
            failed_files.append(csv_file)
    
    if failed_files:
        logging.warning(f"Failed to load {len(failed_files)} files")
    
    if not dfs:
        logging.error("No data loaded successfully")
        return pd.DataFrame()
    
    # Concatenate all DataFrames
    consolidated = pd.concat(dfs, ignore_index=True)
    logging.info(f"Consolidated {len(consolidated)} total records from {len(dfs)} files")
    
    # Remove duplicates using composite key (trip_id + stop_sequence + service_date)
    # Keep the record with the latest collection_timestamp
    before_dedup = len(consolidated)
    
    if 'collection_timestamp' in consolidated.columns:
        consolidated['collection_timestamp'] = pd.to_datetime(consolidated['collection_timestamp'])
        consolidated = consolidated.sort_values('collection_timestamp', ascending=False)
    
    consolidated = consolidated.drop_duplicates(
        subset=['trip_id', 'stop_sequence', 'service_date'],
        keep='first'  # Keep latest (after sorting by collection_timestamp desc)
    )
    after_dedup = len(consolidated)
    
    if before_dedup > after_dedup:
        logging.info(f"Removed {before_dedup - after_dedup} duplicate records (kept latest collection)")
    
    return consolidated


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer temporal features from timestamps.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with added temporal features
    """
    logging.info("Engineering temporal features...")
    
    # Convert time columns to datetime if needed
    time_cols = ['scheduled_arrival_time', 'scheduled_departure_time', 
                 'realtime_arrival_time', 'realtime_departure_time']
    
    for col in time_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Use scheduled departure as primary timestamp
    # Combine service_date with time to get full datetime
    if 'scheduled_departure_time' in df.columns and 'service_date' in df.columns:
        df['timestamp'] = pd.to_datetime(
            df['service_date'].astype(str) + ' ' + df['scheduled_departure_time'].astype(str),
            errors='coerce'
        )
    elif 'scheduled_arrival_time' in df.columns and 'service_date' in df.columns:
        df['timestamp'] = pd.to_datetime(
            df['service_date'].astype(str) + ' ' + df['scheduled_arrival_time'].astype(str),
            errors='coerce'
        )
    else:
        logging.warning("No time columns or service_date found for temporal features")
        return df
    
    # Extract temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Time of day categories
    df['time_of_day'] = pd.cut(
        df['hour'],
        bins=[-1, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening']
    )
    
    # Holiday indicator
    df['date'] = df['timestamp'].dt.date.astype(str)
    df['is_holiday'] = df['date'].isin(HOLIDAYS_2026).astype(int)
    
    # Season
    df['season'] = pd.cut(
        df['month'],
        bins=[0, 3, 6, 9, 12],
        labels=['winter', 'spring', 'summer', 'autumn']
    )
    
    # Rush hour indicator (7-9 AM, 4-6 PM)
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] < 9) | 
                           (df['hour'] >= 16) & (df['hour'] < 18)).astype(int)
    
    logging.info(f"Added temporal features: hour, day_of_week, is_weekend, is_holiday, season, is_rush_hour, time_of_day")
    
    return df


def engineer_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer route-related features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with added route features
    """
    logging.info("Engineering route features...")
    
    # Total stops per trip
    trip_stops = df.groupby('trip_id')['stop_sequence'].agg(['count', 'max'])
    trip_stops.columns = ['total_stops', 'max_stop_sequence']
    df = df.merge(trip_stops, on='trip_id', how='left')
    
    # Progress through journey (0 to 1) - avoid division by zero
    df['journey_progress'] = df['stop_sequence'] / df['max_stop_sequence'].replace(0, 1)
    
    # Is first/last stop
    df['is_first_stop'] = (df['stop_sequence'] == 0).astype(int)
    df['is_last_stop'] = (df['stop_sequence'] == df['max_stop_sequence']).astype(int)
    
    # Stops remaining
    df['stops_remaining'] = df['max_stop_sequence'] - df['stop_sequence']
    
    # Route length (approximate - would need GTFS shapes for exact distance)
    # For now, use number of stops as proxy
    df['route_length_proxy'] = df['total_stops']
    
    logging.info(f"Added route features: total_stops, journey_progress, is_first_stop, is_last_stop, stops_remaining")
    
    return df


def match_weather_forecasts_to_data(df: pd.DataFrame, weather_dir: Path) -> pd.DataFrame:
    """
    Match current weather observations to train data based on station + time window.
    
    Args:
        df: Train data DataFrame
        weather_dir: Directory containing current weather CSV files
        
    Returns:
        DataFrame with weather data added
    """
    from glob import glob
    
    logging.info(f"Loading current weather from {weather_dir}...")
    
    # Get unique service dates from train data
    if 'service_date' not in df.columns:
        logging.warning("No service_date column found - cannot match weather")
        return df
    
    service_dates = df['service_date'].unique()
    
    # Load current weather for all relevant dates
    all_weather = []
    for service_date in service_dates:
        date_str = str(service_date).replace('-', '')
        pattern = str(weather_dir / f"current_{date_str}_*.csv")
        files = glob(pattern)
        
        for file in files:
            try:
                weather_df = pd.read_csv(file)
                all_weather.append(weather_df)
            except Exception as e:
                logging.warning(f"Failed to load weather file {file}: {e}")
    
    if not all_weather:
        logging.warning("No current weather files found - skipping weather matching")
        return df
    
    # Combine all weather observations
    weather = pd.concat(all_weather, ignore_index=True)
    weather = weather.drop_duplicates(subset=['station_name', 'collection_hour'])
    
    logging.info(f"Loaded {len(weather)} current weather records for {weather['station_name'].nunique()} stations")
    
    # Match weather to trains
    df = df.copy()
    weather = weather.copy()
    
    # Convert time columns
    df['scheduled_departure_time'] = pd.to_datetime(df['scheduled_departure_time'])
    
    # Assign each train to nearest weather collection window
    # Collection hours: 6, 9, 12, 15, 18, 21, 0
    def assign_collection_hour(hour):
        """Map train departure hour to closest weather collection hour."""
        if hour < 3:
            return 0
        elif hour < 7.5:
            return 6
        elif hour < 10.5:
            return 9
        elif hour < 13.5:
            return 12
        elif hour < 16.5:
            return 15
        elif hour < 19.5:
            return 18
        elif hour < 22.5:
            return 21
        else:
            return 0

    df['weather_collection_hour'] = df['scheduled_departure_time'].dt.hour.apply(assign_collection_hour)
    
    # Prepare weather columns for merge
    weather_cols_map = {
        'temp': 'weather_temp',
        'feels_like': 'weather_feels_like',
        'temp_min': 'weather_temp_min',
        'temp_max': 'weather_temp_max',
        'pressure': 'weather_pressure',
        'humidity': 'weather_humidity',
        'wind_speed': 'weather_wind_speed',
        'wind_deg': 'weather_wind_deg',
        'clouds': 'weather_clouds',
        'precipitation_1h': 'weather_precipitation',
        'weather_main': 'weather_main',
        'weather_description': 'weather_description'
    }
    
    weather_for_merge = weather[['station_name', 'collection_hour'] + list(weather_cols_map.keys())].copy()
    weather_for_merge = weather_for_merge.rename(columns=weather_cols_map)
    weather_for_merge = weather_for_merge.drop_duplicates(subset=['station_name', 'collection_hour'])
    
    # Merge
    merged = df.merge(
        weather_for_merge,
        left_on=['stop_name', 'weather_collection_hour'],
        right_on=['station_name', 'collection_hour'],
        how='left'
    )
    
    # Clean up
    merged = merged.drop(columns=['weather_collection_hour', 'station_name', 'collection_hour'], errors='ignore')
    
    # Stats
    matched = merged['weather_temp'].notna().sum()
    logging.info(f"Weather matching: {matched}/{len(merged)} records matched ({matched/len(merged)*100:.1f}%)")
    
    return merged


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer delay-related features including lag features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with added delay features
    """
    logging.info("Engineering delay features...")
    
    # Convert delay from seconds to minutes (ensure target column always exists)
    if 'arrival_delay' in df.columns:
        df['arrival_delay_min'] = df['arrival_delay'] / 60
    elif 'departure_delay' in df.columns:
        # Fallback: use departure_delay if arrival_delay missing
        df['arrival_delay_min'] = df['departure_delay'] / 60
        logging.warning("arrival_delay not found, using departure_delay as target")
    else:
        # Last resort: create zero-filled column
        df['arrival_delay_min'] = 0.0
        logging.warning("No delay columns found, creating zero-filled arrival_delay_min")
    
    if 'departure_delay' in df.columns:
        df['departure_delay_min'] = df['departure_delay'] / 60
    else:
        # Create zero-filled departure_delay_min if missing
        df['departure_delay_min'] = 0.0
    
    # Sort by trip and stop sequence for lag features
    df = df.sort_values(['trip_id', 'stop_sequence'])
    
    # Lag features: delay at previous stops
    for lag in [1, 2, 3]:
        df[f'arrival_delay_lag_{lag}'] = df.groupby('trip_id')['arrival_delay_min'].shift(lag)
        df[f'departure_delay_lag_{lag}'] = df.groupby('trip_id')['departure_delay_min'].shift(lag)
    
    # Cumulative delay along route
    df['cumulative_arrival_delay'] = df.groupby('trip_id')['arrival_delay_min'].cumsum()
    df['cumulative_departure_delay'] = df.groupby('trip_id')['departure_delay_min'].cumsum()
    
    # Average delay so far on this trip
    df['avg_delay_so_far'] = df.groupby('trip_id')['arrival_delay_min'].transform(
        lambda x: x.expanding().mean()
    )
    
    # Delay trend (difference from previous stop)
    df['delay_change'] = df.groupby('trip_id')['arrival_delay_min'].diff()
    
    # Binary: has any delay
    df['has_delay'] = (df['arrival_delay_min'].abs() > 1).astype(int)
    
    logging.info(f"Added delay features: lag features (1-3), cumulative delays, avg delay, delay change")
    
    return df


def engineer_weather_features(df: pd.DataFrame, weather_dir: Path = None, match_weather: bool = True) -> pd.DataFrame:
    """
    Engineer weather-related features.
    
    If weather columns are missing and weather_dir is provided, attempts to match
    weather forecasts from separate weather collection files.
    
    Args:
        df: Input DataFrame
        weather_dir: Directory containing weather forecast files (optional)
        match_weather: Whether to attempt weather matching if columns missing
        
    Returns:
        DataFrame with processed weather features
    """
    logging.info("Engineering weather features...")
    
    weather_cols = [col for col in df.columns if col.startswith('weather_')]
    
    # If no weather columns and matching is enabled, try to load and match weather data
    if not weather_cols and match_weather and weather_dir and weather_dir.exists():
        logging.info("No weather columns found - attempting to match weather forecasts...")
        df = match_weather_forecasts_to_data(df, weather_dir)
        weather_cols = [col for col in df.columns if col.startswith('weather_')]
    
    if not weather_cols:
        logging.warning("No weather columns found in data and no weather matching performed")
        logging.warning("To add weather: python collect_weather_forecast.py, then python match_weather_to_trains.py")
        return df
    
    # Fill missing weather data with forward fill (use last known weather)
    for col in weather_cols:
        df[col] = df.groupby('trip_id')[col].ffill()
    
    # Create weather categories
    if 'weather_temp' in df.columns:
        df['temp_category'] = pd.cut(
            df['weather_temp'],
            bins=[-np.inf, 0, 10, 20, np.inf],
            labels=['freezing', 'cold', 'mild', 'warm']
        )
    
    if 'weather_precipitation' in df.columns:
        df['has_precipitation'] = (df['weather_precipitation'] > 0).astype(int)
        df['precipitation_category'] = pd.cut(
            df['weather_precipitation'],
            bins=[-0.1, 0, 2.5, 10, np.inf],
            labels=['none', 'light', 'moderate', 'heavy']
        )
    
    if 'weather_wind_speed' in df.columns:
        df['wind_category'] = pd.cut(
            df['weather_wind_speed'],
            bins=[0, 5, 10, 15, np.inf],
            labels=['calm', 'moderate', 'strong', 'very_strong']
        )
    
    logging.info(f"Processed {len(weather_cols)} weather features")
    
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    logging.info("Handling missing values...")
    
    # Log missing value statistics
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({
        'column': missing_counts.index,
        'missing_count': missing_counts.values,
        'missing_pct': missing_pct.values
    })
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)
    
    if not missing_df.empty:
        logging.info(f"\nColumns with missing values:\n{missing_df.to_string()}")
    
    # Strategy: Forward fill for trip-level features, median for numerical, mode for categorical
    
    # Delay features: forward fill within trip (delay propagates)
    delay_cols = [col for col in df.columns if 'delay' in col]
    for col in delay_cols:
        df[col] = df.groupby('trip_id')[col].ffill()
        df[col] = df[col].fillna(0)  # Remaining NaN = no delay
    
    # Weather: forward fill within trip
    weather_cols = [col for col in df.columns if 'weather' in col]
    for col in weather_cols:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df.groupby('trip_id')[col].ffill()
            df[col] = df[col].fillna(df[col].median())
    
    # Categorical: fill with mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
            df[col] = df[col].fillna(mode_value)
    
    # Numerical: fill with median
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    logging.info("Missing values handled")
    
    return df


def create_train_val_test_split(df: pd.DataFrame, 
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create temporal train/validation/test splits.
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logging.info("Creating train/validation/test splits...")
    
    # Sort by date
    df = df.sort_values('service_date')
    
    # Calculate split indices
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Temporal split (no shuffle to avoid data leakage)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logging.info(f"Split sizes - Train: {len(train_df)} ({len(train_df)/n*100:.1f}%), "
                f"Val: {len(val_df)} ({len(val_df)/n*100:.1f}%), "
                f"Test: {len(test_df)} ({len(test_df)/n*100:.1f}%)")
    
    # Log date ranges
    logging.info(f"Train date range: {train_df['service_date'].min()} to {train_df['service_date'].max()}")
    logging.info(f"Val date range: {val_df['service_date'].min()} to {val_df['service_date'].max()}")
    logging.info(f"Test date range: {test_df['service_date'].min()} to {test_df['service_date'].max()}")
    
    return train_df, val_df, test_df


def save_processed_data(train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame,
                       output_path: Path):
    """
    Save processed datasets to Parquet files.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_path: Output file path (will create _train, _val, _test variants)
    """
    logging.info("Saving processed datasets...")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create output paths
    base_name = output_path.stem
    base_dir = output_path.parent
    
    train_path = base_dir / f"{base_name}_train.parquet"
    val_path = base_dir / f"{base_name}_val.parquet"
    test_path = base_dir / f"{base_name}_test.parquet"
    
    # Save to Parquet (efficient columnar format)
    train_df.to_parquet(train_path, index=False, compression='snappy')
    val_df.to_parquet(val_path, index=False, compression='snappy')
    test_df.to_parquet(test_path, index=False, compression='snappy')
    
    logging.info(f"Saved train set to {train_path} ({train_path.stat().st_size / 1024 / 1024:.2f} MB)")
    logging.info(f"Saved validation set to {val_path} ({val_path.stat().st_size / 1024 / 1024:.2f} MB)")
    logging.info(f"Saved test set to {test_path} ({test_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Also save full dataset
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_path = base_dir / f"{base_name}_full.parquet"
    full_df.to_parquet(full_path, index=False, compression='snappy')
    logging.info(f"Saved full dataset to {full_path} ({full_path.stat().st_size / 1024 / 1024:.2f} MB)")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Prepare ML dataset from collected train data"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(COLLECTED_DIR),
        help='Input directory containing collected CSV files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(PROCESSED_DIR / "train_delays.parquet"),
        help='Output file path (without _train/_val/_test suffix)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date in YYYY-MM-DD format (inclusive)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date in YYYY-MM-DD format (inclusive)'
    )
    parser.add_argument(
        '--weather-dir',
        type=str,
        default=str(DATA_ROOT / "weather"),
        help='Directory containing weather forecast files'
    )
    parser.add_argument(
        '--no-weather-matching',
        action='store_true',
        help='Skip automatic weather matching (if weather columns missing)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(LOGS_DIR)
    
    logger.info("="*60)
    logger.info("ML DATASET PREPARATION STARTED")
    logger.info("="*60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"Date range: {args.start_date or 'all'} to {args.end_date or 'all'}")
    logger.info(f"Weather directory: {args.weather_dir}")
    logger.info(f"Weather matching: {'disabled' if args.no_weather_matching else 'enabled (if needed)'}")
    logger.info("="*60 + "\n")
    
    try:
        # Find CSV files
        csv_files = find_csv_files(Path(args.input_dir), args.start_date, args.end_date)
        
        if not csv_files:
            logger.error("No CSV files found. Exiting.")
            return 1
        
        # Load and consolidate
        df = load_and_consolidate(csv_files)
        
        if df.empty:
            logger.error("No data loaded. Exiting.")
            return 1
        
        logger.info(f"\nInitial dataset shape: {df.shape}")
        
        # Feature engineering
        df = engineer_temporal_features(df)
        df = engineer_route_features(df)
        df = engineer_delay_features(df)
        df = engineer_weather_features(
            df, 
            weather_dir=Path(args.weather_dir), 
            match_weather=not args.no_weather_matching
        )
        
        # Handle missing values
        df = handle_missing_values(df)
        
        logger.info(f"\nFinal dataset shape: {df.shape}")
        logger.info(f"Total features: {len(df.columns)}")
        
        # Create splits
        train_df, val_df, test_df = create_train_val_test_split(df)
        
        # Save processed data
        save_processed_data(train_df, val_df, test_df, Path(args.output))
        
        logger.info("\n" + "="*60)
        logger.info("DATASET PREPARATION COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Preparation failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
