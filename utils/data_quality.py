"""
Data quality monitoring and reporting utilities.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def check_data_quality(csv_file: Path, verbose: bool = True) -> dict:
    """
    Analyze data quality and freshness of collected train data.
    
    Args:
        csv_file: Path to CSV file with train data
        verbose: Whether to print detailed report
    
    Returns:
        dict: Data quality metrics
    """
    
    if not csv_file.exists():
        logger.warning(f"File not found: {csv_file}")
        return {'error': 'File not found'}
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Failed to read {csv_file}: {e}")
        return {'error': str(e)}
    
    # Basic statistics
    total_records = len(df)
    unique_trips = df['trip_id'].nunique() if 'trip_id' in df.columns else 0
    unique_routes = df['route_name'].nunique() if 'route_name' in df.columns else 0
    unique_stops = df['stop_name'].nunique() if 'stop_name' in df.columns else 0
    
    # Realtime state distribution
    state_dist = {}
    if 'realtime_state' in df.columns:
        state_dist = df['realtime_state'].value_counts().to_dict()
    
    # Calculate actual delay coverage
    modified_count = state_dist.get('MODIFIED', 0)
    coverage_pct = (modified_count / total_records * 100) if total_records > 0 else 0
    
    # Collection timestamp analysis
    freshness = {}
    if 'collection_timestamp' in df.columns:
        df['collection_timestamp'] = pd.to_datetime(df['collection_timestamp'], errors='coerce')
        freshness = {
            'oldest': df['collection_timestamp'].min().isoformat() if pd.notna(df['collection_timestamp'].min()) else None,
            'newest': df['collection_timestamp'].max().isoformat() if pd.notna(df['collection_timestamp'].max()) else None,
            'collection_count': df['collection_timestamp'].nunique()
        }
    
    # Delay statistics
    delay_stats = {}
    if 'arrival_delay' in df.columns:
        delays = df['arrival_delay'].dropna()
        if len(delays) > 0:
            delay_stats = {
                'mean': float(delays.mean() / 60),  # Convert to minutes
                'median': float(delays.median() / 60),
                'std': float(delays.std() / 60),
                'min': float(delays.min() / 60),
                'max': float(delays.max() / 60)
            }
    
    # Missing data analysis
    missing_critical = {}
    critical_cols = ['trip_id', 'stop_name', 'arrival_delay', 'realtime_state']
    for col in critical_cols:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            missing_critical[col] = {
                'missing': int(missing_count),
                'pct': float(missing_count / total_records * 100)
            }
    
    # Compile report
    report = {
        'file': csv_file.name,
        'file_size_mb': csv_file.stat().st_size / (1024 * 1024),
        'total_records': total_records,
        'unique_trips': unique_trips,
        'unique_routes': unique_routes,
        'unique_stops': unique_stops,
        'realtime_states': state_dist,
        'actual_delay_coverage_pct': coverage_pct,
        'freshness': freshness,
        'delay_statistics_min': delay_stats,
        'missing_data': missing_critical
    }
    
    # Print report if verbose
    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 DATA QUALITY REPORT: {csv_file.name}")
        print(f"{'='*80}")
        print(f"\n📁 File Information:")
        print(f"   Size: {report['file_size_mb']:.2f} MB")
        print(f"   Total records: {total_records:,}")
        
        print(f"\n🚂 Coverage:")
        print(f"   Unique trains: {unique_trips}")
        print(f"   Unique routes: {unique_routes}")
        print(f"   Unique stops: {unique_stops}")
        
        print(f"\n📡 Realtime States:")
        for state, count in state_dist.items():
            pct = count / total_records * 100
            print(f"   {state:12} {count:5,} ({pct:5.1f}%)")
        
        print(f"\n✅ Actual Delay Coverage:")
        print(f"   MODIFIED records: {modified_count:,}/{total_records:,} ({coverage_pct:.1f}%)")
        
        if freshness:
            print(f"\n⏰ Data Freshness:")
            print(f"   Oldest collection: {freshness['oldest']}")
            print(f"   Newest collection: {freshness['newest']}")
            print(f"   Number of collections: {freshness['collection_count']}")
        
        if delay_stats:
            print(f"\n⏱️ Delay Statistics (minutes):")
            print(f"   Mean:   {delay_stats['mean']:7.2f}")
            print(f"   Median: {delay_stats['median']:7.2f}")
            print(f"   Std:    {delay_stats['std']:7.2f}")
            print(f"   Range:  [{delay_stats['min']:.2f}, {delay_stats['max']:.2f}]")
        
        if missing_critical:
            print(f"\n⚠️ Missing Data (critical columns):")
            for col, stats in missing_critical.items():
                if stats['missing'] > 0:
                    print(f"   {col:20} {stats['missing']:5,} missing ({stats['pct']:5.1f}%)")
        
        print(f"\n{'='*80}\n")
    
    return report


def compare_collection_runs(csv_files: list, verbose: bool = True) -> pd.DataFrame:
    """
    Compare multiple collection runs to track data growth and quality trends.
    
    Args:
        csv_files: List of CSV file paths
        verbose: Whether to print comparison table
    
    Returns:
        DataFrame: Comparison metrics
    """
    
    results = []
    
    for csv_file in csv_files:
        report = check_data_quality(csv_file, verbose=False)
        if 'error' not in report:
            results.append({
                'file': report['file'],
                'records': report['total_records'],
                'trains': report['unique_trips'],
                'routes': report['unique_routes'],
                'modified_pct': report['actual_delay_coverage_pct'],
                'collections': report['freshness'].get('collection_count', 0),
                'newest': report['freshness'].get('newest', '')
            })
    
    df = pd.DataFrame(results)
    
    if verbose and not df.empty:
        print("\n📊 COLLECTION RUNS COMPARISON:")
        print(df.to_string(index=False))
    
    return df
