"""
Incremental merge logic for train data collection.

Handles intelligent merging of new data with existing data to:
1. Prevent duplicates
2. Preserve actual delay data (MODIFIED status)
3. Update to latest state when appropriate
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_record_key(row):
    """
    Create unique composite key for a train stop record.
    
    Args:
        row: DataFrame row or dict with trip_id, stop_sequence, service_date
    
    Returns:
        str: Unique key in format "trip_id_stopseq_date"
    """
    return f"{row['trip_id']}_{row['stop_sequence']}_{row['service_date']}"


def smart_merge_train_data(new_df: pd.DataFrame, existing_csv: Path) -> pd.DataFrame:
    """
    Merge new data with existing, preserving actual delays and updating to latest state.
    
    Decision Rules:
    1. If record doesn't exist: ADD it
    2. If record exists with SCHEDULED and new is MODIFIED: UPDATE (upgrade to actual data)
    3. If record exists with MODIFIED and new is SCHEDULED: KEEP old (preserve actual data)
    4. If both MODIFIED: UPDATE to newest (based on collection_timestamp)
    5. If both SCHEDULED: UPDATE to newest
    
    Args:
        new_df: DataFrame with newly collected data
        existing_csv: Path to existing CSV file
    
    Returns:
        DataFrame: Merged data without duplicates
    """
    
    # If no existing file, return new data
    if not existing_csv.exists():
        logger.info(f"No existing file at {existing_csv}, returning new data")
        return new_df.copy()
    
    # Load existing data
    try:
        old_df = pd.read_csv(existing_csv)
        logger.info(f"Loaded {len(old_df)} existing records from {existing_csv.name}")
    except Exception as e:
        logger.error(f"Failed to load existing data: {e}")
        return new_df.copy()
    
    # Create composite keys for both datasets
    new_df['record_key'] = new_df.apply(create_record_key, axis=1)
    old_df['record_key'] = old_df.apply(create_record_key, axis=1)
    
    # Identify record categories
    existing_keys = set(old_df['record_key'])
    new_keys = set(new_df['record_key'])
    
    keys_to_add = new_keys - existing_keys  # Completely new records
    keys_to_update = new_keys & existing_keys  # Records that need merging
    keys_unchanged = existing_keys - new_keys  # Keep as is
    
    logger.info(f"Merge analysis: {len(keys_to_add)} new, {len(keys_to_update)} to update, {len(keys_unchanged)} unchanged")
    
    # Start with records that don't need updates
    result_records = []
    
    # Keep unchanged old records
    unchanged_df = old_df[old_df['record_key'].isin(keys_unchanged)]
    result_records.append(unchanged_df)
    
    # Add completely new records
    new_records_df = new_df[new_df['record_key'].isin(keys_to_add)]
    result_records.append(new_records_df)
    logger.info(f"Adding {len(new_records_df)} completely new records")
    
    # Handle updates intelligently
    update_count = 0
    preserve_count = 0
    
    for key in keys_to_update:
        old_row = old_df[old_df['record_key'] == key].iloc[0]
        new_row = new_df[new_df['record_key'] == key].iloc[0]
        
        # Decision logic
        should_update = False
        reason = ""
        
        old_state = old_row.get('realtime_state', 'SCHEDULED')
        new_state = new_row.get('realtime_state', 'SCHEDULED')
        
        if old_state == 'SCHEDULED' and new_state == 'MODIFIED':
            # Upgrade from schedule to actual data
            should_update = True
            reason = "SCHEDULED → MODIFIED (upgrade to actual)"
            
        elif old_state == 'MODIFIED' and new_state == 'SCHEDULED':
            # DON'T overwrite actual data with schedule
            should_update = False
            reason = "MODIFIED → SCHEDULED (preserving actual data)"
            
        elif old_state == 'MODIFIED' and new_state == 'MODIFIED':
            # Both modified - take newer timestamp
            old_time = pd.to_datetime(old_row.get('collection_timestamp', '1900-01-01'))
            new_time = pd.to_datetime(new_row.get('collection_timestamp', '1900-01-01'))
            if new_time > old_time:
                should_update = True
                reason = "MODIFIED → MODIFIED (newer timestamp)"
            else:
                reason = "MODIFIED → MODIFIED (keeping older, same timestamp)"
                
        elif old_state == 'SCHEDULED' and new_state == 'SCHEDULED':
            # Both scheduled - take newer
            old_time = pd.to_datetime(old_row.get('collection_timestamp', '1900-01-01'))
            new_time = pd.to_datetime(new_row.get('collection_timestamp', '1900-01-01'))
            if new_time > old_time:
                should_update = True
                reason = "SCHEDULED → SCHEDULED (newer timestamp)"
        
        if should_update:
            result_records.append(pd.DataFrame([new_row]))
            update_count += 1
            if update_count <= 5:  # Log first few
                logger.debug(f"Update {key}: {reason}")
        else:
            result_records.append(pd.DataFrame([old_row]))
            preserve_count += 1
            if preserve_count <= 5:  # Log first few
                logger.debug(f"Preserve {key}: {reason}")
    
    logger.info(f"Updated {update_count} records, preserved {preserve_count} records")
    
    # Combine all records
    result_df = pd.concat(result_records, ignore_index=True)
    
    # Remove helper column
    result_df = result_df.drop('record_key', axis=1)
    
    logger.info(f"Merge complete: {len(result_df)} total records")
    
    return result_df


def get_duplicate_stats(df: pd.DataFrame) -> dict:
    """
    Check for duplicate records in dataset.
    
    Args:
        df: DataFrame with train data
    
    Returns:
        dict: Statistics about duplicates
    """
    df_temp = df.copy()
    df_temp['record_key'] = df_temp.apply(create_record_key, axis=1)
    
    duplicates = df_temp[df_temp.duplicated(subset='record_key', keep=False)]
    
    return {
        'total_records': len(df),
        'unique_records': df_temp['record_key'].nunique(),
        'duplicate_count': len(duplicates),
        'duplicate_keys': duplicates['record_key'].unique().tolist()[:10]  # First 10
    }
