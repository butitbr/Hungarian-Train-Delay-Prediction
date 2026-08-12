"""
Standalone data collection script for train delay prediction.

Collects real-time train data from MÁV GraphQL API including:
- IC train schedules and routes
- Actual arrival/departure times with delays
- Real-time status updates
- Weather data for departure stations

Usage:
    python collect_train_data.py [--max-routes N] [--verbose] [--batch-size N] [--max-concurrent N]
"""

import asyncio
import os
import sys
import json
import logging
import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import pandas as pd
import requests
from dotenv import load_dotenv

# Import API URL from config
from config import mav_api_url, mav_api_url_fallback

# Load environment variables
load_dotenv()

# Configuration
DATA_ROOT = Path("data")
COLLECTED_DIR = DATA_ROOT / "collected"
LOGS_DIR = Path("logs")

# Weather API configuration
OWM_API_KEY = os.getenv('openweathermap_api_key')
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
INTER_BATCH_DELAY_SECONDS = 0.2
REQUEST_TIMEOUT_SECONDS = 10
TRIP_ID_PATTERN = re.compile(r'^[A-Za-z0-9:_\-.]+$')


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"collect_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def _post_graphql(url, query, headers, timeout=30):
    """
    Make a GraphQL POST request and return parsed JSON, or None on failure.
    Logs diagnostic information on unexpected responses.
    """
    try:
        response = requests.post(url, json=query, headers=headers, timeout=timeout)
        if response.status_code != 200:
            logging.warning(f"API returned HTTP {response.status_code} from {url}")
            logging.warning(f"Response body: {response.text[:500]!r}")
            return None
        if not response.text.strip():
            logging.warning(f"API returned empty body from {url}")
            return None
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logging.warning(f"API returned non-JSON from {url}: {e}")
            logging.warning(f"Response body preview: {response.text[:300]!r}")
            return None
    except requests.exceptions.RequestException as e:
        logging.warning(f"Request to {url} failed: {e}")
        return None


async def _post_graphql_async(session, url, query, headers, timeout=10, max_retries=3):
    """
    Make an async GraphQL POST request with retry and exponential backoff.
    Returns parsed JSON, or None on failure.
    """
    for attempt in range(max_retries):
        try:
            request_timeout = aiohttp.ClientTimeout(total=timeout)
            async with session.post(url, json=query, headers=headers, timeout=request_timeout) as response:
                response_text = await response.text()
                if response.status != 200:
                    logging.warning(f"API returned HTTP {response.status} from {url}")
                    logging.warning(f"Response body: {response_text[:500]!r}")
                elif not response_text.strip():
                    logging.warning(f"API returned empty body from {url}")
                else:
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logging.warning(f"API returned non-JSON from {url}: {e}")
                        logging.warning(f"Response body preview: {response_text[:300]!r}")

        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logging.warning(f"Async request to {url} failed: {e}")

        if attempt < max_retries - 1:
            backoff_seconds = 2 ** attempt
            await asyncio.sleep(backoff_seconds)

    return None


def build_batch_trip_query(trip_ids, service_date_formatted):
    """
    Build a GraphQL query that fetches multiple trips using aliases.
    """
    if not trip_ids:
        return None, []

    trip_queries = []
    valid_trip_ids = []
    for trip_id in trip_ids:
        candidate_trip_id = str(trip_id)
        if not TRIP_ID_PATTERN.fullmatch(candidate_trip_id):
            logging.warning(f"Skipping trip with invalid characters in trip_id: {candidate_trip_id!r}")
            continue
        valid_trip_ids.append(candidate_trip_id)

    for idx, safe_trip_id in enumerate(valid_trip_ids):
        trip_queries.append(
            f"""
            trip_{idx}: trip(id: "{safe_trip_id}") {{
                gtfsId
                tripShortName
                tripHeadsign
                routeShortName
                directionId
                serviceId
                activeDates
                route {{
                    longName
                    shortName
                    mode
                }}
                stoptimesForDate(serviceDate: "{service_date_formatted}") {{
                    stop {{
                        name
                        code
                        gtfsId
                        platformCode
                    }}
                    scheduledArrival
                    scheduledDeparture
                    realtimeArrival
                    realtimeDeparture
                    arrivalDelay
                    departureDelay
                    timepoint
                    realtime
                    realtimeState
                    serviceDay
                    pickupType
                    dropoffType
                    headsign
                }}
            }}
            """
        )

    if not trip_queries:
        return None, []

    return "{ " + " ".join(trip_queries) + " }", valid_trip_ids


def _trip_data_to_record(trip_id, route_info, trip_data, service_date):
    """
    Convert GraphQL trip payload to the train record format expected by downstream code.
    """
    stops = []
    stop_codes = []
    stop_gtfs_ids = []
    platforms = []
    scheduled_arrivals = []
    scheduled_departures = []
    realtime_arrivals = []
    realtime_departures = []
    arrival_delays = []
    departure_delays = []
    realtime_states = []

    for st in trip_data.get('stoptimesForDate', []):
        stops.append(st['stop']['name'])
        stop_codes.append(st['stop'].get('code'))
        stop_gtfs_ids.append(st['stop']['gtfsId'])
        platforms.append(st['stop'].get('platformCode'))
        scheduled_arrivals.append(st.get('scheduledArrival'))
        scheduled_departures.append(st.get('scheduledDeparture'))
        realtime_arrivals.append(st.get('realtimeArrival'))
        realtime_departures.append(st.get('realtimeDeparture'))
        arrival_delays.append(st.get('arrivalDelay', 0))
        departure_delays.append(st.get('departureDelay', 0))
        realtime_states.append(st.get('realtimeState', 'SCHEDULED'))

    return {
        'trip_id': trip_id,
        'train_number': trip_data.get('tripShortName', trip_data.get('routeShortName', 'N/A')),
        'route_name': trip_data.get('route', {}).get('longName', route_info['name']),
        'headsign': trip_data.get('tripHeadsign', 'N/A'),
        'direction_id': trip_data.get('directionId'),
        'service_id': trip_data.get('serviceId'),
        'active_dates': trip_data.get('activeDates', []),
        'stops': stops,
        'stop_codes': stop_codes,
        'stop_gtfs_ids': stop_gtfs_ids,
        'platforms': platforms,
        'scheduled_arrivals': scheduled_arrivals,
        'scheduled_departures': scheduled_departures,
        'realtime_arrivals': realtime_arrivals,
        'realtime_departures': realtime_departures,
        'arrival_delays': arrival_delays,
        'departure_delays': departure_delays,
        'realtime_states': realtime_states,
        'service_date': service_date
    }


async def _process_single_route_batches(session, semaphore, route_info, service_date, service_date_formatted,
                                        headers, urls_to_try, batch_size, request_timeout):
    """
    Process one route's trips in batched GraphQL requests.
    """
    route_trains = []
    trip_ids = route_info['trip_ids']
    total_batches = (len(trip_ids) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(trip_ids))
        batch_trip_ids = trip_ids[start_idx:end_idx]
        batch_query_str, query_trip_ids = build_batch_trip_query(batch_trip_ids, service_date_formatted)
        if not batch_query_str:
            continue

        batch_query = {"query": batch_query_str}
        data = None

        async with semaphore:
            for url in urls_to_try:
                data = await _post_graphql_async(
                    session,
                    url,
                    batch_query,
                    headers,
                    timeout=request_timeout
                )
                if data is not None:
                    break

        if data is None:
            logging.warning(f"Batch {batch_idx + 1}/{total_batches} failed for route {route_info['name']}")
            continue

        batch_data = data.get('data', {})
        for alias_idx, trip_id in enumerate(query_trip_ids):
            trip_data = batch_data.get(f"trip_{alias_idx}")
            if not trip_data or not trip_data.get('stoptimesForDate'):
                continue
            route_trains.append(_trip_data_to_record(trip_id, route_info, trip_data, service_date))

        await asyncio.sleep(INTER_BATCH_DELAY_SECONDS)

    return route_trains


async def process_batches_concurrent(routes_to_process, service_date, headers, batch_size=5,
                                     max_concurrent=3, request_timeout=REQUEST_TIMEOUT_SECONDS, verbose=True):
    """
    Process route batches concurrently with bounded request concurrency.
    """
    service_date_formatted = datetime.strptime(service_date, "%Y-%m-%d").strftime("%Y%m%d")
    semaphore = asyncio.Semaphore(max_concurrent)
    urls_to_try = [url for url in [mav_api_url, mav_api_url_fallback] if url]
    if not urls_to_try:
        logging.error("No configured MÁV API URL available for concurrent batch processing.")
        return []

    trains = []
    total_routes = len(routes_to_process)
    completed_routes = 0

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=max(max_concurrent * 2, 10))
    ) as session:
        tasks = [
            asyncio.create_task(
                _process_single_route_batches(
                    session=session,
                    semaphore=semaphore,
                    route_info=route_info,
                    service_date=service_date,
                    service_date_formatted=service_date_formatted,
                    headers=headers,
                    urls_to_try=urls_to_try,
                    batch_size=batch_size,
                    request_timeout=request_timeout
                )
            )
            for _, route_info in routes_to_process
        ]

        for route_task in asyncio.as_completed(tasks):
            try:
                route_trains = await route_task
            except Exception as e:
                logging.error(f"Route batch task failed: {e}", exc_info=True)
                route_trains = []
            trains.extend(route_trains)
            completed_routes += 1
            if verbose:
                print(f"  [{completed_routes}/{total_routes}] completed: {len(route_trains)} trains collected")

    return trains


def get_ic_routes():
    """
    Fetch all RAIL routes from the MÁV GraphQL API.
    (Function name kept for backwards compatibility, now returns all RAIL trains)
    Returns a dict with route_id -> {name, stations, trip_ids}
    """
    query = {
        "query": """{
            routes {
                gtfsId
                shortName
                longName
                mode
                patterns {
                    stops {
                        name
                    }
                    trips {
                        gtfsId
                    }
                }
            }
        }"""
    }
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Origin': 'https://mavplusz.hu',
        'Referer': 'https://mavplusz.hu/',
        'User-Agent': 'Mozilla/5.0 (compatible; train-data-collector/1.0)'
    }
    
    # Try primary URL, then fallback URL
    urls_to_try = [mav_api_url]
    if mav_api_url_fallback and mav_api_url_fallback != mav_api_url:
        urls_to_try.append(mav_api_url_fallback)

    data = None
    for url in urls_to_try:
        logging.info(f"Trying API endpoint: {url}")
        data = _post_graphql(url, query, headers)
        if data is not None:
            logging.info(f"Successfully connected to: {url}")
            break
        logging.warning(f"No valid response from: {url}")

    if data is None:
        logging.error("All API endpoints failed. Check network access and URL configuration.")
        return {}

    routes = {}
    all_routes = data.get('data', {}).get('routes', [])
    
    # Filter for RAIL routes (all passenger trains)
    for route in all_routes:
        if route.get('mode') != 'RAIL':
            continue
        
        route_name = str(route.get('longName', '') or route.get('shortName', ''))
        
        route_id = route['gtfsId']
        
        # Get unique stations from all patterns
        all_stations = set()
        all_trips = set()
        for pattern in route.get('patterns', []):
            for stop in pattern.get('stops', []):
                all_stations.add(stop['name'])
            for trip in pattern.get('trips', []):
                all_trips.add(trip['gtfsId'])
        
        routes[route_id] = {
            'name': route_name,
            'stations': sorted(list(all_stations)),
            'trip_ids': sorted(list(all_trips))
        }
    
    return routes


def get_ic_trains(service_date=None, max_routes=None, verbose=True):
    """
    Fetch all RAIL train trips with detailed information for a specific date.
    (Function name kept for backwards compatibility, now returns all RAIL trains)
    
    Args:
        service_date: Date string in format "YYYY-MM-DD" (defaults to today)
        max_routes: Maximum number of routes to fetch (None = all routes)
        verbose: Print progress information
    
    Returns:
        List of dicts with train details: {trip_id, train_number, route_name, 
                                          headsign, stops, departure_times, arrival_times}
    """
    if service_date is None:
        service_date = datetime.now().strftime("%Y-%m-%d")
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Origin': 'https://mavplusz.hu',
        'Referer': 'https://mavplusz.hu/',
        'User-Agent': 'Mozilla/5.0 (compatible; train-data-collector/1.0)'
    }

    try:
        # First, get all RAIL routes
        ic_routes = get_ic_routes()
        
        if verbose:
            print(f"Fetching RAIL trains for {service_date}")
            print(f"Found {len(ic_routes)} RAIL routes")
        
        trains = []
        routes_to_process = list(ic_routes.items())[:max_routes] if max_routes else list(ic_routes.items())
        
        # For each route, get detailed trip information
        for route_idx, (route_id, route_info) in enumerate(routes_to_process, 1):
            if verbose:
                print(f"  [{route_idx}/{len(routes_to_process)}] {route_info['name']}: {len(route_info['trip_ids'])} trips", end='')
            
            route_trains = 0
            for trip_id in route_info['trip_ids']:
                # Query trip details with ALL available fields
                # IMPORTANT: Use stoptimesForDate with serviceDate parameter to get REAL-TIME data
                # Format: YYYYMMDD (e.g., "20260314")
                service_date_formatted = datetime.strptime(service_date, "%Y-%m-%d").strftime("%Y%m%d")
                
                trip_query = {
                    "query": f"""{{
                        trip(id: "{trip_id}") {{
                            gtfsId
                            tripShortName
                            tripHeadsign
                            routeShortName
                            directionId
                            serviceId
                            activeDates
                            route {{
                                longName
                                shortName
                                mode
                            }}
                            stoptimesForDate(serviceDate: "{service_date_formatted}") {{
                                stop {{
                                    name
                                    code
                                    gtfsId
                                    platformCode
                                }}
                                scheduledArrival
                                scheduledDeparture
                                realtimeArrival
                                realtimeDeparture
                                arrivalDelay
                                departureDelay
                                timepoint
                                realtime
                                realtimeState
                                serviceDay
                                pickupType
                                dropoffType
                                headsign
                            }}
                        }}
                    }}"""
                }
                
                data = _post_graphql(mav_api_url, trip_query, headers, timeout=10)
                if data is None:
                    # Try fallback URL for individual trip queries too
                    if mav_api_url_fallback and mav_api_url_fallback != mav_api_url:
                        data = _post_graphql(mav_api_url_fallback, trip_query, headers, timeout=10)
                
                if data is None:
                    continue

                trip_data = data.get('data', {}).get('trip')
                if not trip_data or not trip_data.get('stoptimesForDate'):
                    continue
                
                # Extract detailed stop information
                stops = []
                stop_codes = []
                stop_gtfs_ids = []
                platforms = []
                scheduled_arrivals = []
                scheduled_departures = []
                realtime_arrivals = []
                realtime_departures = []
                arrival_delays = []
                departure_delays = []
                realtime_states = []
                
                for st in trip_data.get('stoptimesForDate', []):
                    stops.append(st['stop']['name'])
                    stop_codes.append(st['stop'].get('code'))
                    stop_gtfs_ids.append(st['stop']['gtfsId'])
                    platforms.append(st['stop'].get('platformCode'))
                    scheduled_arrivals.append(st.get('scheduledArrival'))
                    scheduled_departures.append(st.get('scheduledDeparture'))
                    realtime_arrivals.append(st.get('realtimeArrival'))
                    realtime_departures.append(st.get('realtimeDeparture'))
                    arrival_delays.append(st.get('arrivalDelay', 0))
                    departure_delays.append(st.get('departureDelay', 0))
                    realtime_states.append(st.get('realtimeState', 'SCHEDULED'))
                
                trains.append({
                    'trip_id': trip_id,
                    'train_number': trip_data.get('tripShortName', trip_data.get('routeShortName', 'N/A')),
                    'route_name': trip_data.get('route', {}).get('longName', route_info['name']),
                    'headsign': trip_data.get('tripHeadsign', 'N/A'),
                    'direction_id': trip_data.get('directionId'),
                    'service_id': trip_data.get('serviceId'),
                    'active_dates': trip_data.get('activeDates', []),
                    'stops': stops,
                    'stop_codes': stop_codes,
                    'stop_gtfs_ids': stop_gtfs_ids,
                    'platforms': platforms,
                    'scheduled_arrivals': scheduled_arrivals,
                    'scheduled_departures': scheduled_departures,
                    'realtime_arrivals': realtime_arrivals,
                    'realtime_departures': realtime_departures,
                    'arrival_delays': arrival_delays,
                    'departure_delays': departure_delays,
                    'realtime_states': realtime_states,
                    'service_date': service_date
                })
                route_trains += 1
            
            if verbose:
                print(f" → {route_trains} running")
        
        if verbose:
            print(f"\n✓ Total: {len(trains)} IC trains running on {service_date}")
        
        return trains
    
    except Exception as e:
        print(f"Error fetching IC trains: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_ic_trains_concurrent(service_date=None, max_routes=None, verbose=True, batch_size=5,
                             max_concurrent=3, request_timeout=REQUEST_TIMEOUT_SECONDS):
    """
    Fetch RAIL train trips using batched GraphQL queries with async concurrency.
    """
    if service_date is None:
        service_date = datetime.now().strftime("%Y-%m-%d")

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Origin': 'https://mavplusz.hu',
        'Referer': 'https://mavplusz.hu/',
        'User-Agent': 'Mozilla/5.0 (compatible; train-data-collector/1.0)'
    }

    try:
        ic_routes = get_ic_routes()

        if verbose:
            print(f"Fetching RAIL trains for {service_date} (concurrent mode)")
            print(f"Found {len(ic_routes)} RAIL routes")

        routes_to_process = list(ic_routes.items())[:max_routes] if max_routes else list(ic_routes.items())

        try:
            asyncio.get_running_loop()
            logging.error(
                "Concurrent train collection cannot start because an event loop is already running. "
                "Run this function from a regular Python process/CLI context."
            )
            return []
        except RuntimeError:
            # No running event loop in this thread; safe to use asyncio.run
            pass

        trains = asyncio.run(
            process_batches_concurrent(
                routes_to_process=routes_to_process,
                service_date=service_date,
                headers=headers,
                batch_size=batch_size,
                max_concurrent=max_concurrent,
                request_timeout=request_timeout,
                verbose=verbose
            )
        )

        if verbose:
            print(f"\n✓ Total: {len(trains)} RAIL trains collected for {service_date}")

        return trains

    except Exception as e:
        logging.error(f"Error fetching RAIL trains concurrently: {e}", exc_info=True)
        return []


def trains_to_dataframe(trains_data):
    """
    Convert train data to pandas DataFrame.
    
    Args:
        trains_data: List of train dicts returned by get_ic_trains()
    
    Returns:
        pandas DataFrame with one row per train-stop combination
    """
    rows = []
    
    for train in trains_data:
        # Create one row for each stop
        for i in range(len(train['stops'])):
            row = {
                'trip_id': train['trip_id'],
                'train_number': train['train_number'],
                'route_name': train['route_name'],
                'headsign': train['headsign'],
                'direction_id': train['direction_id'],
                'service_id': train['service_id'],
                'service_date': train['service_date'],
                'stop_sequence': i,
                'stop_name': train['stops'][i],
                'stop_gtfs_id': train['stop_gtfs_ids'][i],
                'stop_code': train['stop_codes'][i],
                'platform': train['platforms'][i],
                'scheduled_arrival': train['scheduled_arrivals'][i],
                'scheduled_departure': train['scheduled_departures'][i],
                'realtime_arrival': train['realtime_arrivals'][i],
                'realtime_departure': train['realtime_departures'][i],
                'arrival_delay': train['arrival_delays'][i],
                'departure_delay': train['departure_delays'][i],
                'realtime_state': train['realtime_states'][i]
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Convert time columns from seconds to datetime.time objects for easier reading
    def seconds_to_time(seconds):
        if seconds is None:
            return None
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        from datetime import time
        return time(hours % 24, minutes, secs)
    
    time_columns = ['scheduled_arrival', 'scheduled_departure', 'realtime_arrival', 'realtime_departure']
    for col in time_columns:
        if col in df.columns:
            df[f'{col}_time'] = df[col].apply(seconds_to_time)
    
    return df


def get_weather_for_location(lat: float, lon: float, location_name: str = "") -> dict:
    """
    Fetch current weather data from OpenWeatherMap API.
    
    Args:
        lat: Latitude
        lon: Longitude
        location_name: Optional location name for logging
        
    Returns:
        Dict with weather data: temp, feels_like, temp_min, temp_max, 
        pressure, humidity, wind_speed, precipitation
    """
    if not OWM_API_KEY:
        logging.warning("OpenWeatherMap API key not found in environment")
        return {}
    
    try:
        params = {
            'lat': lat,
            'lon': lon,
            'appid': OWM_API_KEY,
            'units': 'metric'
        }
        
        response = requests.get(OWM_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        weather = {
            'temp': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'temp_min': data['main']['temp_min'],
            'temp_max': data['main']['temp_max'],
            'pressure': data['main']['pressure'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'precipitation': 0.0
        }
        
        # Add precipitation if available (rain or snow)
        if 'rain' in data:
            weather['precipitation'] += data['rain'].get('1h', 0)
        if 'snow' in data:
            weather['precipitation'] += data['snow'].get('1h', 0)
        
        location_str = f" for {location_name}" if location_name else ""
        logging.info(f"Fetched weather{location_str}: {weather['temp']:.1f}°C, "
                    f"humidity: {weather['humidity']}%, wind: {weather['wind_speed']:.1f} m/s")
        
        return weather
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch weather data: {e}")
        return {}
    except (KeyError, ValueError) as e:
        logging.error(f"Failed to parse weather data: {e}")
        return {}


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
        stops = stops[['stop_name', 'stop_lat', 'stop_lon']].drop_duplicates()
        logging.info(f"Loaded {len(stops)} station coordinates from GTFS")
        return stops
    except Exception as e:
        logging.error(f"Failed to load GTFS stops: {e}")
        return pd.DataFrame(columns=['stop_name', 'stop_lat', 'stop_lon'])


def collect_train_data(service_date: str = None, max_routes: int = None,
                       verbose: bool = True, batch_size: int = 5,
                       max_concurrent: int = 3,
                       request_timeout: int = REQUEST_TIMEOUT_SECONDS) -> pd.DataFrame:
    """
    Collect train data from MÁV API.
    
    Args:
        service_date: Date in format YYYY-MM-DD (default: today)
        max_routes: Maximum number of routes to fetch (None = all)
        verbose: Print progress information
        batch_size: Number of trips per GraphQL batch query
        max_concurrent: Maximum number of concurrent API requests
        request_timeout: Timeout per API request in seconds
        
    Returns:
        DataFrame with collected train data
    """
    if service_date is None:
        service_date = datetime.now().strftime("%Y-%m-%d")
    
    logging.info(f"Collecting train data for {service_date}")
    logging.info(f"Max routes: {max_routes if max_routes else 'all'}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Max concurrent requests: {max_concurrent}")
    logging.info(f"Request timeout (s): {request_timeout}")
    
    try:
        # Fetch IC trains from API
        trains = get_ic_trains_concurrent(
            service_date=service_date,
            max_routes=max_routes,
            verbose=verbose,
            batch_size=batch_size,
            max_concurrent=max_concurrent,
            request_timeout=request_timeout
        )
        
        if not trains:
            logging.warning("No trains fetched from API")
            return pd.DataFrame()
        
        logging.info(f"Fetched {len(trains)} trains from API")
        
        # Convert to DataFrame
        df = trains_to_dataframe(trains)
        
        if df.empty:
            logging.warning("DataFrame is empty after conversion")
            return df
        
        logging.info(f"Converted to DataFrame: {len(df)} rows × {len(df.columns)} columns")
        
        # Filter for trains that have departed or are departing soon (within last 12 hours or next 2 hours)
        current_time_seconds = datetime.now().hour * 3600 + datetime.now().minute * 60
        
        # Keep records where:
        # 1. Train has already departed (scheduled_departure < current time)
        # 2. OR train is departing very soon (within 2 hours)
        # 3. OR realtime_state is MODIFIED (has actual data)
        time_window_past = current_time_seconds - (12 * 3600)  # 12 hours ago
        time_window_future = current_time_seconds + (2 * 3600)  # 2 hours from now
        
        before_filter = len(df)
        df = df[
            (df['scheduled_departure'] <= time_window_future) |  # Departing soon
            (df['realtime_state'] == 'MODIFIED')  # Or has actual data
        ]
        after_filter = len(df)
        
        logging.info(f"Filtered to relevant trains: {before_filter} → {after_filter} records "
                    f"(removed {before_filter - after_filter} future departures)")
        
        if df.empty:
            logging.warning("No relevant trains after time filtering")
            return df
        
        # Add collection timestamp
        df['collection_timestamp'] = datetime.now()
        
        # Add metadata
        df['data_source'] = 'mav_graphql_api'
        df['collector_version'] = '1.0'
        
        return df
        
    except Exception as e:
        logging.error(f"Failed to collect train data: {e}", exc_info=True)
        return pd.DataFrame()


def enrich_with_weather(df: pd.DataFrame, station_coords: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich train data with weather information for each stop along the route.
    
    Fetches weather data for each unique stop location to capture
    changing conditions along the train's journey.
    
    Args:
        df: Train data DataFrame
        station_coords: Station coordinates DataFrame
        
    Returns:
        DataFrame with weather columns added
    """
    if df.empty or station_coords.empty:
        logging.warning("Empty dataframe or no station coordinates, skipping weather enrichment")
        return df
    
    logging.info("Enriching data with weather information for each stop...")
    
    # Merge with station coordinates
    df = df.merge(
        station_coords,
        on='stop_name',
        how='left'
    )
    
    # Get all unique stop locations (not just departure stations)
    unique_stops = df[['stop_name', 'stop_lat', 'stop_lon']].drop_duplicates(
        subset=['stop_lat', 'stop_lon']
    ).dropna(subset=['stop_lat', 'stop_lon'])
    
    logging.info(f"Fetching weather for {len(unique_stops)} unique stop locations...")
    
    # Fetch weather for each unique stop location
    weather_cache = {}
    
    for _, row in unique_stops.iterrows():
        station_key = (row['stop_lat'], row['stop_lon'])
        
        if station_key not in weather_cache:
            weather = get_weather_for_location(
                row['stop_lat'],
                row['stop_lon'],
                row['stop_name']
            )
            weather_cache[station_key] = weather
            time.sleep(0.1)  # Rate limiting for API
    
    # Add weather columns
    weather_cols = ['temp', 'feels_like', 'temp_min', 'temp_max', 
                   'pressure', 'humidity', 'wind_speed', 'precipitation']
    
    for col in weather_cols:
        df[f'weather_{col}'] = None
    
    # Map weather to each stop based on its coordinates
    for idx, row in df.iterrows():
        if pd.notna(row['stop_lat']) and pd.notna(row['stop_lon']):
            station_key = (row['stop_lat'], row['stop_lon'])
            
            if station_key in weather_cache:
                weather = weather_cache[station_key]
                
                for col in weather_cols:
                    if col in weather:
                        df.at[idx, f'weather_{col}'] = weather[col]
    
    weather_count = df['weather_temp'].notna().sum()
    logging.info(f"Added weather data to {weather_count}/{len(df)} stops ({len(weather_cache)} unique locations)")
    
    return df


def save_collected_data(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Save collected data to CSV file.
    
    Args:
        df: DataFrame to save
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    if df.empty:
        logging.warning("No data to save")
        return None
    
    # Create output directory structure: data/collected/YYYY-MM-DD/
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M")
    
    date_dir = output_dir / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = date_dir / f"{time_str}_ic_trains.csv"
    
    try:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f"Saved {len(df)} records to {output_file}")
        logging.info(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
        
        return output_file
        
    except Exception as e:
        logging.error(f"Failed to save data: {e}", exc_info=True)
        return None


def save_collected_data_incremental(df: pd.DataFrame, output_dir: Path, service_date: str) -> Path:
    """
    Save data with intelligent merging to avoid duplicates and preserve actual delays.
    
    Uses smart merge strategy:
    - Adds new records
    - Updates SCHEDULED → MODIFIED (upgrade to actual data)
    - Preserves MODIFIED when new data is SCHEDULED (don't lose actual delays)
    - Updates MODIFIED → MODIFIED if newer timestamp
    
    Args:
        df: DataFrame to save
        output_dir: Output directory
        service_date: Service date (YYYY-MM-DD format)
        
    Returns:
        Path to saved file
    """
    from utils.incremental_merge import smart_merge_train_data
    from utils.data_quality import check_data_quality
    
    if df.empty:
        logging.warning("No data to save")
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use daily files (all collections for same day go to same file)
    date_str = service_date.replace('-', '')
    output_file = output_dir / f"trains_{date_str}.csv"
    
    # Add collection timestamp
    df['collection_timestamp'] = datetime.now().isoformat()
    
    logging.info(f"Incremental save mode: {len(df)} new records")
    
    # Merge with existing data for this day
    merged_df = smart_merge_train_data(df, output_file)
    
    try:
        # Save merged data
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f"Saved {len(merged_df)} total records to {output_file}")
        logging.info(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
        
        # Run quality check
        check_data_quality(output_file, verbose=False)
        
        return output_file
        
    except Exception as e:
        logging.error(f"Failed to save data: {e}", exc_info=True)
        return None


def print_summary(df: pd.DataFrame):
    """Print summary statistics of collected data."""
    if df.empty:
        logging.info("No data collected")
        return
    
    logging.info("\n" + "="*60)
    logging.info("COLLECTION SUMMARY")
    logging.info("="*60)
    
    logging.info(f"Total records: {len(df)}")
    logging.info(f"Unique trains: {df['trip_id'].nunique()}")
    logging.info(f"Unique routes: {df['route_name'].nunique()}")
    logging.info(f"Unique stations: {df['stop_name'].nunique()}")
    
    # Realtime data availability
    has_realtime = (df['realtime_state'] == 'MODIFIED').sum()
    has_delays = (df['arrival_delay'] != 0).sum()
    
    logging.info(f"\nRealtime data:")
    logging.info(f"  - Modified states: {has_realtime} ({has_realtime/len(df)*100:.1f}%)")
    logging.info(f"  - Records with delays: {has_delays} ({has_delays/len(df)*100:.1f}%)")
    
    if has_delays > 0:
        avg_delay = df[df['arrival_delay'] != 0]['arrival_delay'].mean() / 60
        max_delay = df['arrival_delay'].max() / 60
        min_delay = df['arrival_delay'].min() / 60
        
        logging.info(f"  - Average delay: {avg_delay:.1f} minutes")
        logging.info(f"  - Max delay: {max_delay:.1f} minutes")
        logging.info(f"  - Min delay: {min_delay:.1f} minutes")
    
    # Weather data availability
    if 'weather_temp' in df.columns:
        has_weather = df['weather_temp'].notna().sum()
        logging.info(f"\nWeather data: {has_weather} records ({has_weather/len(df)*100:.1f}%)")
    
    logging.info("="*60 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Collect train delay data from MÁV API"
    )
    parser.add_argument(
        '--max-routes',
        type=int,
        default=None,
        help='Maximum number of routes to fetch (default: all)'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Service date in YYYY-MM-DD format (default: today)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    parser.add_argument(
        '--with-weather',
        action='store_true',
        help='Include weather data collection (DEPRECATED: use separate weather forecast collection instead)'
    )
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Use incremental merge (preserve actual delays, prevent duplicates)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Number of trips per GraphQL query (default: 5)'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=3,
        help='Maximum parallel API requests (default: 3)'
    )
    
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be >= 1")
    
    # Setup logging
    logger = setup_logging(LOGS_DIR)
    
    logger.info("="*60)
    logger.info("TRAIN DATA COLLECTION STARTED")
    logger.info("="*60)
    logger.info(f"Date: {args.date if args.date else 'today'}")
    logger.info(f"Max routes: {args.max_routes if args.max_routes else 'all'}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max concurrent requests: {args.max_concurrent}")
    logger.info(f"Weather collection: {'enabled (legacy)' if args.with_weather else 'disabled (use separate forecast collection)'}")
    logger.info(f"Mode: {'INCREMENTAL (merge)' if args.incremental else 'STANDARD (new file)'}")
    logger.info("="*60 + "\n")
    
    try:
        # Collect train data
        df = collect_train_data(
            service_date=args.date,
            max_routes=args.max_routes,
            verbose=args.verbose,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent
        )
        
        if df.empty:
            logger.error("No data collected. Exiting.")
            return 1
        
        # Load station coordinates and enrich with weather (DEPRECATED)
        # Note: Use collect_weather_forecast.py + match_weather_to_trains.py instead
        if args.with_weather:
            logger.warning("WARNING: --with-weather is deprecated. Use separate weather forecast collection.")
            logger.warning("Run: python collect_weather_forecast.py (4x/day via GitHub Actions)")
            logger.warning("Then: python match_weather_to_trains.py --train-file <file> --date <date>")
            station_coords = load_station_coordinates()
            df = enrich_with_weather(df, station_coords)
        
        # Save collected data (incremental or standard)
        if args.incremental:
            service_date = args.date if args.date else datetime.now().strftime("%Y-%m-%d")
            output_file = save_collected_data_incremental(df, COLLECTED_DIR, service_date)
        else:
            output_file = save_collected_data(df, COLLECTED_DIR)
        
        if output_file is None:
            logger.error("Failed to save data. Exiting.")
            return 1
        
        # Print summary
        print_summary(df)
        
        logger.info("="*60)
        logger.info("COLLECTION COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Collection failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
