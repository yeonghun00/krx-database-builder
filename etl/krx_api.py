"""
KRX API integration module for fetching stock market data.
Handles API requests, data validation, and error handling.
"""

import requests
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
import json
import threading

class KRXAPI:
    """Handles KRX API requests for multiple markets (KOSPI, KOSDAQ, KODEX)."""

    # Default configuration values
    DEFAULT_CONFIG = {
        'request_delay': 1.0,
        'backfill_request_delay': 0.5,
        'max_concurrent_requests': 3,
        'enable_parallel_processing': True
    }

    def __init__(self, auth_key: str, config: Dict = None):
        """
        Initialize KRX API client.

        Args:
            auth_key (str): KRX API authentication key
            config (Dict, optional): Configuration dictionary with keys:
                - request_delay: Delay between requests (default: 1.0)
                - backfill_request_delay: Delay for backfill operations (default: 0.5)
                - max_concurrent_requests: Max parallel requests (default: 3)
                - enable_parallel_processing: Enable parallel fetching (default: True)
        """
        self.auth_key = auth_key
        self.headers = {
            "AUTH_KEY": auth_key,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        self.logger = logging.getLogger(__name__)

        # Merge provided config with defaults
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}

        # Rate limiting settings from config
        self.request_delay = cfg['request_delay']
        self.last_request_time = 0
        self.backfill_request_delay = cfg['backfill_request_delay']
        self.max_concurrent_requests = cfg['max_concurrent_requests']
        self.enable_parallel_processing = cfg['enable_parallel_processing']
        self._rate_limit_lock = threading.Lock()

        # Market endpoints for KRX data (stock trading)
        self.market_endpoints = {
            'kospi': 'https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd',
            'kosdaq': 'https://data-dbg.krx.co.kr/svc/apis/sto/ksq_bydd_trd',
            'kodex': 'https://data-dbg.krx.co.kr/svc/apis/sto/knx_bydd_trd'
        }

        # Index endpoints for market indices, bonds, and derivatives
        self.index_endpoints = {
            'kospi_index': 'https://data-dbg.krx.co.kr/svc/apis/idx/kospi_dd_trd',
            'kosdaq_index': 'https://data-dbg.krx.co.kr/svc/apis/idx/kosdaq_dd_trd',
            'bond_index': 'https://data-dbg.krx.co.kr/svc/apis/idx/bon_dd_trd.json',
            'govt_bond': 'https://data-dbg.krx.co.kr/svc/apis/bon/kts_bydd_trd',
            'derivatives': 'https://data-dbg.krx.co.kr/svc/apis/idx/drvprod_dd_trd'
        }
    
    def _make_request(self, date: str, market: str = 'kospi') -> Optional[Dict]:
        """
        Make API request for a specific date and market.
        
        Args:
            date (str): Date in YYYYMMDD format
            market (str): Market type ('kospi', 'kosdaq', 'kodex')
            
        Returns:
            Optional[Dict]: API response data or None if failed
        """
        if market not in self.market_endpoints:
            self.logger.error(f"Unsupported market: {market}")
            return None
            
        base_url = self.market_endpoints[market]
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        
        params = {"basDd": date}
        
        try:
            self.logger.info(f"Making API request for {market} market, date: {date}")
            response = requests.get(base_url, headers=self.headers, params=params, timeout=30)
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.logger.info(f"Successfully fetched {market} data for {date}")
                    return data
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error for {market} market, date {date}: {e}")
                    return None
            else:
                self.logger.error(f"API request failed for {market} market, {date}: HTTP {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request exception for {market} market, date {date}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error for {market} market, date {date}: {e}")
            return None
    
    def fetch_data_for_date(self, date: str) -> List[Dict]:
        """
        Fetch stock data for a specific date.
        
        Args:
            date (str): Date in YYYYMMDD format
            
        Returns:
            List[Dict]: List of stock data records
        """
        data = self._make_request(date)
        
        if not data:
            return []
        
        # Validate response structure
        if "OutBlock_1" not in data:
            self.logger.error(f"Invalid response structure for date {date}")
            return []
        
        records = data["OutBlock_1"]
        
        # Validate and clean data
        validated_records = []
        for record in records:
            validated_record = self._validate_and_clean_record(record, date)
            if validated_record:
                validated_records.append(validated_record)
        
        self.logger.info(f"Validated {len(validated_records)} records for date {date}")
        return validated_records
    
    def _validate_and_clean_record(self, record: Dict, date: str, market: str = None) -> Optional[Dict]:
        """
        Validate and clean a single stock record.

        Args:
            record (Dict): Raw stock record
            date (str): Date for the record
            market (str, optional): Market type ('kospi', 'kosdaq', 'kodex')

        Returns:
            Optional[Dict]: Validated record or None if invalid
        """
        try:
            market_info = f" ({market} market)" if market else ""

            # Check required fields
            required_fields = ['BAS_DD', 'ISU_CD', 'ISU_NM']
            for field in required_fields:
                if field not in record or not record[field]:
                    self.logger.warning(f"Missing required field {field} in{market_info} record for date {date}")
                    return None

            # Validate date consistency
            if record['BAS_DD'] != date:
                self.logger.warning(f"Date mismatch in{market_info} record: expected {date}, got {record['BAS_DD']}")
                return None

            # Add market information if provided
            if market:
                record['market_type'] = market

            # Clean and validate numeric fields
            numeric_fields = [
                'TDD_CLSPRC', 'CMPPREVDD_PRC', 'TDD_OPNPRC', 'TDD_HGPRC',
                'TDD_LWPRC', 'ACC_TRDVOL', 'ACC_TRDVAL', 'MKTCAP', 'LIST_SHRS'
            ]

            for field in numeric_fields:
                if field in record and record[field]:
                    try:
                        # Remove any non-numeric characters except decimal point
                        value = str(record[field]).replace(',', '')
                        record[field] = value
                    except (ValueError, TypeError):
                        record[field] = None
                else:
                    record[field] = None

            # Validate change rate (should be numeric)
            if 'FLUC_RT' in record and record['FLUC_RT']:
                try:
                    record['FLUC_RT'] = str(record['FLUC_RT']).replace(',', '')
                except (ValueError, TypeError):
                    record['FLUC_RT'] = None
            else:
                record['FLUC_RT'] = None

            return record

        except Exception as e:
            self.logger.error(f"Error validating{market_info} record for date {date}: {e}")
            return None
    
    def fetch_data_range(self, start_date: str, end_date: str) -> Dict[str, List[Dict]]:
        """
        Fetch data for a range of dates.
        
        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            
        Returns:
            Dict[str, List[Dict]]: Dictionary with dates as keys and data as values
        """
        from datetime import datetime, timedelta
        
        # Validate date range
        try:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            return {}
        
        if start_dt > end_dt:
            self.logger.error("Start date cannot be after end date")
            return {}
        
        # Generate date range
        current_date = start_dt
        results = {}
        
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y%m%d')
            
            # Skip weekends (KRX is closed on weekends)
            if current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                self.logger.info(f"Skipping weekend date: {date_str}")
                current_date += timedelta(days=1)
                continue
            
            # Fetch data for this date
            data = self.fetch_data_for_date(date_str)
            if data:
                results[date_str] = data
            
            current_date += timedelta(days=1)
        
        return results
    
    def _make_request_with_rate_limit(self, date: str, market: str = 'kospi', is_backfill: bool = False) -> Optional[Dict]:
        """
        Make API request with optimized rate limiting for backfill operations.
        
        Args:
            date (str): Date in YYYYMMDD format
            market (str): Market type ('kospi', 'kosdaq', 'kodex')
            is_backfill (bool): Whether this is a backfill operation (allows faster requests)
            
        Returns:
            Optional[Dict]: API response data or None if failed
        """
        if market not in self.market_endpoints:
            self.logger.error(f"Unsupported market: {market}")
            return None
            
        base_url = self.market_endpoints[market]
        
        # Use optimized rate limiting for backfill
        delay = self.backfill_request_delay if is_backfill else self.request_delay
        
        # Thread-safe rate limiting
        with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < delay:
                time.sleep(delay - time_since_last)
        
        params = {"basDd": date}
        
        try:
            self.logger.info(f"Making API request for {market} market, date: {date}")
            response = requests.get(base_url, headers=self.headers, params=params, timeout=30)
            
            with self._rate_limit_lock:
                self.last_request_time = time.time()
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.logger.info(f"Successfully fetched {market} data for {date}")
                    return data
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error for {market} market, date {date}: {e}")
                    return None
            else:
                self.logger.error(f"API request failed for {market} market, {date}: HTTP {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request exception for {market} market, date {date}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error for {market} market, date {date}: {e}")
            return None
    
    def fetch_data_for_date_parallel(self, date: str, markets: List[str] = ['kospi'], is_backfill: bool = False) -> Dict[str, List[Dict]]:
        """
        Fetch data for a specific date from multiple markets in parallel.
        
        Args:
            date (str): Date in YYYYMMDD format
            markets (List[str]): List of markets to fetch from
            is_backfill (bool): Whether this is a backfill operation
            
        Returns:
            Dict[str, List[Dict]]: Dictionary with market names as keys and data as values
        """
        if not self.enable_parallel_processing or len(markets) == 1:
            # Fall back to sequential processing for single market or when parallel is disabled
            return self.fetch_data_for_date_multi_market(date, markets)
        
        results = {}
        futures = {}
        
        # Submit all requests in parallel
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            for market in markets:
                future = executor.submit(self._make_request_with_rate_limit, date, market, is_backfill)
                futures[future] = market
        
        # Collect results
        for future in as_completed(futures):
            market = futures[future]
            try:
                data = future.result()
                if not data:
                    results[market] = []
                    continue
                
                if "OutBlock_1" not in data:
                    self.logger.error(f"Invalid response structure for {market} market, date {date}")
                    results[market] = []
                    continue
                
                records = data["OutBlock_1"]
                validated_records = []
                
                for record in records:
                    validated_record = self._validate_and_clean_record(record, date, market)
                    if validated_record:
                        validated_records.append(validated_record)
                
                results[market] = validated_records
                self.logger.info(f"Validated {len(validated_records)} records for {market} market, date {date}")
                
            except Exception as e:
                self.logger.error(f"Error processing {market} market data for {date}: {e}")
                results[market] = []
        
        return results
    
    def fetch_data_range_parallel(self, start_date: str, end_date: str, markets: List[str] = ['kospi'], is_backfill: bool = False) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Fetch data for a range of dates in parallel.
        
        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            markets (List[str]): List of markets to fetch from
            is_backfill (bool): Whether this is a backfill operation
            
        Returns:
            Dict[str, Dict[str, List[Dict]]]: Nested dictionary with dates as outer keys and markets as inner keys
        """
        from datetime import datetime, timedelta
        
        # Validate date range
        try:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            return {}
        
        if start_dt > end_dt:
            self.logger.error("Start date cannot be after end date")
            return {}
        
        # Generate date range
        current_date = start_dt
        dates_to_fetch = []
        
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y%m%d')
            
            # Skip weekends (KRX is closed on weekends)
            if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
                dates_to_fetch.append(date_str)
            
            current_date += timedelta(days=1)
        
        if not dates_to_fetch:
            self.logger.info("No trading dates found in the specified range")
            return {}
        
        self.logger.info(f"Fetching data for {len(dates_to_fetch)} trading dates in parallel")
        
        results = {}
        futures = {}
        
        # Submit all date requests in parallel
        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            for date in dates_to_fetch:
                future = executor.submit(self.fetch_data_for_date_parallel, date, markets, is_backfill)
                futures[future] = date
        
        # Collect results
        for future in as_completed(futures):
            date = futures[future]
            try:
                date_results = future.result()
                if date_results:
                    results[date] = date_results
                    total_records = sum(len(records) for records in date_results.values())
                    self.logger.info(f"Successfully fetched {total_records} records for date {date}")
                else:
                    self.logger.warning(f"No data fetched for date {date}")
                
            except Exception as e:
                self.logger.error(f"Error fetching data for date {date}: {e}")
        
        return results
    
    def validate_auth_key(self) -> bool:
        """
        Validate the authentication key by making a test request.
        
        Returns:
            bool: True if auth key is valid, False otherwise
        """
        # Use a recent date for testing
        test_date = datetime.now().strftime('%Y%m%d')
        
        data = self._make_request(test_date)
        
        if not data:
            self.logger.error("Failed to validate auth key - no response")
            return False
        
        if "OutBlock_1" not in data:
            self.logger.error("Failed to validate auth key - invalid response structure")
            return False
        
        self.logger.info("Auth key validation successful")
        return True
    
    def get_data_for_stock(self, stock_code: str, start_date: str, end_date: str) -> List[Dict]:
        """
        Fetch data for a specific stock over a date range.
        Note: This requires fetching all data and filtering, as KRX API doesn't support stock filtering.
        
        Args:
            stock_code (str): Stock code to fetch data for
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            
        Returns:
            List[Dict]: List of stock data records for the specified stock
        """
        # Fetch all data for the date range
        all_data = self.fetch_data_range(start_date, end_date)
        
        # Filter for the specific stock
        filtered_data = []
        for date, records in all_data.items():
            for record in records:
                if record.get('ISU_CD') == stock_code:
                    filtered_data.append(record)
        
        self.logger.info(f"Found {len(filtered_data)} records for stock {stock_code}")
        return filtered_data
    
    def get_available_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        Get list of dates that have data available (not holidays/weekends).
        
        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            
        Returns:
            List[str]: List of available dates
        """
        from datetime import datetime, timedelta
        
        try:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
        except ValueError as e:
            self.logger.error(f"Invalid date format: {e}")
            return []
        
        available_dates = []
        current_date = start_dt
        
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y%m%d')
            
            # Skip weekends
            if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
                available_dates.append(date_str)
            
            current_date += timedelta(days=1)
        
        return available_dates
    
    def estimate_data_size(self, start_date: str, end_date: str) -> Tuple[int, int]:
        """
        Estimate the amount of data that will be fetched.
        
        Args:
            start_date (str): Start date in YYYYMMDD format
            end_date (str): End date in YYYYMMDD format
            
        Returns:
            Tuple[int, int]: (estimated_records, estimated_api_calls)
        """
        available_dates = self.get_available_dates(start_date, end_date)
        estimated_calls = len(available_dates)
        
        # Make a test request to estimate records per day
        if available_dates:
            test_data = self.fetch_data_for_date(available_dates[0])
            estimated_records_per_day = len(test_data) if test_data else 2000  # Conservative estimate
        else:
            estimated_records_per_day = 2000
        
        estimated_records = estimated_calls * estimated_records_per_day
        
        self.logger.info(f"Estimated {estimated_records} records from {estimated_calls} API calls")
        return estimated_records, estimated_calls
    
    def fetch_data_for_date_multi_market(self, date: str, markets: List[str] = ['kospi']) -> Dict[str, List[Dict]]:
        """
        Fetch stock data for a specific date from multiple markets.
        
        Args:
            date (str): Date in YYYYMMDD format
            markets (List[str]): List of markets to fetch from ['kospi', 'kosdaq', 'kodex']
            
        Returns:
            Dict[str, List[Dict]]: Dictionary with market names as keys and data as values
        """
        results = {}
        
        for market in markets:
            data = self._make_request(date, market)
            
            if not data:
                results[market] = []
                continue
            
            if "OutBlock_1" not in data:
                self.logger.error(f"Invalid response structure for {market} market, date {date}")
                results[market] = []
                continue
            
            records = data["OutBlock_1"]
            validated_records = []
            
            for record in records:
                validated_record = self._validate_and_clean_record(record, date, market)
                if validated_record:
                    validated_records.append(validated_record)
            
            results[market] = validated_records
            self.logger.info(f"Validated {len(validated_records)} records for {market} market, date {date}")
        
        return results
    
    def fetch_all_markets_for_date(self, date: str) -> Dict[str, List[Dict]]:
        """
        Fetch data from all three markets for a specific date.

        Args:
            date (str): Date in YYYYMMDD format

        Returns:
            Dict[str, List[Dict]]: Dictionary with market names as keys and data as values
        """
        return self.fetch_data_for_date_multi_market(date, ['kospi', 'kosdaq', 'kodex'])

    # ============================================================
    # Index Data Methods (KOSPI/KOSDAQ indices, Bonds, Derivatives)
    # ============================================================

    def _make_index_request(self, date: str, index_type: str, is_backfill: bool = False) -> Optional[Dict]:
        """
        Make API request for index data with rate limiting.

        Args:
            date (str): Date in YYYYMMDD format
            index_type (str): One of 'kospi_index', 'kosdaq_index', 'bond_index', 'govt_bond', 'derivatives'
            is_backfill (bool): Whether this is a backfill operation

        Returns:
            Optional[Dict]: API response data or None if failed
        """
        if index_type not in self.index_endpoints:
            self.logger.error(f"Unsupported index type: {index_type}")
            return None

        base_url = self.index_endpoints[index_type]

        # Use optimized rate limiting for backfill
        delay = self.backfill_request_delay if is_backfill else self.request_delay

        # Thread-safe rate limiting
        with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < delay:
                time.sleep(delay - time_since_last)

        params = {"basDd": date}

        try:
            self.logger.info(f"Making API request for {index_type}, date: {date}")
            response = requests.get(base_url, headers=self.headers, params=params, timeout=30)

            with self._rate_limit_lock:
                self.last_request_time = time.time()

            if response.status_code == 200:
                try:
                    data = response.json()
                    self.logger.info(f"Successfully fetched {index_type} data for {date}")
                    return data
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error for {index_type}, date {date}: {e}")
                    return None
            else:
                self.logger.error(f"API request failed for {index_type}, {date}: HTTP {response.status_code}")
                self.logger.error(f"Response: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request exception for {index_type}, date {date}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error for {index_type}, date {date}: {e}")
            return None

    def fetch_index_data(self, date: str, index_type: str, is_backfill: bool = False) -> List[Dict]:
        """
        Fetch index data for a specific date.

        Args:
            date (str): Date in YYYYMMDD format
            index_type (str): One of 'kospi_index', 'kosdaq_index', 'bond_index', 'govt_bond', 'derivatives'
            is_backfill (bool): Whether this is a backfill operation

        Returns:
            List[Dict]: List of index data records from OutBlock_1
        """
        data = self._make_index_request(date, index_type, is_backfill)

        if not data:
            return []

        if "OutBlock_1" not in data:
            self.logger.error(f"Invalid response structure for {index_type}, date {date}")
            return []

        records = data["OutBlock_1"]
        self.logger.info(f"Fetched {len(records)} records for {index_type}, date {date}")
        return records

    def fetch_all_index_data(self, date: str, is_backfill: bool = False) -> Dict[str, List[Dict]]:
        """
        Fetch all index data types for a specific date.

        Args:
            date (str): Date in YYYYMMDD format
            is_backfill (bool): Whether this is a backfill operation

        Returns:
            Dict[str, List[Dict]]: Dictionary with index type as key and records as value
        """
        result = {}
        for index_type in self.index_endpoints.keys():
            try:
                result[index_type] = self.fetch_index_data(date, index_type, is_backfill)
            except Exception as e:
                self.logger.error(f"Failed to fetch {index_type} for {date}: {e}")
                result[index_type] = []
        return result

    def fetch_index_data_parallel(self, date: str, index_types: List[str] = None, is_backfill: bool = False) -> Dict[str, List[Dict]]:
        """
        Fetch multiple index data types for a specific date in parallel.

        Args:
            date (str): Date in YYYYMMDD format
            index_types (List[str], optional): List of index types to fetch. If None, fetches all.
            is_backfill (bool): Whether this is a backfill operation

        Returns:
            Dict[str, List[Dict]]: Dictionary with index type as key and records as value
        """
        if index_types is None:
            index_types = list(self.index_endpoints.keys())

        if not self.enable_parallel_processing or len(index_types) == 1:
            # Fall back to sequential processing
            result = {}
            for index_type in index_types:
                result[index_type] = self.fetch_index_data(date, index_type, is_backfill)
            return result

        results = {}
        futures = {}

        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            for index_type in index_types:
                future = executor.submit(self._make_index_request, date, index_type, is_backfill)
                futures[future] = index_type

        for future in as_completed(futures):
            index_type = futures[future]
            try:
                data = future.result()
                if not data:
                    results[index_type] = []
                    continue

                if "OutBlock_1" not in data:
                    self.logger.error(f"Invalid response structure for {index_type}, date {date}")
                    results[index_type] = []
                    continue

                records = data["OutBlock_1"]
                results[index_type] = records
                self.logger.info(f"Fetched {len(records)} records for {index_type}, date {date}")

            except Exception as e:
                self.logger.error(f"Error processing {index_type} data for {date}: {e}")
                results[index_type] = []

        return results
