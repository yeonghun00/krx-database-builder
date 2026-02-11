"""
KRX Index Scraper using Selenium for listing page and requests for downloads.

This script:
1. Fetches the KRX index listing page (KOSPI or KOSDAQ) using Selenium
2. For each index, visits the detail page to get the REAL constituent codes
   (the listing page URLs have different codes than the constituent tab)
3. Downloads CSV data for each index using OTP
4. Saves to data/krx_indices/<market>/ folder

Usage:
    python fetch_krx_indices.py kospi
    python fetch_krx_indices.py kosdaq
    python fetch_krx_indices.py all       # both markets

Uses Selenium for JavaScript-rendered listing page, requests for OTP downloads.
"""

import argparse
import os
import sys
import time
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, parse_qs
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# webdriver_manager removed — Selenium 4.6+ has built-in driver management

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MARKET_CONFIG = {
    'kospi': {
        'listing_path': '/contents/MKD/03/0304/03040200/MKD03040200.jsp',
        'upmidCd_default': '0102',
    },
    'kosdaq': {
        'listing_path': '/contents/MKD/03/0304/03040300/MKD03040300.jsp',
        'upmidCd_default': '0103',
    },
}


class KRXIndexScraper:
    """Scraper for KRX indices (KOSPI or KOSDAQ)."""

    BASE_URL = 'https://index.krx.co.kr'
    OTP_URL = f'{BASE_URL}/contents/COM/GenerateOTP.jspx'
    DOWNLOAD_URL = 'https://file.krx.co.kr/download.jspx'

    def __init__(self, market: str = 'kospi'):
        if market not in MARKET_CONFIG:
            raise ValueError(f"Unknown market '{market}'. Choose from: {list(MARKET_CONFIG.keys())}")

        self.market = market
        self.config = MARKET_CONFIG[market]
        self.listing_url = f'{self.BASE_URL}{self.config["listing_path"]}'

        self.session = requests.Session()
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'krx_indices', market
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        }

        self.indices = self.scrape_listing_page()

    def _create_driver(self) -> webdriver.Chrome:
        """Create and configure a headless Chrome WebDriver."""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument(f'--user-agent={self.headers["User-Agent"]}')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        # Selenium 4.6+ auto-manages chromedriver (no webdriver-manager needed)
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script('Object.defineProperty(navigator, "webdriver", {get: () => undefined})')
        return driver

    def scrape_listing_page(self) -> List[Dict]:
        """Fetch the KOSPI index listing page using Selenium and extract index links."""
        indices = []
        driver = None

        print(f"Fetching {self.market.upper()} listing page with Selenium: {self.listing_url}")

        try:
            driver = self._create_driver()
            driver.get(self.listing_url)

            # Wait for the page to load - wait for table or links to be present
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="MKD03040101.jsp"]')))

            print(f"Page loaded successfully")

            # Find all links to index detail pages
            links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="MKD03040101.jsp"]')

            for link in links:
                href = link.get_attribute('href') or ''
                text = link.text.strip()

                if not text:
                    continue

                parsed = urlparse(href)
                params = parse_qs(parsed.query)

                upmid_cd = params.get('upmidCd', [''])[0]
                idx_cd = params.get('idxCd', [''])[0]
                idx_id = params.get('idxId', [''])[0]

                if idx_cd and idx_id:
                    indices.append({
                        'name': text,
                        'href': href,
                        'upmidCd': upmid_cd,
                        'idxCd': idx_cd,
                        'idxId': idx_id,
                    })

        except Exception as e:
            print(f"Error during Selenium scraping: {e}")
            raise
        finally:
            if driver:
                driver.quit()

        print(f"Found {len(indices)} index links on listing page")
        return indices

    def resolve_real_codes(self, index_info: Dict) -> Optional[Dict]:
        """
        Visit the index detail page and extract the REAL idxCd/idxId
        from the tab links (e.g. MKD03040101T1.jsp?idxCd=1167&idxId=K2Z01P).

        The listing page URLs have different codes than what the constituent
        download actually needs.
        """
        href = index_info.get('href', '')
        if not href:
            return None

        detail_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href

        try:
            response = self.session.get(detail_url, headers=self.headers, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"   Failed to fetch detail page: {e}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find tab links like MKD03040101T1.jsp, MKD03040101T3.jsp, etc.
        # These contain the REAL idxCd and idxId for this index
        tab_links = soup.find_all('a', href=re.compile(r'MKD03040101T\d\.jsp'))

        for tab_link in tab_links:
            tab_href = tab_link.get('href', '')
            parsed = urlparse(tab_href)
            params = parse_qs(parsed.query)

            real_idx_cd = params.get('idxCd', [''])[0]
            real_idx_id = params.get('idxId', [''])[0]
            real_upmid_cd = params.get('upmidCd', [''])[0]

            if real_idx_cd and real_idx_id:
                return {
                    'idxCd': real_idx_cd,
                    'idxId': real_idx_id,
                    'upmidCd': real_upmid_cd or index_info.get('upmidCd', self.config['upmidCd_default']),
                }
            

        return None

    def generate_otp(self, idx_cd: str, idx_id: str, upmid_cd: str) -> Optional[str]:
        """Generate OTP token using the real index codes."""
        from_date = '20260203'
        to_date = '20260210'

        ind_tp_cd = '1'
        idx_ind_cd = idx_cd[-3:] if len(idx_cd) >= 3 else idx_cd

        otp_params = {
            'name': 'fileDown',
            'filetype': 'csv',
            'url': '/IDX/03/0304/03040101/mkd03040101T3_01',
            'ind_tp_cd': ind_tp_cd,
            'idx_ind_cd': idx_ind_cd,
            'idx_id': idx_id,
            'lang': 'ko',
            'compst_isu_tp': '1',
            'schdate': from_date,
            'fromdate': from_date,
            'todate': to_date,
            'pagePath': '/contents/MKD/03/0304/03040101/MKD03040101T3.jsp',
        }

        headers = {
            **self.headers,
            'Referer': f'{self.BASE_URL}/contents/MKD/03/0304/03040101/MKD03040101.jsp?upmidCd={upmid_cd}&idxCd={idx_cd}&idxId={idx_id}',
            'X-Requested-With': 'XMLHttpRequest',
        }

        try:
            response = self.session.get(self.OTP_URL, params=otp_params,
                                        headers=headers, timeout=30)
            if response.status_code == 200:
                otp_code = response.text.strip()
                if len(otp_code) > 20:
                    return otp_code
            print(f"   OTP generation failed: {response.status_code}")
            return None
        except Exception as e:
            print(f"   OTP request error: {e}")
            return None

    def download_csv(self, otp_code: str, filename: str) -> bool:
        """Download CSV using OTP code. Returns True if CSV has data rows."""
        headers = {
            'Referer': f'{self.BASE_URL}/',
            'User-Agent': self.headers['User-Agent'],
            'Accept': '*/*',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Origin': self.BASE_URL,
        }

        try:
            response = self.session.post(self.DOWNLOAD_URL, data={'code': otp_code},
                                         headers=headers, timeout=30)

            if response.status_code != 200:
                print(f"   Download failed: HTTP {response.status_code}")
                return False

            csv_text = response.content.decode('cp949', errors='replace')
            df = pd.read_csv(StringIO(csv_text))
            if len(df) > 0:
                output_path = os.path.join(self.output_dir, filename)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"   Downloaded {len(df)} rows -> {filename}")
                return True
            else:
                print(f"   Empty CSV (header only)")
                return False
        except Exception as e:
            print(f"   Download error: {e}")
            return False

    def run(self, delay: float = 1.0):
        """Run the full scraping process."""
        print("=" * 60)
        print(f"{self.market.upper()} Index Scraper")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")

        indices = self.indices

        print(f"\n{'='*60}")
        print(f"Resolving real codes & downloading ({len(indices)} indices)")
        print(f"{'='*60}")

        success_count = 0
        for i, index_info in enumerate(indices, 1):
            name = index_info['name']
            listing_cd = index_info['idxCd']
            listing_id = index_info['idxId']
            upmid = index_info.get('upmidCd', self.config['upmidCd_default'])
            print(f"\n[{i}/{len(indices)}] {name}")
            print(f"   Listing codes: idxCd={listing_cd}, idxId={listing_id}")

            # Build detail page URL and resolve real constituent codes
            detail_info = {
                'href': f"/contents/MKD/03/0304/03040101/MKD03040101.jsp?upmidCd={upmid}&idxCd={listing_cd}&idxId={listing_id}",
                'upmidCd': upmid,
            }
            real_codes = self.resolve_real_codes(detail_info)
            if not real_codes:
                print(f"   Could not resolve real codes, using listing codes")
                real_codes = {
                    'idxCd': listing_cd,
                    'idxId': listing_id,
                    'upmidCd': upmid,
                }

            idx_cd = real_codes['idxCd']
            idx_id = real_codes['idxId']
            upmid_cd = real_codes['upmidCd']

            print(f"   Real codes: idxCd={idx_cd}, idxId={idx_id}")

            # Generate OTP with real codes
            otp_code = self.generate_otp(idx_cd, idx_id, upmid_cd)
            if not otp_code:
                continue

            # Download CSV
            safe_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
            filename = f"{safe_name}_{idx_id}.csv"

            if self.download_csv(otp_code, filename):
                success_count += 1

            # Rate limiting
            if i < len(indices):
                time.sleep(delay)

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total indices: {len(indices)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(indices) - success_count}")
        print(f"Output directory: {self.output_dir}")

        return success_count > 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='KRX Index Scraper')
    parser.add_argument('market', nargs='?', default='kospi',
                        choices=['kospi', 'kosdaq', 'all'],
                        help='Market to scrape (default: kospi)')
    args = parser.parse_args()

    markets = list(MARKET_CONFIG.keys()) if args.market == 'all' else [args.market]

    all_success = True
    for market in markets:
        print(f"\n{'#' * 60}")
        print(f"# Scraping {market.upper()}")
        print(f"{'#' * 60}\n")
        scraper = KRXIndexScraper(market=market)
        if not scraper.run(delay=1.0):
            all_success = False

    sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    main()
