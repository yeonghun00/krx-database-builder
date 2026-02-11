"""
Web scraper for KRX Index website to fetch all index constituent data.

This script:
1. Scrapes https://index.krx.co.kr/contents/MKD/03/0304/03040200/MKD03040200.jsp
2. Extracts all index links with their parameters (idxCd, idxId, upmidCd)
3. Downloads CSV data for each index using OTP-based download
4. Saves to data/krx_indices/ folder

Example link:
<a href="/contents/MKD/03/0304/03040101/MKD03040101.jsp?upmidCd=0102&idxCd=1034&idxId=K2G02P">코스피 100</a>
"""

import requests
import pandas as pd
from io import StringIO
from datetime import datetime
from bs4 import BeautifulSoup
import os
import sys
import time
import re
from urllib.parse import urljoin, urlparse, parse_qs
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class KRXIndexScraper:
    """Scraper for KRX index constituent data."""
    
    def __init__(self):
        self.session = requests.Session()
        self.base_url = 'https://index.krx.co.kr'
        self.indices_list_url = 'https://index.krx.co.kr/contents/MKD/03/0304/03040200/MKD03040200.jsp'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        }
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'krx_indices')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def fetch_indices_page(self) -> Optional[str]:
        """Fetch the main indices listing page."""
        try:
            print(f"Fetching indices page: {self.indices_list_url}")
            response = self.session.get(self.indices_list_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            print(f"✅ Page fetched successfully ({len(response.text)} bytes)")
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to fetch indices page: {e}")
            return None
    
    def parse_index_links(self, html: str) -> List[Dict]:
        """Parse HTML to extract all index links with their parameters."""
        soup = BeautifulSoup(html, 'html.parser')
        indices = []
        
        # Find all links that match the pattern
        # Pattern: /contents/MKD/03/0304/03040101/MKD03040101.jsp?upmidCd=XXX&idxCd=XXXX&idxId=XXXXX
        links = soup.find_all('a', href=re.compile(r'/contents/MKD/03/0304/03040101/MKD03040101\.jsp'))
        
        print(f"\nFound {len(links)} index links")
        
        for link in links:
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # Parse URL parameters
            parsed = urlparse(href)
            params = parse_qs(parsed.query)
            
            # Extract parameters
            upmid_cd = params.get('upmidCd', [''])[0]
            idx_cd = params.get('idxCd', [''])[0]
            idx_id = params.get('idxId', [''])[0]
            
            if idx_cd and idx_id:
                # Map to OTP parameters
                index_info = {
                    'name': text,
                    'href': href,
                    'upmid_cd': upmid_cd,
                    'idx_cd': idx_cd,
                    'idx_id': idx_id,
                    # OTP parameters
                    'ind_tp_cd': upmid_cd[0] if upmid_cd else '1',  # First digit
                    'idx_ind_cd': idx_cd[-3:] if len(idx_cd) >= 3 else idx_cd,  # Last 3 digits
                    'idx_id': idx_id,
                }
                indices.append(index_info)
                print(f"  📊 {text}: idxCd={idx_cd}, idxId={idx_id}")
        
        return indices
    
    def generate_otp(self, index_info: Dict) -> Optional[str]:
        """Generate OTP token for an index."""
        otp_url = 'https://index.krx.co.kr/contents/COM/GenerateOTP.jspx'
        
        # Use today's date for the query
        today = datetime.now().strftime('%Y%m%d')
        
        otp_params = {
            'name': 'fileDown',
            'filetype': 'csv',
            'url': '/IDX/03/0304/03040101/mkd03040101T3_01',
            'ind_tp_cd': index_info['ind_tp_cd'],
            'idx_ind_cd': index_info['idx_ind_cd'],
            'idx_id': index_info['idx_id'],
            'lang': 'ko',
            'compst_isu_tp': '1',
            'schdate': today,
            'fromdate': today,
            'todate': today,
            'pagePath': '/contents/MKD/03/0304/03040101/MKD03040101T3.jsp',
        }
        
        headers = {
            **self.headers,
            'Referer': f'https://index.krx.co.kr/contents/MKD/03/0304/03040101/MKD03040101.jsp?upmidCd={index_info["upmid_cd"]}&idxCd={index_info["idx_cd"]}&idxId={index_info["idx_id"]}',
            'X-Requested-With': 'XMLHttpRequest',
        }
        
        try:
            response = self.session.get(otp_url, params=otp_params, headers=headers, timeout=30)
            if response.status_code == 200:
                otp_code = response.text.strip()
                if len(otp_code) > 20:
                    return otp_code
            print(f"⚠️  OTP generation failed: {response.status_code}")
            return None
        except Exception as e:
            print(f"⚠️  OTP request error: {e}")
            return None
    
    def download_csv(self, otp_code: str, filename: str) -> bool:
        """Download CSV using OTP code."""
        download_url = 'https://file.krx.co.kr/download.jspx'
        
        download_headers = {
            'Referer': 'https://index.krx.co.kr/',
            'User-Agent': self.headers['User-Agent'],
            'Accept': '*/*',
        }
        
        download_data = {'code': otp_code}
        
        try:
            response = self.session.post(download_url, data=download_data, 
                                         headers=download_headers, timeout=30)
            
            if response.status_code == 200 and len(response.content) > 100:
                output_path = os.path.join(self.output_dir, filename)
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                # Try to parse to get row count
                try:
                    df = pd.read_csv(StringIO(response.text), encoding='cp949')
                    print(f"   ✅ Downloaded {len(df)} rows")
                    return True
                except:
                    print(f"   ✅ Downloaded {len(response.content)} bytes")
                    return True
            else:
                print(f"   ❌ Download failed: {response.status_code}, {len(response.content)} bytes")
                return False
        except Exception as e:
            print(f"   ❌ Download error: {e}")
            return False
    
    def fetch_index_data(self, index_info: Dict) -> bool:
        """Fetch data for a single index."""
        print(f"\n📥 Fetching: {index_info['name']}")
        
        # Generate OTP
        otp_code = self.generate_otp(index_info)
        if not otp_code:
            return False
        
        # Download CSV
        safe_name = re.sub(r'[^\w\s-]', '', index_info['name']).strip().replace(' ', '_')
        filename = f"{safe_name}_{index_info['idx_id']}.csv"
        
        return self.download_csv(otp_code, filename)
    
    def run(self, delay: float = 1.0):
        """Run the full scraping process."""
        print("=" * 60)
        print("KRX Index Scraper")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")
        
        # Fetch page
        html = self.fetch_indices_page()
        if not html:
            return False
        
        # Parse indices
        indices = self.parse_index_links(html)
        if not indices:
            print("❌ No indices found")
            return False
        
        print(f"\n{'='*60}")
        print(f"Found {len(indices)} indices to download")
        print(f"{'='*60}")
        
        # Download each index
        success_count = 0
        for i, index_info in enumerate(indices, 1):
            print(f"\n[{i}/{len(indices)}] ", end='')
            if self.fetch_index_data(index_info):
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
    scraper = KRXIndexScraper()
    success = scraper.run(delay=1.0)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
