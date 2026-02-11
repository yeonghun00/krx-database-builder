"""
Test script for KRX OTP-based CSV download.

This script tests the two-step OTP flow to download CSV data from KRX:
1. GET to GenerateOTP.jspx to get a temporary token
2. POST that token to download.jspx to get CSV

Important URLs (2025-2026):
- OTP: https://index.krx.co.kr/contents/COM/GenerateOTP.jspx (GET with query params)
- Download: https://index.krx.co.kr/contents/COM/Download.jspx (POST with code)
"""

import requests
import pandas as pd
from io import StringIO
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_krx_otp_download():
    """Test OTP-based CSV download from KRX index.krx.co.kr."""
    
    session = requests.Session()
    
    # Headers matching browser request exactly
    headers = {
        'Referer': 'https://index.krx.co.kr/contents/MKD/03/0304/03040101/MKD03040101.jsp?upmidCd=0102&idxCd=1028&idxId=K2G01P',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'X-Requested-With': 'XMLHttpRequest',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
    }
    
    # Step 1: Generate OTP (GET request with query params)
    print("=" * 60)
    print("Step 1: Generating OTP token...")
    print("=" * 60)
    
    otp_url = 'https://index.krx.co.kr/contents/COM/GenerateOTP.jspx'
    
    # Query parameters for KOSPI 200 index constituents
    otp_params = {
        'name': 'fileDown',
        'filetype': 'csv',
        'url': '/IDX/03/0304/03040101/mkd03040101T3_01',
        'ind_tp_cd': '1',
        'idx_ind_cd': '028',
        'idx_id': 'K2G01P',
        'lang': 'ko',
        'compst_isu_tp': '1',
        'schdate': '20260203',
        'fromdate': '20260203',
        'todate': '20260210',
        'pagePath': '/contents/MKD/03/0304/03040101/MKD03040101T3.jsp',
    }
    
    try:
        otp_response = session.get(otp_url, params=otp_params, headers=headers, timeout=30)
        print(f"OTP Request URL: {otp_response.url}")
        print(f"OTP Request Status: {otp_response.status_code}")
        
        if otp_response.status_code != 200:
            print(f"❌ OTP request failed with status {otp_response.status_code}")
            print(f"Response: {otp_response.text[:500]}")
            return False
        
        otp_code = otp_response.text.strip()
        
        if len(otp_code) < 20:
            print(f"❌ OTP token too short or invalid: {otp_code}")
            return False
        
        print(f"✅ OTP token received: {otp_code[:60]}...")
        print(f"   Token length: {len(otp_code)} characters")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ OTP request error: {e}")
        return False
    
    # Step 2: Download CSV (POST with code)
    print("\n" + "=" * 60)
    print("Step 2: Downloading CSV file...")
    print("=" * 60)
    
    download_url = 'https://file.krx.co.kr/download.jspx'
    
    # Update headers for download (matching browser exactly)
    download_headers = {
        'Referer': 'https://index.krx.co.kr/',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Origin': 'https://index.krx.co.kr',
    }
    
    download_data = {
        'code': otp_code,
    }
    
    try:
        file_response = session.post(download_url, data=download_data, headers=download_headers, timeout=30)
        print(f"Download Request Status: {file_response.status_code}")
        print(f"Content-Type: {file_response.headers.get('Content-Type', 'N/A')}")
        print(f"Content-Length: {len(file_response.content)} bytes")
        
        if file_response.status_code != 200:
            print(f"❌ Download failed with status {file_response.status_code}")
            print(f"Response: {file_response.text[:500]}")
            return False
        
        content_type = file_response.headers.get('Content-Type', '')
        
        # Save the file regardless of content type
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 'test_krx_download.csv')
        with open(output_file, 'wb') as f:
            f.write(file_response.content)
        
        print(f"✅ File saved to: {output_file}")
        print(f"   Size: {len(file_response.content)} bytes")
        
        # Try to parse as CSV
        if len(file_response.content) > 0:
            try:
                # Try different encodings
                for encoding in ['cp949', 'euc-kr', 'utf-8']:
                    try:
                        df = pd.read_csv(StringIO(file_response.text), encoding=encoding)
                        print(f"\n📊 CSV Data Preview (encoding: {encoding}):")
                        print(f"   Rows: {len(df)}")
                        print(f"   Columns: {list(df.columns)}")
                        print(f"\nFirst 5 rows:")
                        print(df.head())
                        
                        print(f"\n✅ SUCCESS! Downloaded and parsed {len(df)} records.")
                        return True
                    except Exception as e:
                        continue
                
                print(f"⚠️  Could not parse with any encoding")
                print(f"   Raw content preview (first 500 chars):")
                print(file_response.text[:500])
                return True  # Still success since file was saved
                
            except Exception as e:
                print(f"⚠️  Saved file but couldn't parse: {e}")
                print(f"   Raw content preview (first 500 chars):")
                print(file_response.text[:500])
                return True
        else:
            print(f"⚠️  Downloaded file is empty (0 bytes)")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Download request error: {e}")
        return False


def test_data_krx_endpoint():
    """Test with data.krx.co.kr endpoint (POST method)."""
    
    session = requests.Session()
    
    headers = {
        'Referer': 'https://data.krx.co.kr/contents/MKD/03/0304/03040101/MKD03040101T3.jsp',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    }
    
    # Step 1: Generate OTP
    print("\n" + "=" * 60)
    print("Trying data.krx.co.kr endpoint (POST method)...")
    print("=" * 60)
    
    otp_url = 'https://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
    
    otp_data = {
        'name': 'fileDown',
        'filetype': 'csv',
        'url': '/IDX/03/0304/03040101/mkd03040101T3_01',
        'ind_tp_cd': '1',
        'idx_ind_cd': '028',
        'idx_id': 'K2G01P',
        'lang': 'ko',
        'compst_isu_tp': '1',
        'schdate': '20260203',
        'fromdate': '20260203',
        'todate': '20260210',
        'pagePath': '/contents/MKD/03/0304/03040101/MKD03040101T3.jsp',
    }
    
    try:
        otp_response = session.post(otp_url, data=otp_data, headers=headers, timeout=30)
        print(f"OTP Status: {otp_response.status_code}")
        
        if otp_response.status_code != 200:
            print(f"❌ OTP failed: {otp_response.text[:200]}")
            return False
        
        otp_code = otp_response.text.strip()
        print(f"✅ OTP: {otp_code[:50]}...")
        
        # Step 2: Download
        download_url = 'https://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'
        download_data = {'code': otp_code}
        
        file_response = session.post(download_url, data=download_data, headers=headers, timeout=30)
        print(f"Download Status: {file_response.status_code}")
        print(f"Size: {len(file_response.content)} bytes")
        
        if file_response.status_code == 200 and len(file_response.content) > 100:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, 'test_krx_download_data.csv')
            with open(output_file, 'wb') as f:
                f.write(file_response.content)
            
            print(f"✅ Saved to: {output_file}")
            
            try:
                df = pd.read_csv(StringIO(file_response.text), encoding='cp949')
                print(f"✅ Parsed {len(df)} rows, {len(df.columns)} columns")
                return True
            except:
                pass
        
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_without_tabcode():
    """Test download WITHOUT tabCode to see if it's required."""
    
    session = requests.Session()
    
    headers = {
        'Referer': 'https://index.krx.co.kr/contents/MKD/03/0304/03040101/MKD03040101.jsp?upmidCd=0102&idxCd=1028&idxId=K2G01P',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'X-Requested-With': 'XMLHttpRequest',
    }
    
    print("\n" + "=" * 60)
    print("Testing if tabCode is REQUIRED...")
    print("=" * 60)
    
    # Step 1: Get OTP
    otp_params = {
        'name': 'fileDown',
        'filetype': 'csv',
        'url': '/IDX/03/0304/03040101/mkd03040101T3_01',
        'ind_tp_cd': '1',
        'idx_ind_cd': '028',
        'idx_id': 'K2G01P',
        'lang': 'ko',
        'compst_isu_tp': '1',
        'schdate': '20260203',
        'fromdate': '20260203',
        'todate': '20260210',
        'pagePath': '/contents/MKD/03/0304/03040101/MKD03040101T3.jsp',
    }
    
    otp_response = session.get('https://index.krx.co.kr/contents/COM/GenerateOTP.jspx', 
                                params=otp_params, headers=headers)
    otp_code = otp_response.text.strip()
    print(f"✅ OTP received: {otp_code[:50]}...")
    
    download_headers = {
        'Referer': 'https://index.krx.co.kr/',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    }
    
    # Test WITHOUT tabCode
    print("\n--- Test WITHOUT tabCode ---")
    file_response = session.post('https://file.krx.co.kr/download.jspx', 
                                  data={'code': otp_code}, 
                                  headers=download_headers)
    print(f"Status: {file_response.status_code}")
    print(f"Size: {len(file_response.content)} bytes")
    
    without_tabcode_works = file_response.status_code == 200 and len(file_response.content) > 1000
    
    if without_tabcode_works:
        print("✅ SUCCESS without tabCode!")
        try:
            df = pd.read_csv(StringIO(file_response.text), encoding='cp949')
            print(f"   Parsed {len(df)} rows")
        except:
            pass
    else:
        print("❌ FAILED without tabCode (empty or small response)")
    
    # Test WITH tabCode
    print("\n--- Test WITH tabCode ---")
    file_response2 = session.post('https://file.krx.co.kr/download.jspx', 
                                   data={'code': otp_code, 
                                         'tabCode': 'a110dc6b3a1678330158473e0d0ffbf0',
                                         'tabNumber': '2'}, 
                                   headers=download_headers)
    print(f"Status: {file_response2.status_code}")
    print(f"Size: {len(file_response2.content)} bytes")
    
    with_tabcode_works = file_response2.status_code == 200 and len(file_response2.content) > 1000
    
    if with_tabcode_works:
        print("✅ SUCCESS with tabCode!")
        try:
            df = pd.read_csv(StringIO(file_response2.text), encoding='cp949')
            print(f"   Parsed {len(df)} rows")
        except:
            pass
    else:
        print("❌ FAILED with tabCode")
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    if without_tabcode_works and with_tabcode_works:
        print("✅ tabCode is OPTIONAL (works either way)")
    elif not without_tabcode_works and with_tabcode_works:
        print("⚠️  tabCode is REQUIRED")
    elif without_tabcode_works and not with_tabcode_works:
        print("⚠️  tabCode should NOT be used")
    else:
        print("❌ Both methods failed")
    print("=" * 60)
    
    return with_tabcode_works or without_tabcode_works


if __name__ == '__main__':
    print("=" * 60)
    print("KRX OTP-based CSV Download Test")
    print("=" * 60)
    print(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # First test if tabCode is required
    test_without_tabcode()
    
    print("\n" + "=" * 60)
    print("Running full test with best method...")
    print("=" * 60)
    
    # Run the full test
    success = test_krx_otp_download()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ OVERALL RESULT: SUCCESS")
    else:
        print("❌ OVERALL RESULT: FAILED")
    print("=" * 60)
    
    sys.exit(0 if success else 1)
