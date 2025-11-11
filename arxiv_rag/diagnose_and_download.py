#!/usr/bin/env python3
"""
Diagnose why ArXiv PDF downloads are failing
"""

import requests
import arxiv
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_network_connectivity():
    """Test basic network connection"""
    print("\n" + "="*60)
    print("TEST 1: Network Connectivity")
    print("="*60)
    
    test_urls = [
        "https://www.google.com",
        "https://arxiv.org",
        "https://export.arxiv.org"
    ]
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=10)
            status = "✓ PASS" if response.status_code == 200 else f"✗ FAIL ({response.status_code})"
            print(f"{status} - {url}")
        except Exception as e:
            print(f"✗ FAIL - {url}: {e}")
    
    return True

def test_arxiv_api():
    """Test ArXiv API access"""
    print("\n" + "="*60)
    print("TEST 2: ArXiv API Access")
    print("="*60)
    
    try:
        search = arxiv.Search(
            query="cat:cs.AI",
            max_results=1,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        result = next(search.results())
        print(f"✓ API works - Found paper: {result.title[:60]}...")
        print(f"  Paper ID: {result.entry_id}")
        print(f"  PDF URL: {result.pdf_url}")
        
        return result
    except Exception as e:
        print(f"✗ API failed: {e}")
        return None

def test_pdf_download_methods(result):
    """Try different methods to download PDF"""
    print("\n" + "="*60)
    print("TEST 3: PDF Download Methods")
    print("="*60)
    
    output_dir = Path("test_downloads")
    output_dir.mkdir(exist_ok=True)
    
    methods = []
    
    # Method 1: arxiv library's built-in download
    print("\nMethod 1: arxiv.Result.download_pdf()")
    try:
        pdf_path = output_dir / "method1.pdf"
        result.download_pdf(str(pdf_path))
        
        if pdf_path.exists():
            size = pdf_path.stat().st_size
            print(f"✓ SUCCESS - Downloaded {size / 1024:.1f} KB")
            methods.append(("Method 1", True))
        else:
            print("✗ FAIL - File not created")
            methods.append(("Method 1", False))
    except Exception as e:
        print(f"✗ FAIL - {e}")
        methods.append(("Method 1", False))
    
    # Method 2: requests with standard URL
    print("\nMethod 2: requests.get() with result.pdf_url")
    try:
        pdf_path = output_dir / "method2.pdf"
        response = requests.get(result.pdf_url, timeout=30, stream=True)
        response.raise_for_status()
        
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size = pdf_path.stat().st_size
        print(f"✓ SUCCESS - Downloaded {size / 1024:.1f} KB")
        methods.append(("Method 2", True))
    except Exception as e:
        print(f"✗ FAIL - {e}")
        methods.append(("Method 2", False))
    
    # Method 3: requests with modified URL (export.arxiv.org)
    print("\nMethod 3: requests.get() with export.arxiv.org")
    try:
        pdf_path = output_dir / "method3.pdf"
        # Convert arxiv.org/pdf/ID.pdf to export.arxiv.org/pdf/ID
        paper_id = result.entry_id.split('/')[-1]
        export_url = f"https://export.arxiv.org/pdf/{paper_id}"
        
        print(f"  URL: {export_url}")
        response = requests.get(export_url, timeout=30, stream=True)
        response.raise_for_status()
        
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size = pdf_path.stat().st_size
        print(f"✓ SUCCESS - Downloaded {size / 1024:.1f} KB")
        methods.append(("Method 3", True))
    except Exception as e:
        print(f"✗ FAIL - {e}")
        methods.append(("Method 3", False))
    
    # Method 4: requests with user agent
    print("\nMethod 4: requests.get() with User-Agent header")
    try:
        pdf_path = output_dir / "method4.pdf"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(result.pdf_url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size = pdf_path.stat().st_size
        print(f"✓ SUCCESS - Downloaded {size / 1024:.1f} KB")
        methods.append(("Method 4", True))
    except Exception as e:
        print(f"✗ FAIL - {e}")
        methods.append(("Method 4", False))
    
    return methods

def test_pdf_parsing():
    """Test if we can parse downloaded PDFs"""
    print("\n" + "="*60)
    print("TEST 4: PDF Parsing")
    print("="*60)
    
    test_dir = Path("test_downloads")
    pdf_files = list(test_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("✗ No PDFs to test")
        return False
    
    print(f"Found {len(pdf_files)} PDF(s) to test")
    
    try:
        from PyPDF2 import PdfReader
        
        for pdf_path in pdf_files:
            print(f"\nTesting: {pdf_path.name}")
            try:
                reader = PdfReader(pdf_path)
                num_pages = len(reader.pages)
                
                # Try to extract text from first page
                first_page_text = reader.pages[0].extract_text()
                
                print(f"  ✓ Pages: {num_pages}")
                print(f"  ✓ First page text length: {len(first_page_text)} chars")
                print(f"  ✓ Sample: {first_page_text[:100]}...")
                
            except Exception as e:
                print(f"  ✗ Failed to parse: {e}")
        
        return True
        
    except ImportError:
        print("✗ PyPDF2 not installed: pip install PyPDF2")
        return False

def generate_recommendation(methods):
    """Generate recommendation based on test results"""
    print("\n" + "="*60)
    print("DIAGNOSIS & RECOMMENDATION")
    print("="*60)
    
    working_methods = [m for m, success in methods if success]
    
    if not working_methods:
        print("""
✗ ALL DOWNLOAD METHODS FAILED

Possible causes:
1. Firewall/proxy blocking ArXiv
2. Network restrictions
3. ArXiv temporarily blocking your IP
4. SSL/TLS issues

SOLUTIONS TO TRY:

A) Use a VPN or different network
   - Try mobile hotspot
   - Try different WiFi network

B) Use abstract-only mode (no PDFs needed):
   # Edit your pipeline to skip PDF downloads
   # I can provide code for this

C) Download papers manually:
   - Go to arxiv.org
   - Search for papers
   - Download PDFs to data/raw/
   - Run processing only

D) Check proxy settings:
   export HTTP_PROXY="http://your-proxy:port"
   export HTTPS_PROXY="http://your-proxy:port"

Which solution would you like to try?
""")
    else:
        print(f"\n✓ WORKING METHOD FOUND: {working_methods[0]}")
        print(f"\nI'll update the code to use this method.")
        
        if working_methods[0] == "Method 3":
            print("\nUse export.arxiv.org URLs instead of arxiv.org")
        elif working_methods[0] == "Method 4":
            print("\nUse User-Agent headers in requests")

def run_full_diagnosis():
    """Run complete diagnostic"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           ArXiv Download Diagnostic Tool                     ║
╚══════════════════════════════════════════════════════════════╝

This will test:
1. Network connectivity
2. ArXiv API access  
3. Multiple PDF download methods
4. PDF parsing

""")
    
    # Test 1
    test_network_connectivity()
    
    # Test 2
    result = test_arxiv_api()
    if not result:
        print("\n✗ Cannot access ArXiv API. Check network connection.")
        return
    
    # Test 3
    methods = test_pdf_download_methods(result)
    
    # Test 4
    test_pdf_parsing()
    
    # Recommendation
    generate_recommendation(methods)
    
    print("\n" + "="*60)
    print("Diagnostic complete!")
    print("="*60)

if __name__ == "__main__":
    run_full_diagnosis()