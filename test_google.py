"""
Google search with Playwright - handles JavaScript rendering (2025+)
"""

import asyncio
import json
from playwright.async_api import async_playwright


async def google_search(query):
    """
    Perform a Google search using Playwright with JavaScript rendering
    
    Args:
        query: Search query string
    
    Returns:
        List of dictionaries containing search results
    """
    print(f"Searching Google for: {query}")
    print("-" * 50)
    
    async with async_playwright() as p:
        # Launch browser (headless=True for production, False for debugging)
        browser = await p.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled']
        )
        
        # Create context with realistic settings
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            timezone_id='America/New_York',
        )
        
        # Create new page
        page = await context.new_page()
        
        try:
            # Navigate to Google search
            search_url = f"https://www.google.com/search?q={query}&num=10"
            print(f"Navigating to: {search_url}")
            
            await page.goto(search_url, wait_until='networkidle')
            print("✓ Page loaded successfully")
            
            # Wait for search results to appear
            await page.wait_for_selector('div#search', timeout=10000)
            print("✓ Search results container found")
            
            # Extract search results
            results = []
            
            # Try multiple selector strategies
            result_selectors = [
                'div.g',  # Standard result container
                'div.tF2Cxc',  # 2024-2025 structure
            ]
            
            for selector in result_selectors:
                containers = await page.query_selector_all(selector)
                if containers:
                    print(f"✓ Found {len(containers)} results with selector: {selector}")
                    break
            
            if not containers:
                print("✗ No result containers found")
                return []
            
            for container in containers:
                try:
                    # Extract title
                    title_elem = await container.query_selector('h3')
                    if not title_elem:
                        continue
                    title = await title_elem.inner_text()
                    
                    # Extract URL
                    link_elem = await container.query_selector('a')
                    if not link_elem:
                        continue
                    url = await link_elem.get_attribute('href')
                    
                    # Skip invalid links
                    if not url or url.startswith('#') or '/search?' in url:
                        continue
                    
                    # Extract snippet
                    snippet = ''
                    snippet_selectors = ['.VwiC3b', '.lyLwlc', '.s3v9rd', 'div[data-sncf="1"]']
                    for snip_sel in snippet_selectors:
                        snippet_elem = await container.query_selector(snip_sel)
                        if snippet_elem:
                            snippet = await snippet_elem.inner_text()
                            break
                    
                    results.append({
                        'title': title.strip(),
                        'url': url,
                        'snippet': snippet.strip()
                    })
                    
                except Exception as e:
                    # Skip problematic results
                    continue
            
            print(f"✓ Successfully extracted {len(results)} results")
            return results
            
        except Exception as e:
            print(f"✗ Error: {type(e).__name__}: {e}")
            return []
        
        finally:
            await browser.close()


def print_results(results):
    """Pretty print search results"""
    if not results:
        print("\nNo results to display")
        return
    
    print(f"\n{'=' * 100}")
    print(f"SEARCH RESULTS ({len(results)} found)")
    print('=' * 100)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        if result['snippet']:
            snippet = result['snippet']
            if len(snippet) > 200:
                snippet = snippet[:200] + '...'
            print(f"   {snippet}")
        print('-' * 100)



async def main():
    print("=" * 50)
    print("GOOGLE SEARCH WITH PLAYWRIGHT (2025)")
    print("=" * 50)
    
    # Test 1: Basic search
    print("\n\nTest 1: Python Programming")
    print("=" * 50)
    results = await google_search("python programming")
    print_results(results)
    
  
    # Test 2: Different query
    print("\n\n" + "=" * 50)
    print("Test 2: Machine Learning")
    print("=" * 50)
    results2 = await google_search("machine learning")
    print_results(results2[:5])  # Show top 5
   


if __name__ == "__main__":
    asyncio.run(main())