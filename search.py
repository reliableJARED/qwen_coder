from ddgs import DDGS
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import urllib.error
import random
# Define your search query
query = "today's gaza israel news"
dd = DDGS()

# Perform the search
results = DDGS().text(query, max_results=5)
print(results)

# Print the results
for result in results:
    for key, value in result.items():
        print(f"{key.upper()}: {value}\n")

def open_url(url):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
        "Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36",
        "Mozilla/5.0 (Android 10; Mobile; rv:91.0) Gecko/91.0 Firefox/91.0",
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 OPR/77.0.4054.141",
        "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (iPad; CPU OS 14_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Linux; Android 10; SM-G975F Build/QP1A.190711.002; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/87.0.4280.141 Mobile Safari/537.36 UCBrowser/12.13.1.1007",
        "Mozilla/5.0 (Linux; Android 10; SM-G975F Build/QP1A.190711.002; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/87.0.4280.141 Mobile Safari/537.36 SamsungBrowser/14.0",
        "Mozilla/5.0 (Kindle Fire; Linux; en_US) AppleWebKit/537.36 (KHTML, like Gecko) Silk/4.6 Safari/537.36",
        "Opera Mini/60.0.3200.585 Version/12.0.2254.115 Mobile Safari/537.36",
        "Mozilla/5.0 (Linux; Android 10; SM-G975F Build/QP1A.190711.002; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/87.0.4280.141 Mobile Safari/537.36 Baiduspider/2.0 (compatible; Baiduspider/2.0; +http://www.baidu.com/search/spider.html)"
    ]
    headers = {
        'User-Agent': random.choice(user_agents)
    }
    try:
        page = urllib.request.Request(url, headers=headers)
        html = urllib.request.urlopen(page).read()
        return html
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code} while fetching {url}. Reason: {e.reason}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching {url}: {e}")
        return None

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    texts = soup.find_all(text=True)
    visible_texts = filter(tag_visible, texts)
    website_text = " ".join(t.strip() for t in visible_texts)
    return website_text

# Process each result
for result in results:
    url = result['href']
    print(f"Fetching content from: {url}")
    
    # Open the URL and get the HTML content
    html_content = open_url(url)
    if html_content is None:
        continue  # Skip to the next URL if there was an error
    
    # Extract visible text from the HTML content
    website_text = text_from_html(html_content)
    
    # Print the extracted text
    print(website_text)