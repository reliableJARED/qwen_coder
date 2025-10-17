# search.py
from ddgs import DDGS
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import urllib.error
import random

from bart_lg import Summarizer


# Define your search query
query = "what do people think of Taylor Swift's new album?"
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
        #Chrome on Windows:
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        #Chrome on macOS:
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        #Chrome on Linux:
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"]
    headers = {
        'User-Agent': random.choice(user_agents)
    }
    try:
        page = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(page)
        html = response.read()
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
    if element.name == 'br':
        return True  # Include <br> tags for line breaks
    return True

def text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    texts = soup.find_all(text=True)
    visible_texts = filter(tag_visible, texts)
    website_text = " ".join(t.strip() for t in visible_texts)
    return website_text

summarizer = Summarizer()
all_summaries = []
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
    all_summaries.append(summarizer.summarize_text(website_text,query_match=query))

print("\n\nALL SUMMARIES:\n\n")
print(all_summaries,"\n\n",len(all_summaries))

x = summarizer.summarize_text("".join(all_summaries))
print("\n\nSUMMARY OF ALL SUMMARIES:\n\n")
print(x)