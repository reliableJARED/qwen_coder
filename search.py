# search.py
from ddgs import DDGS
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import urllib.error
import random

from bart_lg import TextSummarizer

class WebSearch:
    def __init__(self):
        self.search = DDGS()
        self.summarizer = TextSummarizer()

    def open_url(self,url):
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

    def tag_visible(self,element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        if element.name == 'br':
            return True  # Include <br> tags for line breaks
        return True

    def text_from_html(self,html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        texts = soup.find_all(string=True)
        visible_texts = filter(self.tag_visible, texts)
        website_text = " ".join(t.strip() for t in visible_texts)
        return website_text
    
    def summarize_text(self,text):
        summary = self.summarizer.summarize_text(text)
        return summary

    def askInternet(self,query,max_results=5):
        # Perform the search https://pypi.org/project/ddgs/
        results = self.search.text(query, backend="duckduckgo, bing, yahoo", max_results=max_results)
        print(results)
        vresults = self.search.videos(query, max_results=max_results)
        print(f"\n\n{vresults}\n\n")
        all_summaries = []
        # Process each result
        for result in results:
            url = result['href']
            snippet = result['body']
            print(f"Fetching content from: {url}\n\n")
            
            # Open the URL and get the HTML content
            html_content = self.open_url(url)
            if html_content is None:
                #the snippets have been useful for headline information sometimes behind paywall, keep it
                all_summaries.append(snippet)
                continue  # Skip to the next URL if there was an error
            
            # Extract visible text from the HTML content
            website_text = self.text_from_html(html_content)
            
            # Print the extracted text
            summary = self.summarizer.summarize_text(website_text,query_match=query)
            print(snippet+":"+summary)
            all_summaries.append(snippet+":"+summary)

        return all_summaries

if __name__ == "__main__":
    ws = WebSearch()
    summary = ws.askInternet("did the Patriots win today?")
    print("\n\nALL SUMMARIES:\n\n")
    print(summary,"\n\n",len(summary))

    x = ws.summarize_text("".join(summary))
    print("\n\nSUMMARY OF ALL SUMMARIES:\n\n")
    print(x)