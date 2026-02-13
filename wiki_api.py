import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
from typing import List, Optional, Dict


class WikipediaAPI:
    """
    A thread-safe wrapper for the Wikipedia Action API to fetch links from pages.
    Uses a shared requests.Session with connection pooling for performance.
    """

    USER_AGENT = (
        "WikipediaChallengerBot/1.0 "
        "(mailto:mael@example.com) requests/2.31.0"
    )

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.base_url = f"https://{lang}.wikipedia.org/w/api.php"
        
        # Shared session with connection pooling and retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})
        
        # Configure robust retries
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        # Pool size matches typical max threads
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_links(self, title: str) -> List[str]:
        """
        Fetches all internal links from a given Wikipedia page title.
        Handles pagination to get all links. Thread-safe.
        """
        links: List[str] = []
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "links",
            "pllimit": "max",
            "plnamespace": 0,
            "redirects": 1,
        }

        while True:
            try:
                # 10s timeout to prevent hanging
                response = self.session.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                pages = data.get("query", {}).get("pages", {})
                if not pages:
                    break

                for page_data in pages.values():
                    for link in page_data.get("links", []):
                        links.append(link["title"])

                if "continue" in data:
                    params.update(data["continue"])
                else:
                    break
            except Exception as e:
                # print(f"Error fetching links for {title}: {e}")
                break

        return links

    def validate_page(self, title: str) -> Optional[str]:
        """
        Validates if a page exists and returns its normalized title.
        """
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "redirects": 1,
        }
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if int(page_id) < 0:
                    return None
                return page_data["title"]
        except Exception:
            return None
        return None

    def get_metadata_batch(self, titles: List[str]) -> Dict[str, str]:
        """
        Fetches descriptions for a batch of titles (max 50).
        Returns a mapping of title -> description.
        """
        if not titles:
            return {}
            
        params = {
            "action": "query",
            "format": "json",
            "titles": "|".join(titles),
            "prop": "description",
            "redirects": 1,
        }
        
        results = {}
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get("query", {}).get("pages", {})
            for page_data in pages.values():
                title = page_data.get("title")
                description = page_data.get("description", "")
                if title:
                    results[title] = description
        except Exception as e:
            # print(f"Error fetching metadata batch: {e}")
            pass
            
        return results


if __name__ == "__main__":
    # Quick test
    api = WikipediaAPI()
    test_title = "France"
    print(f"Fetching links for: {test_title}")
    links = api.get_links(test_title)
    print(f"Found {len(links)} links. First 5: {links[:5]}")

