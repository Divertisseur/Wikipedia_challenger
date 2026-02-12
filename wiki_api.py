import requests
import threading
from typing import List, Optional


class WikipediaAPI:
    """
    A thread-safe wrapper for the Wikipedia Action API to fetch links from pages.
    Each thread gets its own ``requests.Session`` via ``threading.local()``.
    """

    USER_AGENT = (
        "WikipediaChallengerBot/1.0 "
        "(mailto:mael@example.com) requests/2.31.0"
    )

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.base_url = f"https://{lang}.wikipedia.org/w/api.php"
        self._local = threading.local()

    # ── Per-thread session ──────────────────────────────────────────
    def _get_session(self) -> requests.Session:
        """Return a ``requests.Session`` bound to the current thread."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": self.USER_AGENT})
            self._local.session = session
        return session

    # ── Public API ──────────────────────────────────────────────────
    def get_links(self, title: str) -> List[str]:
        """
        Fetches all internal links from a given Wikipedia page title.
        Handles pagination to get all links.  Thread-safe.
        """
        session = self._get_session()
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
                response = session.get(self.base_url, params=params)
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
                print(f"Error fetching links for {title}: {e}")
                break

        return links

    def validate_page(self, title: str) -> Optional[str]:
        """
        Validates if a page exists and returns its normalized title.
        """
        session = self._get_session()
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "redirects": 1,
        }
        try:
            response = session.get(self.base_url, params=params)
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


if __name__ == "__main__":
    # Quick test
    api = WikipediaAPI()
    test_title = "France"
    print(f"Fetching links for: {test_title}")
    links = api.get_links(test_title)
    print(f"Found {len(links)} links. First 5: {links[:5]}")

