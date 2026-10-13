"""Fetch and parse public webpages. Respect website terms and robots policies."""
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from .common import Document, clean_text

def load_website(url: str, timeout: int = 15) -> list[Document]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a full http:// or https:// URL.")
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "EnterpriseRAGDemo/1.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = clean_text(soup.get_text(separator="\n"))
    if not text:
        return []
    title = soup.title.get_text(" ", strip=True) if soup.title else url
    return [Document(text, title, "website", {"url": url, "title": title})]
