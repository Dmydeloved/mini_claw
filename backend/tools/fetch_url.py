"""
Fetch URL Tool for Mini-OpenClaw
Wraps RequestsGetTool with HTML cleaning to reduce token usage
"""

import html2text
from bs4 import BeautifulSoup
from langchain_community.tools import RequestsGetTool
from langchain_community.utilities.requests import TextRequestsWrapper
from pydantic import PrivateAttr


class CleanRequestsGetTool(RequestsGetTool):
    """
    Enhanced RequestsGetTool that returns clean markdown instead of raw HTML
    """

    _html_converter: html2text.HTML2Text = PrivateAttr()

    def __init__(self):
        super().__init__(
            name="fetch_url",
            description=(
                "Fetch content from a URL and return it as clean markdown text. "
                "Use this to retrieve web pages, API responses, or any online content."
            ),
            requests_wrapper=TextRequestsWrapper(),
            allow_dangerous_requests=True,
        )
        self._html_converter = html2text.HTML2Text()
        self._html_converter.ignore_links = False
        self._html_converter.ignore_images = True
        self._html_converter.ignore_emphasis = False

    def _run(self, url: str) -> str:
        """Fetch URL and return cleaned content"""
        try:
            # Get raw HTML using parent class
            raw_html = super()._run(url)

            # Check if it's already plain text (API response)
            if not raw_html.strip().startswith("<"):
                return raw_html

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(raw_html, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get text content
            html_content = str(soup)

            # Convert to markdown
            markdown_content = self._html_converter.handle(html_content)

            # Clean up excessive whitespace
            lines = [line.strip() for line in markdown_content.split("\n")]
            cleaned_lines = [line for line in lines if line]
            result = "\n\n".join(cleaned_lines)

            # Limit length to prevent token overflow
            max_chars = 10000
            if len(result) > max_chars:
                result = result[:max_chars] + f"\n\n[Content truncated. Total length: {len(result)} chars]"

            return result

        except Exception as e:
            return f"Error fetching URL: {str(e)}"
def create_fetch_url_tool() -> CleanRequestsGetTool:
    """
    Factory function to create fetch URL tool

    Returns:
        CleanRequestsGetTool instance
    """
    return CleanRequestsGetTool()
