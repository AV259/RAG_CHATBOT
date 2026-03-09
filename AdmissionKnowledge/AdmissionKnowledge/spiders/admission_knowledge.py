
import scrapy
from bs4 import BeautifulSoup
from scrapy.linkextractors import LinkExtractor
from scrapy import Request


def extract_title(soup):
    return soup.title.text.strip() if soup.title else None


def extract_main_content(soup):
    # Remove non-content elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # Try main first 
    main = soup.find("main")
    if main:
        return main.get_text(separator="\n", strip=True)

    # Try id="content"
    content = soup.find(attrs={"id": "content"})
    if content:
        return content.get_text(separator="\n", strip=True)

    # checking the body
    if soup.body:
        return soup.body.get_text(separator="\n", strip=True)

    return None


class AdmissionKnowledgeSpider(scrapy.Spider):
    name = "admission_knowledge"

    start_urls = [
        "https://www.uni-marburg.de/en/fb12/studying/degree-programs/m-sc-data-science",
        "https://www.uni-marburg.de/en/studying/after-your-first-degree/masters-programs/application-for-a-masters-programme/assist/mastersapplication-degree-abroad-first-semester",
        "https://www.uni-marburg.de/en/studying/after-your-first-degree/masters-programs/application-for-a-masters-programme/marvin/mastersapplication-german-degree-first-semester-marvin",
        "https://www.uni-marburg.de/en/studying/after-your-first-degree/masters-programs/application-for-a-masters-programme/eap/eap-data-science",
        "https://www.uni-marburg.de/en/studying/before-studying/get-started/finance",
        "https://www.uni-marburg.de/en/studying/during-studying/formalities-fees/information-about-contributions-and-fees",
        "https://www.uni-marburg.de/en/studying/before-studying/get-started/visa",
        "https://www.uni-marburg.de/en/studying/before-studying/studying-in-marburg/explore-marburg"
    ]

      #allowed_domains = ["uni-marburg.de"]

    # Exclude irrelevant pages
    deny_patterns = [
        r"forms\.uni-marburg\.de"
    ]

    custom_settings = {
        "DOWNLOAD_DELAY": 3,  
        'ROBOTSTXT_OBEY': True,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.link_extractor = LinkExtractor(
            allow=[
        # Faculty program pages
        r".*/m-sc-data-science.*",
        r".*/degree-programs/datasciencems.*",
        r"^https://www.uni-marburg.de/en/fb12/studying/modules-and-courses/types-of-courses",
        r"^https://www.uni-marburg.de/en/fb12/studying/modules-and-courses/glossary",
        r"^https://www.uni-marburg.de/en/studying/after-your-first-degree/masters-programs/application-for-a-masters-programme/master-application-deadlines",
            ],
            deny=self.deny_patterns,
        )

    def parse(self, response):
        soup = BeautifulSoup(response.text, "html.parser")

        title = extract_title(soup)
        content = extract_main_content(soup)

        if content:
            yield {
                "url": response.url,
                "title": title,
                "content": content,
            }

        for link in self.link_extractor.extract_links(response):
            yield Request(link.url, callback=self.parse)