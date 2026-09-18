import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.text, "html.parser")

content = soup.find("main")

paragraphs = soup.find_all("p")

for p in paragraphs:
    text = p.get_text(strip=True)

    if text:
        print(text)

print(text)