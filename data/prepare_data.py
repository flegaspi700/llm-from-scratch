import requests

def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    with open("data/raw/shakespeare.txt", "w", encoding="utf-8") as f:
        f.write(response.text)

    print("Shakespeare dataset downloaded.")
    print(f"Dataset size: {len(response.text)} characters")

if __name__ == "__main__":
    download_shakespeare()