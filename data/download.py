"""Download the tiny Shakespeare dataset (~1 MB) into data/training.txt"""
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(URL, "data/training.txt")
print("saved data/training.txt")
