from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    Language,
)

from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

Sample_text =  """




Introduction to Machine Learning

For decades, computers operated strictly on explicit, human-written instructions. If a developer wanted a program to solve a problem, they had to anticipate every possible scenario and write rigid rules to handle it. Machine learning (ML), a core branch of artificial intelligence, fundamentally changes this paradigm. Instead of programming computers with step-by-step commands, machine learning enables systems to learn patterns directly from data and improve their performance over time without being explicitly programmed.
At its heart, the machine learning process revolves around training algorithms on historical information. During this training phase, an algorithm analyzes large volumes of data—such as images, text, or numerical records—to detect underlying relationships and build a mathematical model. Once trained, this model can take in new, unseen data and make accurate predictions or decisions, whether that means flagging a fraudulent credit card transaction or predicting tomorrow's weather.
Machine learning approaches generally fall into three primary categories depending on how the system learns. Supervised learning uses labeled examples, much like a student studying with an answer key, to predict known outcomes. Unsupervised learning explores raw, unlabeled data to uncover hidden groupings or anomalies on its own. Finally, reinforcement learning trains an agent through trial and error, rewarding it for making good decisions in a dynamic environment, which is the driving force behind robotics and game-playing AI.
Today, machine learning quietly powers much of the modern digital world. From personalized streaming recommendations and voice assistants to advanced medical diagnostics and autonomous vehicles, its ability to turn massive datasets into actionable intelligence has made it indispensable. As data continues to grow exponentially, understanding the fundamentals of machine learning is becoming essential across nearly every industry.



def binary_search(arr, target):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2

        if arr[mid] == target:
            return mid

        elif arr[mid] < target:
            low = mid + 1

        else:
            high = mid - 1

    return -1


arr = [10, 20, 30, 40, 50, 60, 70]
target = 50

result = binary_search(arr, target)

if result != -1:
    print(f"Element found at index {result}")
else:
    print("Element not found")"""


def recursive_splitter():

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        separators = ['\n\n', '\n', ' ', '']
    )

    # Split the sample text into chunks
    CHUNKS = splitter.split_text(Sample_text) 
    print(f"original lenth of text: {len(Sample_text)}")
    print(f"Number of chunks created: {len(CHUNKS)}")
    print(f"No. of characters in each chunk: {[len(chunk) for chunk in CHUNKS]}")
    print(f"\nfirst chunk preview: {CHUNKS[0]}...")  # Print the first 100 characters of the first chunk
    #print all chunks
    for i, chunk in enumerate(CHUNKS):
        print(f"\nChunk {i+1}:\n{chunk}\n{'-'*40}") #this line prints a separator line of 40 dashes after each chunk for better readability


#recursive_splitter() #to run the function and see the output

def overlap_importance():
    text = "there i a fox and a lazy dog. the fox is very clever and the dog is very lazy. the fox and the dog are friends."
    chunks_no_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0).split_text(text)

    print("with overlap:")
    chunks_with_overlap = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20).split_text(text)

    print(f"Chunk 1 end ...{chunks_no_overlap[0][-20:]}")
    print(f"Chunk 2 start ...{chunks_no_overlap[1][:20]}")

    print("with overlap:")
    print(f"Chunk 1 end ...{chunks_with_overlap[0][-20:]}")
    print(f"Chunk 2 start ...{chunks_with_overlap[1][:20]}")



overlap_importance() #to run the function and see the output