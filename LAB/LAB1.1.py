# Part 1: Implementing BPE from Scratch
# Step 1: Understanding the Data Structures
import re
from collections import defaultdict, Counter
def preprocess_text(text):
 """
 Lowercase and split text into words.
 """
 # Lowercase
 text = text.lower()
 # Split on whitespace and punctuation
 words = re.findall(r'\w+', text)
 return words

corpus = """
low low low lower lower lowest
the the the quick quick brown fox
"""
words = preprocess_text(corpus)
print("Words:", words)

#Count word frequencies
word_freqs = Counter(words)
print("Word frequencies:", word_freqs)

# Step 3: Initialize Character-Level Vocabulary
def get_vocab(word_freqs):
 """
 Split each word into characters + </w> marker.
 Returns dictionary: word -> list of tokens
 """
 vocab = {}
 for word, freq in word_freqs.items():
    # Split into characters, add </w> marker
    vocab[word] = list(word) + ['</w>']
 return vocab
vocab = get_vocab(word_freqs)
print("Initial vocabulary:")
for word, tokens in vocab.items():
 print(f" {word}: {tokens}")

# Step 4: Count Pairs
def get_stats(vocab, word_freqs):
 """
 Count frequency of adjacent token pairs.
 """
 pairs = defaultdict(int)
 for word, freq in word_freqs.items():
    symbols = vocab[word]
    for i in range(len(symbols) - 1):
        pair = (symbols[i], symbols[i+1])
        pairs[pair] += freq
 return pairs

pairs = get_stats(vocab, word_freqs)
print("Pair frequencies:")
for pair, freq in sorted(pairs.items(), key=lambda x: -x[1])[:10]:
 print(f" {pair}: {freq}")

# Step 5: Merge Most Frequent Pair
def merge_vocab(pair, vocab):
 """
 Merge the given pair in the vocabulary.
 """
 new_vocab = {}

 # Create pattern to find the pair
 # Example: ('l', 'o') -> 'l o'
 pattern = ' '.join(pair)
 replacement = ''.join(pair)
 for word, tokens in vocab.items():
    # Convert tokens list to string for replacement
    tokens_str = ' '.join(tokens)
    # Replace first occurrence of pair
    tokens_str = tokens_str.replace(pattern, replacement)
    # Convert back to list
    new_vocab[word] = tokens_str.split()
 return new_vocab

most_frequent_pair = max(pairs, key=pairs.get)
print(f"\nMerging: {most_frequent_pair} → {''.join(most_frequent_pair)}")

vocab = merge_vocab(most_frequent_pair, vocab)
print("Vocabulary after merge:")
for word, tokens in vocab.items():
 print(f" {word}: {tokens}")

#Step 6: Complete BPE Training Loop
def train_bpe(word_freqs, num_merges):
 """
 Train BPE tokenizer for num_merges iterations.
 Returns vocabulary and list of merge operations.
 """
 vocab = get_vocab(word_freqs)
 merges = []
 for i in range(num_merges):
    pairs = get_stats(vocab, word_freqs)
    if not pairs:
        break

 # Find most frequent pair
 best_pair = max(pairs, key=pairs.get)

 # Merge it
 vocab = merge_vocab(best_pair, vocab)
 merges.append(best_pair)
 print(f"Iteration {i+1}: Merged {best_pair} → {''.join(best_pair)}(freq={pairs[best_pair]})")
 return vocab, merges

vocab, merges = train_bpe(word_freqs, num_merges=10)

print("\nFinal vocabulary:")
for word, tokens in vocab.items():
 print(f" {word}: {tokens}")
 
print("\nMerge operations (in order):")
for i, merge in enumerate(merges, 1):
 print(f" {i}. {merge[0]} + {merge[1]} → {''.join(merge)}")