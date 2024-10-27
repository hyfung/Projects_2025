# String Matching

## Principle
- Calculates similarity between two strings

## Examples
|Method|Advantages|
|-|-|
|Jaccard Similarity| Good for comparing sets of words or characters|
|Cosine Similarity| Excellent for comparing documents or longer texts|
|Hamming Distance| Works well with strings of equal length|
|Jaro-Winkler Distance| Suitable for matching names or strings with small errors|
|Damerau-Levenshtein Distance| Best for correcting typographical errors|

### Levenshtein Distance | Fuzzy Matching

#### Definition
- Levenshtein distance calculation
- Number of insertion, deletions or substitution required to transform one string to another
- Ratio = (1 - LD/max(lenA, lenB) ) x 100
- Partial Matches Handling: Distance between shorter string and best matching substring
- Token Sort Ratio: Sorts the word alphabetically before calculating LR
- Token Set Ratio: Finds common between two strings, compute LR, gives weighted score

#### Sample Code
```python
from fuzzywuzzy import process

def find_closest_match(ground_truth, input_string):
    # Find the closest match in the ground truth list to the input string
    closest_match = process.extractOne(input_string, ground_truth)
    return closest_match

# Example usage:
ground_truth = ["apple", "pineapple", "banana", "grape", "orange"]
input_string = "appl"
result = find_closest_match(ground_truth, input_string)
print(f"Closest match: {result[0]} with a similarity score of {result[1]}")
```

### Jaccard Similarity

#### Definition
- `size_of_intersection` / `size_of_union`
- Strings are tokenized

#### Sample Code
```python
def jaccard_similarity(str1, str2):
    set1, set2 = set(str1), set(str2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

# Usage:
string1 = "apple"
string2 = "appl"
similarity = jaccard_similarity(string1, string2)
print(f"Jaccard Similarity: {similarity}")
```

### Cosine Similarity

#### Definition
- Cosine angle between two vetcors in N-d space
- String is represented as vector by Term Frequency-Inverse Document Frequency (TF-IDF)

#### Sample Code
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(input_string, ground_truth):
    # Add input_string to ground_truth for vectorization
    strings = ground_truth + [input_string]
    
    # Vectorize the strings using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform(strings)
    
    # Calculate cosine similarity with the last string (input_string)
    cosine_similarities = cosine_similarity(vectorizer[-1], vectorizer).flatten()
    
    # Find the index of the maximum similarity (excluding self-comparison)
    best_match_idx = cosine_similarities[:-1].argmax()
    return ground_truth[best_match_idx], cosine_similarities[best_match_idx]

# Usage:
ground_truth = ["apple", "pineapple", "banana", "grape", "orange"]
input_string = "appl"
result, similarity_score = cosine_sim(input_string, ground_truth)
print(f"Closest match: {result} with a similarity score of {similarity_score}")
```

### Hamming Distance

#### Definition
- Number of position at which two strings of equal length differs

#### Sample Code
```python
def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("Strings must be of equal length")
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

# Usage:
string1 = "apple"
string2 = "apply"
distance = hamming_distance(string1, string2)
print(f"Hamming Distance: {distance}")
```

### Jaro-Winkler Distance

#### Definition
- Extension of Jaro distance metric
- Similarity between two strings based on common characters and transpositions
- Higher scores to string with common prefix
- Good for comparing similar names or words

#### Sample Code
```python
from jellyfish import jaro_winkler_similarity

def jaro_winkler(str1, str2):
    return jaro_winkler_similarity(str1, str2)

# Usage:
similarity = jaro_winkler("apple", "appl")
print(f"Jaro-Winkler Similarity: {similarity}")
```

### Damerau-Levenshtein Distance

#### Definition
- Variant of Levenshtein distance
- Accounts for transposition of adjacent characters too
- Suitable for typographical errors

#### Sample Code
```python
import textdistance

def damerau_levenshtein(str1, str2):
    return textdistance.damerau_levenshtein.normalized_similarity(str1, str2)

# Usage:
similarity = damerau_levenshtein("apple", "aplep")
print(f"Damerau-Levenshtein Similarity: {similarity}")
```
