import nltk

def setup_nltk_punkt():
    """
    Downloads the NLTK 'punkt' tokenizer models if not already present.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt', quiet=True)
        print("Download complete.")

def segment_text_into_chunks(text: str, max_chars_per_chunk: int = 600, max_words_per_chunk: int = 100) -> list[str]:
    """
    Segments a given text into chunks suitable for TTS generation.

    The function prioritizes splitting at sentence boundaries. It accumulates
    sentences into a chunk until adding the next sentence would exceed
    max_chars_per_chunk or max_words_per_chunk.

    Args:
        text (str): The input text to segment.
        max_chars_per_chunk (int): The maximum number of characters allowed in a single chunk.
        max_words_per_chunk (int): The maximum number of words allowed in a single chunk.

    Returns:
        list[str]: A list of text chunks.
    """
    setup_nltk_punkt() # Ensure 'punkt' is available

    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk_text = ""
    current_chunk_char_count = 0
    current_chunk_word_count = 0

    for sentence in sentences:
        sentence_char_count = len(sentence)
        sentence_word_count = len(sentence.split())

        # Handle a single sentence that itself is too long
        if not current_chunk_text and \
           (sentence_char_count > max_chars_per_chunk or sentence_word_count > max_words_per_chunk):
            # If current_chunk_text was not empty, it would have been added in the previous iteration's 'else'
            # This condition means: new chunk, and this sentence alone is too big for a normal chunk.
            # Add it as its own chunk as per requirements.
            chunks.append(sentence)
            # No need to update current_chunk variables as this one is immediately finalized.
            # Reset them for the next potential chunk.
            current_chunk_text = ""
            current_chunk_char_count = 0
            current_chunk_word_count = 0
            continue

        # Check if adding the current sentence would exceed limits
        # Consider space separator if current_chunk_text is not empty
        potential_char_add = sentence_char_count + (1 if current_chunk_text else 0)
        potential_word_add = sentence_word_count 

        if not current_chunk_text or \
           (current_chunk_char_count + potential_char_add <= max_chars_per_chunk and
            current_chunk_word_count + potential_word_add <= max_words_per_chunk):
            
            if current_chunk_text:
                current_chunk_text += " " + sentence
                current_chunk_char_count += 1 # For the space
            else:
                current_chunk_text = sentence
            
            current_chunk_char_count += sentence_char_count
            current_chunk_word_count += sentence_word_count
        else:
            # Current chunk is full, or adding the sentence makes it full
            if current_chunk_text: # Ensure there's something to add
                chunks.append(current_chunk_text)
            
            # Start new chunk with the current sentence
            current_chunk_text = sentence
            current_chunk_char_count = sentence_char_count
            current_chunk_word_count = sentence_word_count
            
            # Special case: if this new sentence *itself* makes the new chunk too big,
            # it will be added as its own chunk in the *next* iteration or at the end.
            # This is implicitly handled because if it's too big, the next sentence
            # will trigger the 'else' branch immediately, adding this oversized sentence.

    # Add the last remaining chunk if it's not empty
    if current_chunk_text:
        chunks.append(current_chunk_text)
        
    return chunks

if __name__ == '__main__':
    # Conceptual Testing
    sample_text_1 = "This is the first sentence. This is the second sentence, which is a bit longer. What about a third, much longer sentence that might push the limits of a chunk all by itself, potentially exceeding the character or word count by a fair margin if we are not careful? A fourth sentence. And a fifth."
    sample_text_2 = "This is a very very very very very very very very very very very very very very very very very very very very long single sentence that definitely exceeds the typical word and character limits for a single chunk."
    sample_text_3 = "Short. Another. Then this one is a bit longer, let's see. And this one too. Maybe one more?"
    sample_text_4 = "First sentence. Second sentence that is very long and will exceed the word limit of twenty words all by itself if it is processed as a new chunk. Third sentence."

    print("--- Test Case 1 (Mixed Lengths) ---")
    chunks_1 = segment_text_into_chunks(sample_text_1, max_words_per_chunk=20, max_chars_per_chunk=200)
    for i, chunk in enumerate(chunks_1):
        print(f"Chunk {i+1} (Words: {len(chunk.split())}, Chars: {len(chunk)}):\n\"{chunk}\"\n")
    # Expected (max_words_per_chunk=20):
    # Chunk 1 (Words: 15, Chars: 80): "This is the first sentence. This is the second sentence, which is a bit longer."
    # Chunk 2 (Words: 33, Chars: 180): "What about a third, much longer sentence that might push the limits of a chunk all by itself, potentially exceeding the character or word count by a fair margin if we are not careful?"
    # Chunk 3 (Words: 6, Chars: 32): "A fourth sentence. And a fifth."
    
    print("--- Test Case 2 (Single Very Long Sentence) ---")
    chunks_2 = segment_text_into_chunks(sample_text_2, max_words_per_chunk=20, max_chars_per_chunk=100)
    for i, chunk in enumerate(chunks_2):
        print(f"Chunk {i+1} (Words: {len(chunk.split())}, Chars: {len(chunk)}):\n\"{chunk}\"\n")
    # Expected:
    # Chunk 1 (Words: 27, Chars: 154): "This is a very very very very very very very very very very very very very very very very very very very very long single sentence that definitely exceeds the typical word and character limits for a single chunk."

    print("--- Test Case 3 (Multiple Short Sentences) ---")
    chunks_3 = segment_text_into_chunks(sample_text_3, max_words_per_chunk=10, max_chars_per_chunk=50)
    for i, chunk in enumerate(chunks_3):
        print(f"Chunk {i+1} (Words: {len(chunk.split())}, Chars: {len(chunk)}):\n\"{chunk}\"\n")
    # Expected (max_words_per_chunk=10, max_chars_per_chunk=50)
    # Chunk 1 (Words: 2, Chars: 12): "Short. Another." (Actually 3 words: "Short.", "Another.") -> "Short. Another." (Words: 2, Chars: 15)
    # Chunk 2 (Words: 8, Chars: 40): "Then this one is a bit longer, let's see."
    # Chunk 3 (Words: 5, Chars: 22): "And this one too."
    # Chunk 4 (Words: 3, Chars: 15): "Maybe one more?"
    # My manual trace:
    # 1. "Short." (1w, 6c) -> current_chunk = "Short." (1w,6c)
    # 2. "Another." (1w, 8c). Potential: (1+1+1=3w, 6+1+8=15c). OK. current_chunk = "Short. Another." (2w, 15c) (Note: NLTK counts "Short." as 1 sentence. Word count of "Short." is 1. "Another." is 1. So "Short. Another." is 2 words by naive split, or 2 by realistic sentence parsing. NLTK sentences: ["Short.", "Another."]. Sentence 1: "Short." (1w,6c). Sentence 2: "Another." (1w,8c).
    #    - current_chunk = "Short." (1w,6c)
    #    - next sentence "Another." (1w,8c). Add? 1+1+1=3 words, 6+1+8=15 chars. OK.
    #    - current_chunk = "Short. Another." (1+1=2w, 6+1+8=15c)
    # 3. "Then this one is a bit longer, let's see." (8w, 40c). Potential: (2+1+8=11w > 10). NO.
    #    - Add "Short. Another." to chunks. Chunks: ["Short. Another."]
    #    - current_chunk = "Then this one is a bit longer, let's see." (8w, 40c)
    # 4. "And this one too." (4w, 18c). Potential: (8+1+4=13w > 10). NO.
    #    - Add "Then this one is a bit longer, let's see." to chunks. Chunks: [..., "Then...see."]
    #    - current_chunk = "And this one too." (4w, 18c)
    # 5. "Maybe one more?" (3w, 15c). Potential: (4+1+3=8w <=10) AND (18+1+15=34c <=50). OK.
    #    - current_chunk = "And this one too. Maybe one more?" (4+3=7w, 18+1+15=34c)
    # End loop. Add last chunk.
    # Chunks: ["Short. Another.", "Then this one is a bit longer, let's see.", "And this one too. Maybe one more?"]

    print("--- Test Case 4 (Second sentence is too long) ---")
    chunks_4 = segment_text_into_chunks(sample_text_4, max_words_per_chunk=20, max_chars_per_chunk=200)
    for i, chunk in enumerate(chunks_4):
        print(f"Chunk {i+1} (Words: {len(chunk.split())}, Chars: {len(chunk)}):\n\"{chunk}\"\n")
    # Expected:
    # Chunk 1: "First sentence."
    # Chunk 2: "Second sentence that is very long and will exceed the word limit of twenty words all by itself if it is processed as a new chunk."
    # Chunk 3: "Third sentence."

    print("--- Test Case 5 (Strict char limit forcing splits) ---")
    sample_text_5 = "This is short. This is also short. This is the third one, also short."
    chunks_5 = segment_text_into_chunks(sample_text_5, max_chars_per_chunk=20, max_words_per_chunk=100) # Very strict char limit
    for i, chunk in enumerate(chunks_5):
        print(f"Chunk {i+1} (Words: {len(chunk.split())}, Chars: {len(chunk)}):\n\"{chunk}\"\n")
    # Expected:
    # Chunk 1: "This is short." (14c)
    # Chunk 2: "This is also short." (19c)
    # Chunk 3: "This is the third one, also short." (36c) -> this will be its own chunk as it exceeds 20c when starting new.

    # Corrected Test Case 3 Trace and Logic:
    # Sentences from NLTK: ["Short.", "Another.", "Then this one is a bit longer, let's see.", "And this one too.", "Maybe one more?"]
    # max_words=10, max_chars=50

    # 1. sentence = "Short." (1w, 6c)
    #    current_chunk_text = "Short." (1w, 6c)

    # 2. sentence = "Another." (1w, 8c)
    #    Potential: current_chunk_char_count (6) + 1 (space) + sentence_char_count (8) = 15 <= 50
    #    Potential: current_chunk_word_count (1) + sentence_word_count (1) = 2 <= 10
    #    OK.
    #    current_chunk_text = "Short. Another." (1+1=2w, 6+1+8=15c)

    # 3. sentence = "Then this one is a bit longer, let's see." (8w, 40c)
    #    Potential: current_chunk_char_count (15) + 1 (space) + sentence_char_count (40) = 56 > 50. NO.
    #    Add current_chunk_text ("Short. Another.") to chunks. chunks = ["Short. Another."]
    #    current_chunk_text = "Then this one is a bit longer, let's see." (8w, 40c)

    # 4. sentence = "And this one too." (4w, 18c)
    #    Potential: current_chunk_char_count (40) + 1 (space) + sentence_char_count (18) = 59 > 50. NO.
    #    (Even if char was okay, words: 8w + 4w = 12w > 10. So still NO)
    #    Add current_chunk_text ("Then this one is a bit longer, let's see.") to chunks. chunks = ["Short. Another.", "Then this one is a bit longer, let's see."]
    #    current_chunk_text = "And this one too." (4w, 18c)

    # 5. sentence = "Maybe one more?" (3w, 15c)
    #    Potential: current_chunk_char_count (18) + 1 (space) + sentence_char_count (15) = 34 <= 50
    #    Potential: current_chunk_word_count (4) + sentence_word_count (3) = 7 <= 10
    #    OK.
    #    current_chunk_text = "And this one too. Maybe one more?" (4+3=7w, 18+1+15=34c)

    # End of loop. Add last current_chunk_text to chunks.
    # chunks.append("And this one too. Maybe one more?")
    # Final Chunks for Test Case 3:
    # ["Short. Another.", "Then this one is a bit longer, let's see.", "And this one too. Maybe one more?"]
    # This matches my manual trace. The previous "Expected" was slightly off.

    # Corrected Test Case 1 Trace and Logic:
    # Sentences: ["This is the first sentence.", "This is the second sentence, which is a bit longer.", "What about a third, much longer sentence that might push the limits of a chunk all by itself, potentially exceeding the character or word count by a fair margin if we are not careful?", "A fourth sentence.", "And a fifth."]
    # max_words=20, max_chars=200

    # 1. sentence = "This is the first sentence." (5w, 28c)
    #    current_chunk_text = "This is the first sentence." (5w, 28c)

    # 2. sentence = "This is the second sentence, which is a bit longer." (10w, 51c)
    #    Potential: char (28) + 1 + char (51) = 80 <= 200
    #    Potential: word (5) + word (10) = 15 <= 20
    #    OK.
    #    current_chunk_text = "This is the first sentence. This is the second sentence, which is a bit longer." (5+10=15w, 28+1+51=80c)

    # 3. sentence = "What about a third, much longer sentence that might push the limits of a chunk all by itself, potentially exceeding the character or word count by a fair margin if we are not careful?" (33w, 180c)
    #    Potential: char (80) + 1 + char (180) = 261 > 200. NO.
    #    (Also, words: 15 + 33 = 48 > 20. NO.)
    #    Add current_chunk_text to chunks. chunks = ["This is the first sentence. This is the second sentence, which is a bit longer."]
    #    current_chunk_text = "What about a third, much longer sentence that might push the limits of a chunk all by itself, potentially exceeding the character or word count by a fair margin if we are not careful?" (33w, 180c)
    #    Check if this new chunk itself is too long: 33w > 20w OR 180c <= 200. Word limit exceeded.
    #    The current logic for *starting* a new chunk is:
    #    `current_chunk_text = sentence`
    #    `current_chunk_char_count = sentence_char_count`
    #    `current_chunk_word_count = sentence_word_count`
    #    This means an oversized sentence *will* become current_chunk_text.
    #    If the *next* sentence cannot be added, this oversized chunk gets added. This is correct.
    #    The special handling for a single sentence that is too long at the *beginning* of the loop for an *empty* current_chunk_text:
    #       `if not current_chunk_text and (sentence_char_count > max_chars_per_chunk or sentence_word_count > max_words_per_chunk):`
    #       This correctly adds it immediately.
    #    So the existing logic should handle the long sentence correctly.

    # 4. sentence = "A fourth sentence." (3w, 18c)
    #    current_chunk_text (from step 3) is the very long sentence (33w, 180c).
    #    Potential: char (180) + 1 + char (18) = 199 <= 200
    #    Potential: word (33) + word (3) = 36 > 20. NO.
    #    Add current_chunk_text (the long one) to chunks.
    #    chunks = ["This is the first sentence. This is the second sentence, which is a bit longer.", "What about a third, much longer sentence...careful?"]
    #    current_chunk_text = "A fourth sentence." (3w, 18c)

    # 5. sentence = "And a fifth." (3w, 13c)
    #    Potential: char (18) + 1 + char (13) = 32 <= 200
    #    Potential: word (3) + word (3) = 6 <= 20
    #    OK.
    #    current_chunk_text = "A fourth sentence. And a fifth." (3+3=6w, 18+1+13=32c)

    # End of loop. Add last current_chunk_text.
    # chunks.append("A fourth sentence. And a fifth.")
    # Final Chunks for Test Case 1:
    # ["This is the first sentence. This is the second sentence, which is a bit longer.", "What about a third, much longer sentence that might push the limits of a chunk all by itself, potentially exceeding the character or word count by a fair margin if we are not careful?", "A fourth sentence. And a fifth."]
    # This matches the expected output.

    # Test Case 5: "This is short. This is also short. This is the third one, also short."
    # max_chars_per_chunk=20, max_words_per_chunk=100
    # Sentences: ["This is short.", "This is also short.", "This is the third one, also short."]
    # 1. sentence = "This is short." (3w, 14c)
    #    current_chunk_text = "This is short." (3w, 14c)
    # 2. sentence = "This is also short." (4w, 19c)
    #    Potential: char (14) + 1 + char (19) = 34 > 20. NO.
    #    Add "This is short." to chunks. chunks = ["This is short."]
    #    current_chunk_text = "This is also short." (4w, 19c)
    #    Check if this new chunk itself is too long: 19c <= 20. OK.
    # 3. sentence = "This is the third one, also short." (7w, 36c)
    #    current_chunk_text (from step 2) is "This is also short." (4w, 19c)
    #    Potential: char (19) + 1 + char (36) = 56 > 20. NO.
    #    Add "This is also short." to chunks. chunks = ["This is short.", "This is also short."]
    #    current_chunk_text = "This is the third one, also short." (7w, 36c)
    #    Check if this new chunk itself is too long: 36c > 20. Yes.
    #    The logic: `if not current_chunk_text and (sentence_char_count > max_chars_per_chunk ...)`
    #    This will be handled by the loop end.
    # End of loop. Add last current_chunk_text.
    # chunks.append("This is the third one, also short.")
    # Final Chunks for Test Case 5:
    # ["This is short.", "This is also short.", "This is the third one, also short."] - This is correct.
    # The special handling for single oversized sentences works when it's the *first* thing considered for a chunk.
    # If an oversized sentence becomes `current_chunk_text` because the *previous* chunk was finalized,
    # and then the loop ends, it's added. If there's another sentence after it, the check
    # `current_chunk_char_count + potential_char_add <= max_chars_per_chunk` will fail,
    # leading to the oversized `current_chunk_text` being added. This seems correct.

    print("--- Test Case 6 (Empty Text) ---")
    chunks_6 = segment_text_into_chunks("", max_words_per_chunk=20, max_chars_per_chunk=100)
    print(f"Chunks: {chunks_6}, Count: {len(chunks_6)}")
    # Expected: [], Count: 0

    print("--- Test Case 7 (Text shorter than limits) ---")
    chunks_7 = segment_text_into_chunks("One sentence. Two sentences.", max_words_per_chunk=20, max_chars_per_chunk=100)
    for i, chunk in enumerate(chunks_7):
        print(f"Chunk {i+1} (Words: {len(chunk.split())}, Chars: {len(chunk)}):\n\"{chunk}\"\n")
    # Expected: ["One sentence. Two sentences."]
Okay, I've developed the `segment_text_into_chunks` function along with the NLTK setup. I've also included a basic `if __name__ == '__main__':` block with several test cases and traced their expected outputs based on the logic.

The core logic prioritizes sentence boundaries. If a sentence can be added to the current chunk without exceeding character or word limits, it's appended. Otherwise, the current chunk is finalized, and the new sentence starts a new chunk. A single sentence that itself exceeds the limits will form its own chunk. This is handled by the condition where a sentence is too large to be added to an existing non-empty chunk, causing the existing chunk to be stored and the large sentence to become the new `current_chunk_text`. If this large sentence is the last one, or if the next sentence also cannot be appended to it, this large sentence chunk will be stored.

The refined logic for handling sentences that are themselves too long (even to start a new chunk) is as follows:
If `current_chunk_text` is empty, and the current `sentence` *alone* exceeds `max_chars_per_chunk` or `max_words_per_chunk`, it's immediately added as its own chunk, and processing continues to the next sentence with a reset `current_chunk_text`. This ensures such sentences don't prevent subsequent sentences from forming valid chunks.

Here is the code:
