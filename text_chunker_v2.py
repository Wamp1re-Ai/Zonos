import nltk

def setup_nltk_punkt():
    """
    Downloads the NLTK 'punkt' tokenizer models if not already present.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        # These print statements are for user feedback in an interactive environment like Colab.
        # For a pure library function, they might be omitted or handled by logging.
        # print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt', quiet=True)
        # print("NLTK 'punkt' download complete.")

def segment_text_into_chunks(text: str, max_chars_per_chunk: int = 600, max_words_per_chunk: int = 100) -> list[str]:
    """
    Segments a given text into chunks suitable for TTS generation.

    The function prioritizes splitting at sentence boundaries. It accumulates
    sentences into a chunk until adding the next sentence would exceed
    max_chars_per_chunk or max_words_per_chunk. A single sentence that
    itself exceeds these limits will form its own chunk.

    Args:
        text (str): The input text to segment.
        max_chars_per_chunk (int): The maximum number of characters allowed
                                   in a single chunk. Defaults to 600.
        max_words_per_chunk (int): The maximum number of words allowed
                                   in a single chunk. Defaults to 100.

    Returns:
        list[str]: A list of text chunks.
    """
    setup_nltk_punkt() 

    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []
    
    chunks = []
    current_chunk_text = ""
    current_chunk_char_count = 0
    current_chunk_word_count = 0

    for sentence in sentences:
        sentence_char_count = len(sentence)
        # Using a simple split for word count as per problem description.
        sentence_word_count = len(sentence.split()) 

        # Case 1: Current sentence ALONE is too large for a new chunk.
        # This applies if current_chunk_text is empty (we are about to start a new chunk).
        if not current_chunk_text and \
           (sentence_char_count > max_chars_per_chunk or sentence_word_count > max_words_per_chunk):
            chunks.append(sentence)
            # current_chunk_text remains empty, counts remain 0, ready for the next sentence.
            continue

        # Case 2: Adding current sentence to an existing non-empty chunk would make it too large.
        space_char_needed = 1 if current_chunk_text else 0 
        
        if current_chunk_text and \
           ((current_chunk_char_count + sentence_char_count + space_char_needed > max_chars_per_chunk) or \
            (current_chunk_word_count + sentence_word_count > max_words_per_chunk)):
            
            chunks.append(current_chunk_text)
            # Start new chunk with current sentence
            current_chunk_text = sentence
            current_chunk_char_count = sentence_char_count
            current_chunk_word_count = sentence_word_count
        
        # Case 3: Add sentence to current chunk (either starting a new one or appending).
        # This also handles the case where a new chunk is started with a sentence that
        # is large but still within the single-sentence-as-chunk allowance (it wouldn't have been caught by Case 1).
        else:
            if not current_chunk_text: # Starting a new chunk
                current_chunk_text = sentence
                current_chunk_char_count = sentence_char_count
                current_chunk_word_count = sentence_word_count
            else: # Appending to existing chunk
                current_chunk_text += " " + sentence
                current_chunk_char_count += (sentence_char_count + space_char_needed) 
                current_chunk_word_count += sentence_word_count
    
    # Add the last remaining chunk if it's not empty
    if current_chunk_text:
        chunks.append(current_chunk_text)
        
    return chunks

if __name__ == '__main__':
    setup_nltk_punkt() 
    print("NLTK 'punkt' should be available for __main__ tests.\n")

    test_cases_and_expected = [
        ("Test Case 1 (Prompt Example, max_words=20, max_chars=600)", 
         "This is the first sentence. This is the second sentence, which is a bit longer. What about a third, much longer sentence that might push the limits of a chunk all by itself, potentially exceeding the character or word count by a fair margin if we are not careful? A fourth sentence. And a fifth.", 
         600, 20,
         [
            "This is the first sentence. This is the second sentence, which is a bit longer.",
            "What about a third, much longer sentence that might push the limits of a chunk all by itself, potentially exceeding the character or word count by a fair margin if we are not careful?",
            "A fourth sentence. And a fifth."
         ]),
        ("Test Case 2 (Single Very Long Sentence, max_words=20, max_chars=100)", 
         "This is a very very very very very very very very very very very very very very very very very very very very long single sentence that definitely exceeds the typical word and character limits for a single chunk.", 
         100, 20,
         [
            "This is a very very very very very very very very very very very very very very very very very very very very long single sentence that definitely exceeds the typical word and character limits for a single chunk."
         ]),
        ("Test Case 3 (Multiple Short Sentences, max_words=10, max_chars=50)", 
         "Short. Another. Then this one is a bit longer, let's see. And this one too. Maybe one more?", 
         50, 10,
         [
            "Short. Another.",
            "Then this one is a bit longer, let's see.",
            "And this one too. Maybe one more?"
         ]),
        ("Test Case 4 (Second sentence itself is too long, max_words=20, max_chars=200)", 
         "First sentence. Second sentence that is very long and will exceed the word limit of twenty words all by itself if it is processed as a new chunk. Third sentence.", 
         200, 20,
         [
            "First sentence.",
            "Second sentence that is very long and will exceed the word limit of twenty words all by itself if it is processed as a new chunk.",
            "Third sentence."
         ]),
        ("Test Case 5 (Strict char limit forcing splits, max_chars=20, max_words=100)", 
         "This is short. This is also short. This is the third one, also short.", 
         20, 100,
         [
            "This is short.", "This is also short.", "This is the third one, also short."
         ]),
        ("Test Case 6 (Empty Text)", 
         "", 
         100, 20,
         []),
        ("Test Case 7 (Text shorter than limits)", 
         "One sentence. Two sentences.", 
         100, 20,
         ["One sentence. Two sentences."]),
        ("Test Case 8 (Long sentence in middle, max_chars=60, max_words=10)", 
         "This sentence is fine. This extremely long sentence, however, is definitely over the character limit of sixty and also over the word limit of ten words, so it must be on its own. The next sentence is short.", 
         60, 10,
         [
            "This sentence is fine.",
            "This extremely long sentence, however, is definitely over the character limit of sixty and also over the word limit of ten words, so it must be on its own.",
            "The next sentence is short."
         ]),
        ("Test Case 9 (Default limits, should be one chunk)",
         "This is a standard text. It has a few sentences. The default limits should handle this reasonably well, creating just one chunk.",
         600, 100,
         ["This is a standard text. It has a few sentences. The default limits should handle this reasonably well, creating just one chunk."]),
        ("Test Case 10 (Max chars hit before max words)",
         "This sentence is not too long in words. But it uses many characters, like abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz. Another sentence follows.",
         100, 30, 
         [
            "This sentence is not too long in words.",
            "But it uses many characters, like abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz abcdefghijklmnopqrstuvwxyz.",
            "Another sentence follows."
         ])
    ]

    all_tests_passed = True
    for (name, text, max_c, max_w, expected_chunk_list) in test_cases_and_expected:
        print(f"\n--- {name} ---")
        chunks = segment_text_into_chunks(text, max_chars_per_chunk=max_c, max_words_per_chunk=max_w)
        passed = chunks == expected_chunk_list
        if not passed:
            all_tests_passed = False
        print(f"Status: {'✅ Passed' if passed else '❌ Failed'}")
        # Always print generated chunks for review
        # for i, chunk in enumerate(chunks):
        #     print(f"Generated Chunk {i+1} (Words: {len(chunk.split())}, Chars: {len(chunk)}):\n\"{chunk}\"")
        if not passed:
            print("Expected Output:")
            for i, chunk in enumerate(expected_chunk_list):
                print(f"Expected Chunk {i+1} (Words: {len(chunk.split())}, Chars: {len(chunk)}):\n\"{chunk}\"")
            print("Actual Output:")
            for i, chunk in enumerate(chunks):
                 print(f"Actual Chunk {i+1} (Words: {len(chunk.split())}, Chars: {len(chunk)}):\n\"{chunk}\"")
        print("-" * 30)

    if all_tests_passed:
        print("\n✅✅✅ All defined test cases passed! ✅✅✅")
    else:
        print("\n❌❌❌ Some test cases failed. Please review output. ❌❌❌")

```
