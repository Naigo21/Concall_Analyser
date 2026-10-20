from nltk.tokenize import PunktSentenceTokenizer

def chunk_text(text, max_words=400, overlap_words=50):
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        words = sentence.split()
        sentence_length = len(words)

        if current_length + sentence_length > max_words:
            chunks.append(" ".join(current_chunk))

            overlap = current_chunk[-overlap_words:] if overlap_words < len(current_chunk) else current_chunk
            current_chunk = overlap.copy()
            current_length = len(current_chunk)

        current_chunk.extend(words)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


