import pytest
from main import text_to_chunks, TfidfSearch

# tests for chunking
def test_text_to_chunks_normal_case():
    """check that text is split into chunks of the right size."""
    assert text_to_chunks("word1 word2 word3 word4 word5", 2) == ["word1 word2", "word3 word4", "word5"]

def test_text_to_chunks_with_empty_text():
    """check that empty text or spaces return an empty list. """
    assert text_to_chunks("") == []
    assert text_to_chunks("   \n  ")== []

def test_text_to_chunks_with_invalid_size():
    """check that chunk_size <= 0 raises a ValueError."""
    with pytest.raises(ValueError):
        text_to_chunks("test", 0)
    with pytest.raises(ValueError):
        text_to_chunks("test", -5)


# tests for tf-idf search engine
class TestTfidfSearch:
    def test_init_with_empty_chunks_raises_error(self):
        """check that an empty list of chunks raises a ValueError.""" 
        with pytest.raises(ValueError):
            TfidfSearch([])

    def test_init_with_empty_vocabulary_raises_error(self):
        """check that chunks with only punctuation raise a ValueError."""
        with pytest.raises(ValueError):
            TfidfSearch(["???", "!!!"])
        with pytest.raises(ValueError):
            TfidfSearch(["a b", "! c"])

    def test_search_returns_most_relevant_chunk(self):
        """check that the best matching chunk is returned first."""
        chunks = [
            "the cat eats food",
            "je suis partie en vacance à paris",
            "the weather is sunny today"
        ]
        results = TfidfSearch(chunks).search("paris vacance", 1)

        assert len(results) == 1
        document, score = results[0]
        assert document == chunks[1]
        assert score> 0

    def test_search_with_unknown_words(self):
        """check result when query words are not in the document."""
        chunks = ["the cat eats food", "je suis partie en vacance à paris"]
        results = TfidfSearch(chunks).search("travaille", 2)

        assert results == [("le vecteur de la question est nul", 0)]

    def test_search_with_invalid_top_k_raises_error(self):
        """check that top_k <= 0 raises a ValueError."""
        with pytest.raises(ValueError):
            TfidfSearch( ["first document", "second document"]).search("document", 0)

    def test_search_with_top_k_larger_than_chunks(self):
        """check that top_k is limited to the total number of chunks."""
        results = TfidfSearch(["first document", "second document"]).search("document", 10)
        assert len(results) == 2

