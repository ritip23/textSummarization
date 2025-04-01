import unittest
from app import app, spacy_summarizer, luhn_summarizer

class FlaskTestCase(unittest.TestCase):

    # Test Home Page Loads
    def test_home(self):
        tester = app.test_client(self)
        response = tester.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Text Summarizer', response.data)

    # Test POST Request for SpaCy Summarizer
    def test_summarize_spacy(self):
        tester = app.test_client(self)
        response = tester.post('/summarize', data=dict(
            text="The Marvel Cinematic Universe is a popular superhero franchise with many movies and shows.",
            algorithm="spacy"
        ))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'SpaCy Extractive Summarizer', response.data)

    # Test Luhn Summarizer Logic
    def test_luhn_summarizer(self):
        text = ("The Marvel Cinematic Universe is a series of superhero movies. "
                "It is the highest grossing franchise in the world. "
                "It includes Iron Man, Thor, Captain America and more.")
        result = luhn_summarizer(text, sentences_count=1)
        self.assertTrue(len(result) > 0)

    # Test SpaCy Summarizer Logic
    def test_spacy_summarizer_function(self):
        text = ("The Marvel Cinematic Universe is a series of superhero movies. "
                "It is the highest grossing franchise in the world. "
                "It includes Iron Man, Thor, Captain America and more.")
        result = spacy_summarizer(text)
        self.assertTrue(len(result) > 0)

if __name__ == '__main__':
    unittest.main()
