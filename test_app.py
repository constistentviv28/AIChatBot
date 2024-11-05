import unittest
from app import extract_text_from_pdf, create_embeddings

class TestPDFChat(unittest.TestCase):

    def test_extract_text_from_pdf(self):
        text = extract_text_from_pdf("test.pdf")  # Create a sample PDF for testing
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_create_embeddings(self):
        text = "This is a test sentence."
        sentences, embeddings = create_embeddings(text)
        self.assertEqual(len(sentences), 1)
        self.assertEqual(embeddings.shape[0], 1)

if __name__ == "__main__":
    unittest.main()
