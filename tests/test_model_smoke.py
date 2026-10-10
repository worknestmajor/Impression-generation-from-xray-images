import os
import unittest


@unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1","Set RUN_MODEL_TESTS=1 to run the large model smoke test")
class ModelSmokeTests(unittest.TestCase):
    def test_model_and_tokenizer_load(self):
        import create_model
        model,tokenizer = create_model.create_model()
        self.assertIn("<cls>",tokenizer.word_index)
        self.assertIn("<end>",tokenizer.word_index)
        self.assertEqual(tuple(model.inputs[0].shape[1:]),(224,224,3))
        self.assertEqual(tuple(model.output.shape[1:]),(29,len(tokenizer.word_index)+1))


if __name__ == "__main__":
    unittest.main()
