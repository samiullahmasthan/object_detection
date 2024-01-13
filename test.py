import unittest
import json
from yolo import detect_objects, main  # Replace 'your_script' with the actual filename
import os
class TestObjectDetection(unittest.TestCase):
    def setUp(self):
        self.images_folder = 'images'  # Change this to your images folder path
        self.output_json_path = 'output_test.json'

    def tearDown(self):
        # Clean up after the test if needed
        pass

    def test_object_detection(self):
        print("Current working directory:", os.getcwd())
        # Ensure that the script runs without errors
        main()

        # Check if the output JSON file is created
        self.assertTrue(os.path.exists(self.output_json_path))

        # Load the results from the JSON file
        with open(self.output_json_path, 'r') as json_file:
            results = json.load(json_file)

        # Ensure that the results have the expected format
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(result, dict) for result in results))
        self.assertTrue(all("frame_number" in result and "objects" in result and "no_of_objects" in result and "image_name" in result for result in results))

        # You can add more specific assertions based on your requirements

if __name__ == '__main__':
    unittest.main()
