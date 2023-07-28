import unittest
import src.input as input
import numpy as np

class Test_Contour_detection(unittest.TestCase):

    def test_input_file(self):
        res= input.get_from_file("random")
        self.assertEqual(res, None, "Input from file faulty")

    def test_input_camera(self):
        res= input.get_from_camera()
        self.assertFalse(res is None, "Input from camera faulty")


if __name__ == '__main__':
    unittest.main()