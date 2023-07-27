import unittest
import src.preprocessing as process
import numpy as np

class Processing_Test(unittest.TestCase):

    def test_extract_contours(self):
        img= np.array([[0,0,0,0,0,0],
              [0,1,1,1,1,0],
              [0,1,0,0,1,0],
              [0,1,0,0,1,0],
              [0,1,1,1,1,0],
              [0,0,0,0,0,0]])
        cnts= process.extract_contours(img)
        res= (cnts- np.array([[[1, 1],[1, 2],[1, 3],[1, 4],[2, 4],[3, 4],[4, 4],[4, 3],[4, 2],[4, 1],[3, 1],[2, 1],[1, 1]]])).all()
        self.assertTrue(res, "extract_contours failed")


if __name__ == '__main__':
    unittest.main()