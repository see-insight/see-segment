from see import JupyterGUI
import pytest
import numpy as np

def test_showtwo():
    img1 = np.ones((20, 20, 3))
    img2 = np.ones((20, 20))
    assert len(JupyterGUI.showtwo(img1, img2).axes) == 2
    
def test_showthree():
    img1 = np.ones((20, 20, 3))
    img2 = np.ones((20, 20))
    img3 = np.zeros((20, 20))
    assert len(JupyterGUI.showthree(img1, img2, img3).axes) == 3
    
def test_showSegment():
    img1 = np.ones((20, 20, 3))
    img2 = np.ones((20, 20))
    assert len(JupyterGUI.showSegment(img1, img2).axes) == 2



