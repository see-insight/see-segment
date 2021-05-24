"""This runs unit tests for functions that can be found in JupyterGUI.py."""
import numpy as np
from see import JupyterGUI


def test_showtwo():
    """Unit test for showtwo function. Tests for figure shape."""
    img1 = np.ones((20, 20, 3))
    img2 = np.ones((20, 20))
    assert len(JupyterGUI.showtwo(img1, img2).axes) == 2


def test_showthree():
    """Unit test for showthree function. Tests for figure shape."""
    img1 = np.ones((20, 20, 3))
    img2 = np.ones((20, 20))
    img3 = np.zeros((20, 20))
    assert len(JupyterGUI.showthree(img1, img2, img3).axes) == 3


def test_show_segment():
    """Unit test for show_segment function. Tests for figure shape."""
    img1 = np.ones((20, 20, 3))
    img2 = np.ones((20, 20))
    assert len(JupyterGUI.show_segment(img1, img2).axes) == 2
