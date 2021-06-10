"""This runs unit tests for functions that can be found in Segmentors.py."""

def test_load_color_space_library():
    """Unit test for importing ColorSpace module."""
    from see import ColorSpace
    print(ColorSpace)


def test_loading_image_examples():
    """Unit test for loading images."""
    import imageio
    img = imageio.imread('Image_data/Examples/AA_Chameleon.jpg')
    gmask = imageio.imread('Image_data/Examples/AA_Chameleon_GT.png')
    return img, gmask


def test_make_colorspace():
    """Unit test for properly creating a ColorSpace instance."""
    from see import ColorSpace
    from see import base_classes
    cs = ColorSpace.colorspace()
    assert issubclass(type(cs.params), dict)
    assert issubclass(type(cs.params), base_classes.param_space)
    assert issubclass(type(cs), base_classes.algorithm)


def test_colorspace_evaluate():
    """Unit test for evaluate function."""
    from see import ColorSpace
    img, gmask = test_loading_image_examples()
    cs = ColorSpace.colorspace()
    test = cs.evaluate(img)


def test_colorspace_pipe():
    """Unit test for pipe function."""
    from see import ColorSpace
    from see import base_classes
    img, gmask = test_loading_image_examples()
    data = base_classes.pipedata()
    data.img = img
    data.gmask = gmask
    cs = ColorSpace.colorspace()
    data = cs.pipe(data)
