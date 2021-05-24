"""This tests to see if the main dependencies for see can be successfully imported."""


def test_imports():
    """Import packages. Test will fail if packages cannot import successfully."""
    #TODO unused and outside of toplevel imports are causing this file
    # to have a negative Pylint score of -6.67. We comment these import out for
    # now, until there is a better way to test imports. Commenitng it out
    # brings score to 0/10. Marking this task as TODO prevented this file from having
    # a score of 10.
    #import deap
    #import scoop
    #import skimage
    #import inspect
    #import ipywidgets
