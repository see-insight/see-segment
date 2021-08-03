import numpy as np

from see.base_classes import algorithm
from sklearn.model_selection import cross_val_score


class ClassifierFitness(algorithm):
    """Contains functions to return result of fitness function.
    and run classifier algorithm.

    Attributes
    ----------
    metric : string
        The metric to be used to test the classifier.

    Methods
    -------
    evaluate(predictions, targets)
        Returns the error/fitness rate of predictions.

    pipe_evaluate(data)
        Calls the evaluate method within the context of the
        pipeline.

    pipe(data)
        Evaluates the classifier on the dataset as the final stage
        of the classifier pipeline.
    """

    def __init__(self, paramlist=None, metric="simple"):
        """Generate algorithm params from parameter list."""
        super(ClassifierFitness, self).__init__(paramlist)
        self.metric = metric

    def evaluate(self, predictions, targets):
        """
        Returns the error rate/fitness score of predictions.

        Parameters
        ----------
        predictions : array-like of shape (n_samples,)
            The predicted labels of each item.

        targets : array-like of shape (n_samples,)
            The target labels to predict.

        Returns
        -------
        The error/fitness rate of predictions.
        """

        length = len(predictions)
        return float(length - np.sum(predictions == targets)) / length

    def pipe_evaluate(self, data):
        """
        Determines the fitness value of the attached classifier.

        Parameters
        ----------
        data : PipelineClassifyDataset

        Returns
        -------
        fitness : float
            The fitness score of the classifier (data.clf) after
            trained on the training set and tested on the testing
            set.

        Notes
        -----
        This method should be overridden by subclasses.
        """

        if data.testing_set is None:
            raise ValueError("Testing set cannot be none")
        if len(data.testing_set.X) <= 0:
            raise ValueError("Testing set must have at least one item")
        clf = data.clf

        clf.fit(data.training_set.X, data.training_set.y)

        predictions = clf.predict(data.testing_set.X)

        return self.evaluate(predictions, data.testing_set.y)

    def pipe(self, data):
        """
        Evaluates the classifier on the dataset as the final stage
        of the classifier pipeline.

        Parameters
        ----------
        data : PipelineClassifyDataset

        Returns
        -------
        data : PipelineClassifyDataset
            Attaches the fitness score to the data object.

        Notes
        -----
        Unless there is good reason to, one should not override this
        method.
        """

        if data.clf is None:
            print(
                "ERROR: classifier cannot be None. This must be set prior in the pipeline"
            )

        data.fitness = self.pipe_evaluate(data)

        return data


class CVFitness(ClassifierFitness):
    """Uses the Stratified Cross-Validaiton scheme to measure
    the fitness of a classifier algorithm.

    Attributes
    ----------
    cv : int
        The number of folds to split the dataset.

    Methods
    -------
    set_cv(cv)
        Class method that sets the cv class attribute.

    pipe_evaluate(predictions, targets)
        Returns the average cross validation error
        of the classifier (data.clf).

    Notes
    -----
    When this is used during the classifier pipeline (i.e. as the 
    last item of a Workflow), the class attribute cv will be
    used to initialize this fitness instance by default. The
    default cv class attribute is 5. To change this use
    the class method CVFitness#set_cv(cv).
    """

    cv = 5

    def __init__(self, paramlist=None, cv=None):
        super(CVFitness, self).__init__(paramlist=paramlist, metric="CV")
        if cv is None:
            self.cv = CVFitness.cv
        else:
            self.cv = cv

    def pipe_evaluate(self, data):
        """
        Determines the fitness value of the attached classifier.

        Parameters
        ----------
        data : PipelineClassifyDataset

        Returns
        -------
        data : PipelineClassifyDataset
        """

        if data.training_set is None:
            raise ValueError("Training set cannot be none")
        if len(data.training_set.X) <= 0:
            raise ValueError("Training set must have at least one item")

        cv_fitness = cross_val_score(
            data.clf, data.training_set.X, data.training_set.y, cv=self.cv
        ).mean()
        cv_fitness = 1 - cv_fitness
        print("cv_fitness: ", cv_fitness)
        print("type cv_fitness: ", type(cv_fitness))
        return cv_fitness

    @classmethod
    def set_cv(clf, cv):
        """
        Class method that sets the cv class attribute.

        Parameters
        ----------
        cv : int
            The number of folds to split a dataset.

        Side Effects
        ------------
        Sets the class attribute cv. This should be done only once at the beginning.
        Instances of this class will use the class cv attribute to determine the
        number of splits to use for cross validation.

        Returns
        -------
        None
        """

        if type(cv) != int:
            raise ValueError("cv must be an int")

        clf.cv = cv
