from abc import ABCMeta, abstractmethod
from pipeline.BaseNode import BaseNode


class TemplateNode(BaseNode):
    __metaclass__ = ABCMeta

    def __init__(self, param_1: int):
        """
        Initializes the PipelineNode using the given parameters
        :param parameters: 
        """
        # set params as internal variable
        self._param_1 = param_1

        # call parent constructor
        super().__init__()

    def check_params(self):
        """
        Checks the parameter dictionary matches the requirements. Is called during object construction
        from the parent class.
        :return: True if the parameters match the requirements, else False.
        """
        # TODO: Check your parameter here, e.g. using assert. The following checks may be of use
        # assert len(self._parameters) is 1, "number of params ({}) exceeds expectation (1)".format(len(self._parameters))
        # assert isinstance(self._parameters[0], int), "param should be of type int"
        raise NotImplementedError("Must override run")

    def check_data(self, data):
        """
        Checks if the given data dictionary matches the requirements.
        :param data: the data dictionary to be checked
        :return: True if the data match the requirements, else False.
        """
        # TODO: Check your data here, e.g. using assert. Most IMPORTANT: key for keys in the dictionary that
        # this pipelinestage needs.
        raise NotImplementedError("Must override run")

    def _my_helper(self, data):
        # modify data
        return data

    def run(self, data):
        """
        Runs the nodes functionality on the data and returns the data dictionary, including the new, generated fields
        as well as a metrics dictionary, holding eventual results.
        :param data: the data to do the calculation
        :return: tuple of (enhanced) dataset and the metrics
        """
        super().run(data)  # dont not remove this!

        # Do your calulations here+
        data = self._my_helper()

        # aggregate metrics
        metrics = dict()
        metrics['auc'] = 0.98

        # return final result
        return data, metrics

