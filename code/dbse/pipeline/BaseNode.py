from abc import ABCMeta, abstractmethod

class BaseNode(object):
    __metaclass__ = ABCMeta


    def __init__(self):
        """
        Initializes the PipelineNode using the given parameters
        :param parameters: 
        """
        # perform parameter check, defined by child class
        self.check_params()

    def check_params(self):
        """
        Checks the parameter dictionary matches the requirements.
        :return: True if the parameters match the requirements, else False.
        """
        raise NotImplementedError("Must be overwritten by child-class.")

    def check_data(self, data):
        """
        Checks if the given data dictionary matches the requirements.
        :param data: the data dictionary to be checked
        :return: True if the data match the requirements, else False.
        """
        raise NotImplementedError("Must override run")

    def _pre_run_checks(self, data):
        """
        This method should not be called manually. It is must be only called from run of the parent (PipelineNode)
        class
        :param data: the data to check
        :return: 
        """
        assert isinstance(data, dict), 'data need to be of type dict, current type is {}'.format(str(type(data)))

        if self.check_data(data) is False:
            raise ValueError('Data does not match the required shape.')
            return
        # params are checked before run during constructor

    @abstractmethod
    def run(self, data):
        """
        Runs the nodes functionality on the data and returns the data dictionary, including the new, generated fields
        as well as a metrics dictionary, holding eventual results.
        :param data: the data to do the calculation
        :return: tuple of (enhanced) dataset and the metrics
        """

        self._pre_run_checks(data=data)

