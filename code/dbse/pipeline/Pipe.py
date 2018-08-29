from abc import ABCMeta, abstractmethod
from dbse.pipeline.BaseNode import BaseNode
from dbse.tools.Logging import Logging

class Pipe(object):

    def __init__(self):
        """
        Initializes the PipelineNode using the given parameters
        :param parameters:
        """
        self._stages = {}
        self._data = None     # holds dataset after last stage from pipeline has been executed
        self._metrics = None  # holds metrics after last stage from pipeline has been executed

    def add_stage(self, stage_key, pipelinenodeclass, *parameters):
        """
        Adds a pipeline node to the execution
        :param PipelineNodeClass: class to add
        :return: True if the parameters match the requirements, else False.
        """
        assert isinstance(pipelinenodeclass, BaseNode)
        self._stages[stage_key] = pipelinenodeclass

    def run_stage(self, key, data=None):
        """
        
        :param key: The key referencing the stage
        :param data: data dictionary
        :return: 
        """

        # current pipelineNode
        pipelineNode = self._stages[key]
        Logging().log("Running Stage: {}".format(pipelineNode.__class__.__name__))

        # always update data from previous stage
        if data == None:
            data = self._data

        # run stage
        self._data, self._metrics = pipelineNode.run(data)

        return self._data, self._metrics

    def run_all_stages(self, data=None):

        for pipelineNode in self._stages:
            if data == None:
                data = self._data

            self._data, self._metrics = pipelineNode.run(data)

        return self._data, self._metrics

