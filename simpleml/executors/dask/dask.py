"""
Dask based executors
"""

__author__ = "Elisha Yadgaran"


from dask_ml.wrappers import Incremental, ParallelPostFit
from simpleml.executors.processor import AbstractBaseExecutor


class DaskExecutor(AbstractBaseExecutor):
    """
    Implements generic dask executor. Submits computation to the cluster and awaits futures.
    This implementation will not attempt to intuit handling for specific ml usage. Use specialized
    executors for that (see DaskIncrementalExecutor and DaskPostParallelExecutor)
    """

    pass


class DaskIncrementalExecutor(AbstractBaseExecutor):
    """
    Implements dask_ml.wrappers.Incremental to leverage partial_fit with blockwise processing
    and (implicitly) dask_ml.wrappers.ParallelPostFit for transformation

    Requires Incremental compatible objects and use case (incremental fit will not necessarily result
    in the same parameters as standard fit)

    see: https://ml.dask.org/incremental.html
    """

    @staticmethod
    def process(op, *args, **kwargs):
        # decompose op into class + attr to wrap
        obj = op.__self__
        attr = op.__name__

        wrapped = Incremental(obj)
        ret = getattr(wrapped, attr)(*args, **kwargs)

        # unwrap
        if isinstance(ret, Incremental):
            ret = ret.estimator_

        return ret


class DaskPostParallelExecutor(AbstractBaseExecutor):
    """
    Implements dask_ml.wrappers.ParallelPostFit to leverage massively parallel transformation
    processing

    see: https://ml.dask.org/meta-estimators.html
    """

    @staticmethod
    def process(op, *args, **kwargs):
        # decompose op into class + attr to wrap
        obj = op.__self__
        attr = op.__name__

        wrapped = ParallelPostFit(obj)
        ret = getattr(wrapped, attr)(*args, **kwargs)

        # unwrap
        if isinstance(ret, ParallelPostFit):
            ret = ret.estimator

        return ret
