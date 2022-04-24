"""
Pipeline derived datasets
"""

__author__ = "Elisha Yadgaran"


from simpleml.utils.errors import DatasetError

from .base import BaseNumpyDataset


class NumpyPipelineDataset(BaseNumpyDataset):
    """
    Dataset class with a predefined build
    routine, assuming dataset pipeline existence.

    WARNING: this class will fail if build_dataframe is not overwritten or a
    pipeline provided!
    """

    def build_dataframe(self) -> None:
        """
        Transform raw dataset via dataset pipeline for production ready dataset
        Overwrite this method to disable raw dataset requirement
        """
        if self.pipeline is None:
            raise DatasetError("Must set pipeline before building dataframe")

        split_names = self.pipeline.get_split_names()
        self.dataframe = dict(
            [
                (split_name, self.pipeline.transform(X=None, split=split_name))
                for split_name in split_names
            ]
        )
