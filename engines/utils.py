import logging
from functools import partial
from typing import Callable

logger = logging.getLogger(__name__)


class ModelFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """ """

        def inner_wrapper(wrapped_class: Callable) -> Callable:
            if name in cls.registry:
                logger.warning("Model %s already exists. Will replace it", name)  # noqa
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str) -> Callable:
        assert name in cls.registry, f"Model {name} does not exist in."  # noqa
        return cls.registry[name]


class DataFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """ """

        def inner_wrapper(wrapped_class: Callable) -> Callable:
            if name in cls.registry:
                logger.warning("Model %s already exists. Will replace it", name)  # noqa
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str) -> Callable:
        assert name in cls.registry, f"Model {name} does not exist in."  # noqa
        return cls.registry[name]


class TrainerFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Callable) -> Callable:
            if name in cls.registry:
                logger.warning(
                    "Trainer %s already exists. Will replace it", name
                )  # noqa
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str) -> Callable:
        assert name in cls.registry, f"Trainer {name} does not exist in."  # noqa
        return cls.registry[name]


class RewardFunctionFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Callable) -> Callable:
            if name in cls.registry:
                logger.warning(
                    "Trainer %s already exists. Will replace it", name
                )  # noqa
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str, **kwargs) -> Callable:
        assert name in cls.registry, f"Trainer {name} does not exist in."  # noqa
        return partial(cls.registry[name], **kwargs)
