"""
Reusable decorators for common functionality (DRY principle).
Includes timing, caching, logging, and error handling decorators.
"""

import functools
import time
from typing import Any, Callable, Dict, Optional

from loguru import logger


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to measure and log function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result

    return wrapper


def cache_result(ttl: Optional[int] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to cache function results.

    Args:
        ttl: Time-to-live in seconds (None for no expiration)

    Returns:
        Decorator function
    """
    cache: Dict[str, tuple[Any, float]] = {}

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Check if cached result exists and is not expired
            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or (time.time() - timestamp) < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
                else:
                    logger.debug(f"Cache expired for {func.__name__}")

            # Call function and cache result
            logger.debug(f"Cache miss for {func.__name__}, computing...")
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result

        def clear_cache() -> None:
            """Clear the cache for this function."""
            cache.clear()
            logger.info(f"Cache cleared for {func.__name__}")

        wrapper.clear_cache = clear_cache  # type: ignore
        return wrapper

    return decorator


def log_execution(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to log function entry and exit.

    Args:
        func: Function to log

    Returns:
        Wrapped function that logs entry/exit
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.info(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Exiting {func.__name__} successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise

    return wrapper


def validate_input(**validators: Callable[[Any], bool]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to validate function inputs.

    Args:
        validators: Keyword arguments mapping parameter names to validation functions

    Returns:
        Decorator function

    Example:
        @validate_input(x=lambda x: x > 0, y=lambda y: isinstance(y, str))
        def my_func(x, y):
            pass
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function signature to map args to param names
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Validation failed for parameter '{param_name}' " f"with value {value}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator
