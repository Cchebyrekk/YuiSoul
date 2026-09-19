# -*- coding: utf-8 -*-
"""
Утилиты для замера времени выполнения функций (логирование задержек).
"""
import time
import functools
from typing import Callable, Any


def log_latency(log_func: Callable[[str], None] = print) -> Callable:
    """
    Декоратор для замера времени выполнения функции.
    Выводит в лог имя функции и затраченное время в миллисекундах.

    :param log_func: функция для вывода сообщения (по умолчанию print).
    :return: декоратор.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # в миллисекундах
            log_func(f"[LATENCY] {func.__name__}: {elapsed:.2f} ms")
            return result
        return wrapper
    return decorator


def measure_time(func: Callable) -> Callable:
    """
    Декоратор-обёртка для замера времени, использующий log_latency с print.
    """
    return log_latency(print)(func)