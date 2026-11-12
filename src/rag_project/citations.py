"""Helpers for resolving the numbered sources cited in generated answers."""

import re


def cited_source_numbers(answer: str, available_count: int) -> tuple[int, ...]:
    """Return unique, valid source numbers in the order they are cited."""
    if available_count < 0:
        raise ValueError("The available source count cannot be negative.")
    numbers: list[int] = []
    for value in re.findall(r"\[Source\s+(\d+)\]", answer, re.I):
        number = int(value)
        if 1 <= number <= available_count and number not in numbers:
            numbers.append(number)
    return tuple(numbers)
