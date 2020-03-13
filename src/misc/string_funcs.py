import typing


def split_on_upper_case(string: str, strip=True) -> typing.List[str]:
    """Splits a string on upper case and returns the substrings in a list."""
    splits = []
    start = 0
    end = 0
    for i, c in enumerate(string):
        if c.isupper():
            if end > start:
                substring = string[start:end].strip() if strip else string[start:end]
                splits.append(substring)
            start = i
        end += 1

    splits.append(string[start:end])

    return splits


def snake_case_to_names(string: str) -> typing.List[str]:
    """Converts a string written in 'snake_case' to a list of names, where each words first letter is upper case."""
    splits = []
    words = string.split("_")
    for w in words:
        if w:
            w = w[0].upper() + w[1:]
            splits.append(w)

    return splits
