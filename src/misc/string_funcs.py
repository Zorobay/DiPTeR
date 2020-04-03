import typing

EXCLUDED_WORDS = ["RGB", "HSV"]


def split_on_upper_case(string: str, strip=True, ignore_excluded=True) -> typing.List[str]:
    """Splits a string on upper case and returns the substrings in a list."""
    splits = []
    ignored_indices = set()

    if ignore_excluded:
        for ex in EXCLUDED_WORDS:
            if ex in string:
                index = string.find(ex)
                ignored_indices.update(range(index, index + len(ex)))
                splits.append(ex)

    start = 0
    end = 0
    for i, c in enumerate(string):
        if i in ignored_indices:
            end += 1
            start += 1
            continue

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
