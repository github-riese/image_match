from os import scandir, path, DirEntry
from typing import Callable, Optional


def scan_dir(directory: str, accept_fn: Optional[Callable[[DirEntry, int], bool]]):
    yield from _scan_dir(directory, accept_fn, 0)


def _scan_dir(directory: str, accept_fn, depth: int = 0):
    for entry in scandir(directory):
        if entry.name[0] == '.':
            continue
        if path.isdir(entry.path):
            if accept_fn is not None and not accept_fn(entry, depth):
                continue
            else:
                yield from _scan_dir(entry.path, accept_fn, depth + 1)
                continue
        yield entry.path
