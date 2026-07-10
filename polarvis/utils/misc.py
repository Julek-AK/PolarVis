
# Builtin
from typing import List, Any, Generator


def split_batches(items: List[Any], batch_size: int):

    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]