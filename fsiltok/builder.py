import numpy as np
import tempfile
import os

class DatasetBuilder:

    capacity: int = 1_000_000_000  # Default capacity
    dtype: np.dtype = np.uint16  # Default data type

    _memmap: np.memmap = None
    _tmp_path: str = None
    _prefix: os.PathLike = None
    _size: int = 0

    def __init__(self, prefix: os.PathLike, dtype: np.dtype = np.uint16):

        assert len(prefix.split('.')) == 1, "Path must not contain a file extension."

        self._prefix = prefix
        self.dirname = os.path.dirname(prefix)
        self.offsets = [0] # Start with an initial offset of 0
        self.dtype = dtype

        self._make_memmap()

    def _make_memmap(self) -> None:
        tmp_path = tempfile.mktemp(suffix='.tmp', dir=self.dirname)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        self._memmap = np.memmap(tmp_path, dtype=self.dtype, mode='w+', shape=(self.capacity,))
        self._tmp_path = tmp_path

    def _resize_memmap(self, to_capacity: int) -> None:
        print("Resizing memory map to", to_capacity)
        assert self._memmap is not None, "Memory map not initialized."
        assert to_capacity >= self._size, "New capacity must be greater than or equal to current size."

        # Make a tempfile name
        new_tmp_path = tempfile.mktemp(suffix='.tmp', dir=self.dirname)
        if os.path.exists(new_tmp_path):
            os.remove(new_tmp_path)
        new_memmap = np.memmap(new_tmp_path, dtype=self.dtype, mode='w+', shape=(to_capacity,))
        new_memmap[:self._size] = self._memmap[:self._size]
        self._memmap = new_memmap
        os.remove(self._tmp_path)
        self._tmp_path = new_tmp_path
        self.capacity = to_capacity

    def finalize(self) -> None:
        assert self._memmap is not None, "Memory map not initialized."

        if self._size < self.capacity:
            self._resize_memmap(self._size)
        
        del self._memmap
        os.rename(self._tmp_path, f"{self._prefix}.bin")

        # Write offsets to a separate file
        offsets = np.memmap(f"{self._prefix}.idx", dtype=np.uint64, mode='w+', shape=(len(self.offsets),))
        offsets[:] = self.offsets
        del offsets
    
    def extend(self, data: np.ndarray) -> None:
        assert self._memmap is not None, "Memory map not initialized."
        assert data.ndim == 1, "Data must be a 1D array."
        assert data.dtype == self.dtype, f"Data type must be {self.dtype}, got {data.dtype}."

        if self._size + len(data) > self.capacity:
            new_capacity = max(self.capacity * 2, self._size + len(data))
            self._resize_memmap(new_capacity)

        self._memmap[self._size:self._size + len(data)] = data
        self._size += len(data)
        self.offsets.append(self._size)

