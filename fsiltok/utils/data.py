import os
import numpy as np
from torch.utils.data import Dataset

# Dataset class for DocumentTapeDataset
# This class is designed to handle a dataset of documents stored in a memory-mapped file format
# It's mostly designed for pretraining, because it uses a fixed chunk size and
# packs sequences of tokens into attention-separated chunks.
class DocumentTapeDataset(Dataset):
    
    # General static parameters
    prefix : os.PathLike = None
    _token_dtype : np.dtype = None
    _offsets_dtype : np.dtype = np.uint64
    _chunk_size : int = 4096
    _eod_token_id: int = None  # End of Document token ID, if applicable

    _chunk_offsets: np.ndarray = None

    def __init__(self, prefix: os.PathLike, 
                token_dtype: np.dtype = np.uint16,
                chunk_size: int = 4096,
                eod_token_id: int = None):
        super().__init__()
        self.prefix = prefix
        self._token_dtype = token_dtype
        self._chunk_size = chunk_size
        self._eod_token_id = eod_token_id

        self._total_tokens = os.path.getsize(f"{self.prefix}.bin") // np.dtype(self._token_dtype).itemsize
        self._total_chunks = (self._total_tokens + self._chunk_size - 1) // self._chunk_size
        self._total_docs = os.path.getsize(f"{self.prefix}.idx") // np.dtype(self._offsets_dtype).itemsize

        self._handle = None # Lazy initialization for pickling
        self._offsets = None # Lazy initialization for pickling

    def __getitem__(self, index : int) -> np.ndarray:
        # Ensure index is within bounds
        if index < 0 or index >= self._total_docs:
            raise IndexError("Index out of bounds")
        
        # Fetch the offsets for the document
        if not self._handle:
            self._handle = np.memmap(f"{self.prefix}.bin", dtype=self._token_dtype, mode='r', shape=(self._total_tokens,))

        if not self._offsets:
            self._offsets = np.memmap(f"{self.prefix}.idx", dtype=self._offsets_dtype, mode='r', shape=(self._total_docs,))

        id_start = self._chunk_size * index
        id_end = min(id_start + self._chunk_size, self._total_tokens)

        tokens = self._handle[id_start:id_end]
        if len(tokens) == 0:
            return np.array([], dtype=self._token_dtype)
        
        # If the slice contains a document boundary, we need to fix position ids
        # This is done by checking the offsets

        if self._eod_token_id is None:
            return tokens

        ### Packed Position IDs

        resets = (tokens == self._eod_token_id)

        group_ids = np.cumsum(np.roll(resets, 1))
        group_ids[0] = 0 # Fix the first element
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...] <- idx
        # [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, ...] <- group ids
        # [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...] <- resets

        # [0, 1, 2, 0, 1, 2, 3, 4, 0, 1, ...] <- desired output

        _, first_indices, counts = np.unique(group_ids, return_index=True, return_counts=True)

        offsets = first_indices[group_ids]

        position_ids = np.arange(len(tokens), dtype=np.uint16) - offsets # <- Position IDs with resets

        ### Packed Causal Mask

        # Probably not the most efficient way. I'd really like to vectorize this.
        # But this is a simple way to get the causal mask for the packed tokens.
        # - Rodrigo

        mask = np.zeros((len(tokens), len(tokens)), dtype=np.bool_)

        for start, count in zip(first_indices, counts):
            end = start + count
            mask[start:end, start:end] = np.tri(count, k=0, dtype=np.bool_)

        return {
            "tokens": tokens,
            "position_ids": position_ids,
            "mask": mask
        }

    def __len__(self):
        return self._total_chunks