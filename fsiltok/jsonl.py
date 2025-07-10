from typing import List
import os
import xopen
import orjson

class JSONLFileReader:
    def __init__(self, paths: List[os.PathLike], textField: str, threads: int = 0, compressLevel: int = 3):
        self.paths = paths
        self.textField = textField
        self.threads = threads
        self.compressLevel = compressLevel

    def __iter__(self):
        # Python will raise StopIteration naturally when file 
        
        for path in self.paths:
            with xopen.xopen(path, 'rt', threads=self.threads, compresslevel=self.compressLevel) as handle:
                for line in handle:
                    json = self._parse_line(line)
                    if json is not None:
                        yield json

    def _parse_line(self, line: str):
        # Returns the parsed JSON object from a line
        json = orjson.loads(line)
        if self.textField in json and json[self.textField] != "":
            return json[self.textField]
        return None