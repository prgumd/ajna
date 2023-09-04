#!/usr/bin/env python
from functools import wraps

class Test():
    def __init__(self):
        self.count = 0
    
    def count(func):
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            self.count += 1        
            return func(self, *args, **kwargs)
        return wrapped

    @count
    def A(self):
        pass

    @count
    def B(self):
        pass



