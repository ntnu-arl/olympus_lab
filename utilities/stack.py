from typing import Callable

import torch


class Stack:
    def __init__(self, generator: Callable[...,torch.Tensor]):
        self._generator = generator
        self._stack_pointer = 0
        self._stack= generator()
        self._stack_capasity = self._stack.shape[0]
        self._device = self._stack.device


    def pop(self, num_elements: int) -> torch.Tensor:
        out = torch.zeros(num_elements, *self._stack.shape[1:], device=self._device)    
        stack_shortage = num_elements - self._current_stack_size()

        if stack_shortage <= 0: # no need to generate new stack
            out[:] = self._stack[self._stack_pointer : self._stack_pointer + num_elements]
            self._stack_pointer += num_elements
            return out
        
        
        out[:self._current_stack_size()] = self._stack[self._stack_pointer :]
        self._reset_stack()
        out[-stack_shortage:] = self.pop(stack_shortage)
        return out
        
        


    def _current_stack_size(self) -> int:
        return self._stack_capasity - self._stack_pointer
    
    def _reset_stack(self) -> None:
        self._stack_pointer = 0
        self._stack[:] = self._generator()



if __name__ == "__main__":
    def generator():
        return torch.rand((10, 3))

    stack = Stack(generator)

    for i in range(15):
        print(i)
        a = stack.pop(i)