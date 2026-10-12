from typing import List, Optional, Union

import numpy as np
import torch

__all__ = ['Context']


class Context:
    def __init__(self, device: Optional[torch.device | str] = None,
                 dtype: Optional[torch.dtype] = None,
                 use_native: Optional[bool] = None):
        device = self._resolve_device(device, use_native)
        if use_native is None:
            use_native = device.type == 'cuda'

        if device.type == 'cuda':
            assert torch.cuda.is_available(), \
                ('cuda device explicitly requested but cuda is not available!')
        elif device.type == 'mps':
            assert self._mps_available(), \
                ('mps device explicitly requested but mps is not available!')
        else:
            assert device.type == 'cpu', \
                (f"lettuce is designed to work on cpu, mps or cuda devices. "
                 f"{device} is not supported!")

        assert not use_native or device.type == 'cuda', \
            ('can not use explicitly requested cuda_native extension '
             'on non-cuda device!')

        dtype = dtype or torch.float32  # default dtype to single
        assert dtype in [torch.float16, torch.float32, torch.float64], \
            (f"lettuce is designed to work with common float types "
             f"(16, 32 and 64 bit). {dtype.__name__} is not supported!")
        assert device.type != 'mps' or dtype != torch.float64, \
            ('mps does not support double precision; use torch.float32 '
             'or torch.float16.')

        # store context configuration

        self.device = torch.device(device)
        self.dtype = dtype
        self.use_native = use_native

    @classmethod
    def _resolve_device(cls, device: Optional[torch.device | str],
                        use_native: Optional[bool]) -> torch.device:
        if device is None:
            if use_native:
                assert torch.cuda.is_available(), \
                    ('cuda_native extension explicitly requested but '
                     'cuda is not available!')
                return torch.device('cuda:0')
            return cls._auto_device()

        if isinstance(device, str):
            if device == 'auto':
                resolved = cls._auto_device()
                if use_native:
                    assert resolved.type == 'cuda', \
                        ('cuda_native extension explicitly requested but '
                         'cuda is not available!')
                return resolved
            if device == 'cuda':
                return torch.device('cuda:0')
        return torch.device(device)

    @staticmethod
    def _mps_available() -> bool:
        return (hasattr(torch.backends, 'mps')
                and torch.backends.mps.is_available())

    @classmethod
    def _auto_device(cls) -> torch.device:
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        if cls._mps_available():
            return torch.device('mps')
        return torch.device('cpu')

    def synchronize(self):
        """
        Use explicit sync to catch errors right away.
        Notice cost of sync, so better call only when necessary or in setup code.
        """
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        elif self.device.type == 'mps' and hasattr(torch, 'mps'):
            torch.mps.synchronize()

    def empty_tensor(self, size: Union[List[int], torch.Size], *args,
                     dtype=None, **kwargs) -> torch.Tensor:
        return torch.empty(size, *args, **kwargs, device=self.device,
                           dtype=(dtype or self.dtype))

    def zero_tensor(self, size: Union[List[int], torch.Size], *args,
                    dtype=None, **kwargs) -> torch.Tensor:
        return torch.zeros(size, *args, **kwargs, device=self.device,
                           dtype=(dtype or self.dtype))

    def full_tensor(self, size: Union[List[int], torch.Size], value, *args,
                    dtype=None, **kwargs) -> torch.Tensor:
        return torch.full(size, value, *args, **kwargs, device=self.device,
                           dtype=(dtype or self.dtype))

    def one_tensor(self, size: Union[List[int], torch.Size], *args, dtype=None,
                   **kwargs) -> torch.Tensor:
        return torch.ones(size, *args, **kwargs, device=self.device,
                          dtype=(dtype or self.dtype))

    def convert_to_tensor(self, array, *args,
                          dtype: Optional[torch.dtype] = None, **kwargs
                          ) -> torch.Tensor:
        is_tensor = isinstance(array, torch.Tensor)
        new_dtype = dtype

        if dtype is None:
            bool_types = [bool, torch.bool, torch.uint8, np.uint8]
            if hasattr(array, 'dtype') and array.dtype in bool_types:
                new_dtype = torch.uint8
            else:
                new_dtype = self.dtype

        # Convert nested lists to NumPy array first
        if is_tensor:
            return array.to(*args, **kwargs, device=self.device, dtype=new_dtype)

        array = np.array(array)
        array = np.ascontiguousarray(array)

        tensor = torch.from_numpy(array)
        tensor = tensor.to(*args, **kwargs, device=self.device, dtype=new_dtype)

        return tensor

    @staticmethod
    def convert_to_ndarray(tensor: Union[torch.Tensor, List]) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        else:
            return np.array(tensor)
