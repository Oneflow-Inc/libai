# coding=utf-8
# Copyright 2021 The OneFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Code are mainly adopted from
https://github.com/facebookresearch/iopath/blob/main/iopath/common/file_io.py
"""


import base64
import concurrent.futures
import errno
import logging
import os
import shutil
import tempfile
import traceback
import uuid
from collections import OrderedDict
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Set,
    Union,
)
from urllib.parse import urlparse

from libai.utils.download import download
from libai.utils.non_blocking_io import NonBlockingIOManager


__all__ = ["LazyPath", "PathManager", "get_cache_dir", "file_lock"]

def get_cache_dir(cache_dir: Optional[str] = None) -> str:
    """
    Returns a default directory to cache static files
    (usually downloaded from Internet), if None is provided.

    Args:
        cache_dir (None or str): if not None, will be returned as is.
            If None, returns the default cache directory as:

        1) $LIBAI_CACHE, if set
        2) otherwise ~/.oneflow/iopath_cache
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(
            os.getenv("LIBAI_CACHE", "~/.oneflow/iopath_cache")
        )
    try:
        g_pathmgr.mkdirs(cache_dir)
        assert os.access(cache_dir, os.W_OK)
    except (OSError, AssertionError):
        tmp_dir = os.path.join(tempfile.gettempdir(), "iopath_cache")
        logger = logging.getLogger(__name__)
        logger.warning(f"{cache_dir} is not accessible! Using {tmp_dir} instead!")
        cache_dir = tmp_dir
    return cache_dir


def file_lock(path: str):  # type: ignore
    """
    A file lock. Once entered, it is guaranteed that no one else holds the
    same lock. Others trying to enter the lock will block for 30 minutes and
    raise an exception.
    This is useful to make sure workers don't cache files to the same location.
    Args:
        path (str): a path to be locked. This function will create a lock named
            `path + ".lock"`
    Examples:
        filename = "/path/to/file"
        with file_lock(filename):
            if not os.path.isfile(filename):
                do_create_file()
    """
    dirname = os.path.dirname(path)
    try:
        os.makedirs(dirname, exist_ok=True)
    except OSError:
        # makedir is not atomic. Exceptions can happen when multiple workers try
        # to create the same dir, despite exist_ok=True.
        # When this happens, we assume the dir is created and proceed to creating
        # the lock. If failed to create the directory, the next line will raise
        # exceptions.
        pass
    return portalocker.Lock(path + ".lock", timeout=3600)  # type: ignore


class LazyPath(os.PathLike):
    """
    A path that's lazily evaluated when it's used.

    Users should be careful to not use it like a str, because
    it behaves differently from a str.
    Path manipulation functions in Python such as `os.path.*` all accept
    PathLike objects already.

    It can be materialized to a str using `os.fspath`.
    """

    def __init__(self, func: Callable[[], str]) -> None:
        """
        Args:
            func: a function that takes no arguments and returns the
                actual path as a str. It will be called at most once.
        """
        self._func = func
        self._value: Optional[str] = None
    
    def _get_value(self) -> str:
        if self._value is None:
            self._value = self._func()
        return self._value  # pyre-ignore
    
    def __fspath__(self) -> str:
        return self._get_value()

    # before more like a str after evaluated
    def __getattr__(self, name: str):  # type: ignore
        if self._value is None:
            raise AttributeError(f"Uninitialized LazyPath has no attribute: {name}.")
        return getattr(self._value, name)
    
    def __getitem__(self, key):  # type: ignore
        if self._value is None:
            raise TypeError("Uninitialized LazyPath is not subscriptable.")
        return self._value[key]  # type: ignore

    def __str__(self) -> str:
        if self._value is not None:
            return self._value  # type: ignore
        else:
            return super().__str__()


class PathHandler:
    """
    PathHandler is a base class that defines common I/O functionality for a URI
    protocol. It routes I/O for a generic URI which may look like "protocol://*"
    or a canonical filepath "/foo/bar/baz".
    """

    _strict_kwargs_check = True

    def __init__(
        self,
        async_executor: Optional[concurrent.futures.Executor] = None,
    ) -> None:
        """
        When registering a `PathHandler`, the user can optionally pass in a
        `Executor` to run the asynchronous file operations.
        NOTE: For regular non-async operations of `PathManager`, there is
        no need to pass `async_executor`.

        Args:
            async_executor (optional `Executor`): Used for async file operations.
                Usage:
                ```
                    path_handler = NativePathHandler(async_executor=exe)
                    path_manager.register_handler(path_handler)
                ```
        """
        self._non_blocking_io_manager = None
        self._non_blocking_io_executor = async_executor
    
    def _check_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Checks if the given arguments are empty. Throws a ValueError if strict
        kwargs checking is enabled and args are non-empty. If strict kwargs
        checking is disabled, only a warning is logged.

        Args:
            kwargs (Dict[str, Any])
        """
        if self._strict_kwargs_check:
            if len(kwargs) > 0:
                raise ValueError("Unused arguments: {}".format(kwargs))
        else:
            logger = logging.getLogger(__name__)
            for k, v in kwargs.items():
                logger.warning("[PathManager] {}={} argument ignored".format(k, v))


    def _get_supported_prefixes(self) -> List[str]:
        """
        Returns:
            List[str]: the list of URI prefixes this PathHandler can support
        """
        raise NotImplementedError()

    def _get_local_path(self, path: str, force: bool = False, **kwargs: Any) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.
        If URI points to a remote resource, this function may download and cache
        the resource to local disk. In this case, the cache stays on filesystem
        (under `file_io.get_cache_dir()`) and will be used by a different run.
        Therefore this function is meant to be used with read-only resources.
        Args:
            path (str): A URI supported by this PathHandler
            force(bool): Forces a download from backend if set to True.
        Returns:
            local_path (str): a file path which exists on the local file system
        """
        raise NotImplementedError()

    def _copy_from_local(
        self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> None:
        """
        Copies a local file to the specified URI.
        If the URI is another local path, this should be functionally identical
        to copy.
        Args:
            local_path (str): a file path which exists on the local file system
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing URI
        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def _opent(
        self, path: str, mode: str = "r", buffering: int = 32, **kwargs: Any
        ) ->Iterable[Any]:
        raise NotImplementedError()

    def _open(
        self, path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI, similar to the built-in `open`.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy depends on the
                underlying I/O implementation.

        Returns:
            file: a file-like object.
        """
        raise NotImplementedError()

    def _opena(
        self,
        path: str,
        mode: str = "r",
        callback_after_file_close: Optional[Callable[[None], None]] = None,
        buffering: int = -1,
        **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI with asynchronous permissions.

        NOTE: Writes to the same path are serialized so they are written in
        the same order as they were called but writes to distinct paths can
        happen concurrently.

        Usage (default / without callback function):
            for n in range(50):
                results = run_a_large_task(n)
                with path_manager.opena(uri, "w") as f:
                    f.write(results)            # Runs in separate thread
                # Main process returns immediately and continues to next iteration
            path_manager.async_close()

        Usage (advanced / with callback function):
            # To write local and then copy to Manifold:
            def cb():
                path_manager.copy_from_local(
                    "checkpoint.pt", "manifold://path/to/bucket"
                )
            f = pm.opena("checkpoint.pt", "wb", callback_after_file_close=cb)
            flow.save({...}, f)
            f.close()

        Args:
            ...same args as `_open`...
            callback_after_file_close (Callable): An optional argument that can
                be passed to perform operations that depend on the asynchronous
                writes being completed. The file is first written to the local
                disk and then the callback is executed.
            buffering (int): An optional argument to set the buffer size for
                buffered asynchronous writing.

        Returns:
            file: a file-like object with asynchronous methods.
        """
        # Restrict mode until `NonBlockingIO` has async read feature.
        valid_modes = {'w', 'a', 'b'}
        if not all(m in valid_modes for m in mode):
            raise ValueError("`opena` mode must be write or append")
        
        # TODO: Each `PathHandler` should set its own `self._buffered`
        # parameter and pass that in here. Until then, we assume no
        # buffering for any storage backend.
        if not self._non_blocking_io_manager:
            self._non_blocking_io_manager = NonBlockingIOManager(
                buffered=False,
                executor=self._non_blocking_io_executor,
            )

        try:
            return self._non_blocking_io_manager.get_non_blocking_io(
                path=self._get_path_with_cwd(path),
                io_obj=self._open(path, mode, **kwargs),
                callback_after_file_close=callback_after_file_close,
                buffering=buffering,
            )
        except ValueError:
            # When `_strict_kwargs_check = True`, then `open_callable`
            # will throw a `ValueError`. This generic `_opena` function
            # does not check the kwargs since it may include any `_open`
            # args like `encoding`, `ttl`, `has_user_data`, etc.
            logger = logging.getLogger(__name__)
            logger.exception(
                "An exception occurred in `NonBlockingIOManager`. This "
                "is most likely due to invalid `opena` args. Make sure "
                "they match the `open` args for the `PathHandler`."
            )
            self._async_close()
        
    def _async_join(self, path: Optional[str] = None, **kwargs: Any) -> bool:
        """
        Ensures that desired async write threads are properly joined.

        Args:
            path (str): Pass in a file path to wait until all asynchronous
                activity for that path is complete. If no path is passed in,
                then this will wait until all asynchronous jobs are complete.

        Returns:
            status (bool): True on success
        """
        if not self._non_blocking_io_manager:
            logger = logging.getLogger(__name__)
            logger.warning(
                "This is an async feature. No threads to join because "
                "`opena` was not used."
            )
        self._check_kwargs(kwargs)
        return self._non_blocking_io_manager._join(
            self._get_path_with_cwd(path) if path else None
        )
    
    def _async_close(self, **kwargs: Any) -> bool:
        """
        Closes the thread pool used for the asynchronous operations.

        Returns:
            status (bool): True on success
        """
        if not self._non_blocking_io_manager:
            logger = logging.getLogger(__name__)
            logger.warning(
                "This is an async feature. No threadpool to close because "
                "`opena` was not used."
            )
        self._check_kwargs(kwargs)
        return self._non_blocking_io_manager._close_thread_pool()

    def _copy(
        self, src_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def _mv(
        self, src_path: str, dst_path: str, **kwargs: Any
    ) -> bool:
        """
        Moves (renames) a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler

        Returns:
            status (bool): True on success
        """
        raise NotImplementedError()

    def _exists(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        raise NotImplementedError()

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        raise NotImplementedError()

    def _isdir(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        raise NotImplementedError()

    def _ls(self, path: str, **kwargs: Any) -> List[str]:
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        raise NotImplementedError()

    def _mkdirs(self, path: str, **kwargs: Any) -> None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.

        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()

    def _rm(self, path: str, **kwargs: Any) -> None:
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()

    def _symlink(self, src_path: str, dst_path: str, **kwargs: Any) -> bool:
        """
        Symlink the src_path to the dst_path

        Args:
            src_path (str): A URI supported by this PathHandler to symlink from
            dst_path (str): A URI supported by this PathHandler to symlink to
        """
        raise NotImplementedError()

    def _set_cwd(self, path: Union[str, None], **kwargs: Any) -> bool:
        """
        Set the current working directory. PathHandler classes prepend the cwd
        to all URI paths that are handled.

        Args:
            path (str) or None: A URI supported by this PathHandler. Must be a valid
                absolute path or None to set the cwd to None.

        Returns:
            bool: true if cwd was set without errors
        """
        raise NotImplementedError()

    def _get_path_with_cwd(self, path: str) -> str:
        """
        Default implementation. PathHandler classes that provide a `_set_cwd`
        feature should also override this `_get_path_with_cwd` method.

        Args:
            path (str): A URI supported by this PathHandler.

        Returns:
            path (str): Full path with the cwd attached.
        """
        return path


