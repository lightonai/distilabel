import hashlib
import json
import logging
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from pydantic import NonNegativeInt
from typing import TYPE_CHECKING, Dict, Generator, Union

import portalocker
from pydantic import BaseModel, Field, PrivateAttr

from distilabel.mixins.runtime_parameters import RuntimeParameter

if TYPE_CHECKING:
    from logging import Logger


_VLLM_SERVER_SHARING_FILE = None

def set_vllm_server_sharing_file(pipeline_name: str) -> Path:
    global _VLLM_SERVER_SHARING_FILE
    _VLLM_SERVER_SHARING_FILE = (
        Path(tempfile.gettempdir())
        / "distilabel"
        / "vllm_server_sharing"
        / socket.gethostname()
        / pipeline_name
        / "distilabel_vllm_server_sharing.json"
    )
    return _VLLM_SERVER_SHARING_FILE


_logger = logging.getLogger('vllm_server_sharing')


class VLLMServerSharingMixin(BaseModel):
    """Mixin class to enable multiple LLM replicas to share the same vLLM server instance.
    
    This allows oversubscription where N replicas can share a single vLLM server,
    reducing GPU memory usage and startup time.

    Attributes:
        disable_vllm_server_sharing: Whether to disable the vLLM server sharing logic
            or not. Defaults to `False`.
        _replica_id: the replica ID of the `LLM`.
        _llm_identifier: the identifier of the `LLM` replica
        _server_info: information about the shared server (port, pid, etc.)
    """

    disable_vllm_server_sharing: RuntimeParameter[bool] = Field(
        default=False,
        description="Whether to disable the vLLM server sharing logic or not.",
    )
    replicas_per_vllm_server: RuntimeParameter[int] = Field(
        default=1,
        description="Number of replicas that share a single vLLM server instance.",
    )
    model_path: RuntimeParameter[str] = Field(
        default="",
        description="The model path to be used.",
    )
    
    _replica_id: NonNegativeInt = PrivateAttr(default=0)
    _server_info: Dict[str, any] = PrivateAttr(default_factory=dict)
    _cleanup: bool = PrivateAttr(default=False)
    _llm_identifier: Union[str, None] = PrivateAttr(default=None)
    _is_server_owner: bool = PrivateAttr(default=False)

    def load(self) -> None:
        """Assigns this LLM replica to a shared vLLM server or starts a new one.
            
        self._server_info will be updated with the server information which
            should be used to start or join the server
        """
        if self.disable_vllm_server_sharing:
            # No sharing, act as server owner
            self._server_info = {'port': None, 'pid': None, 'is_owner': True}
            return

        if self.replicas_per_vllm_server <= 1:
            # No sharing configured
            self._server_info = {'port': None, 'pid': None, 'is_owner': True}
            return

        server_key = self._compute_server_key()

        # First-come election under a file lock to avoid races
        creator = False
        with self._server_sharing_map() as sharing_map:
            server_data = sharing_map.get(server_key)
            if server_data is None:
                # Elect this replica as owner and create entry atomically
                creator = True
                self._is_server_owner = True
                sharing_map[server_key] = {
                    'model_path': self.model_path,
                    'port': None,
                    'pid': None,
                    'owner_identifier': self._llm_identifier,
                    'owner_replica_id': int(self._replica_id),
                    'replica_identifiers': [self._llm_identifier],
                }
                self._server_info = {
                    'port': None,
                    'pid': None,
                    'is_owner': True,
                    'server_key': server_key,
                }
            else:
                # Join existing server (may not be ready yet)
                self._is_server_owner = False
                server_data['replica_identifiers'] = list(set(server_data.get('replica_identifiers', []) + [self._llm_identifier]))
                port = server_data.get('port')
                pid = server_data.get('pid')
                self._server_info = {
                    'port': port,
                    'pid': pid,
                    'is_owner': False,
                    'server_key': server_key,
                }

        if creator:
            _logger.info(
                f"🚀 LLM '{self._llm_identifier}' key='{server_key}' elected as owner of new vLLM server "
                f"(will support {self.replicas_per_vllm_server} replicas)"
            )
        else:
            # Wait until the owner populates port/pid
            ready = self._wait_for_server_ready(server_key)
            self._server_info['port'] = ready['port']
            self._server_info['pid'] = ready['pid']
            _logger.info(
                f"🔗 LLM '{self._llm_identifier}' joining existing vLLM server on port {ready['port']} "
                f"(replica {len(server_data['replica_identifiers'])}/{self.replicas_per_vllm_server})"
            )

    def update_server_info(self, port: int, pid: int) -> None:
        """Update the shared server info with port and PID after server starts.
        
        Should only be called by the server owner after starting vLLM.
        """
        if not self._is_server_owner:
            return
            
        server_key = self._server_info['server_key']
        with self._server_sharing_map() as sharing_map:
            sharing_map[server_key]['port'] = port
            sharing_map[server_key]['pid'] = pid
            self._server_info['port'] = port
            self._server_info['pid'] = pid
            _logger.info(
                f"🔗 LLM '{self._llm_identifier}' key='{server_key}' updated server info with port {port} and pid {pid}"
            )

    def unload(self) -> None:
        """Unloads the LLM and removes it from the shared server.
        
        If this is the last replica using the server, returns True to signal
        that the server should be cleaned up.
        
        Sets:
            self._cleanup: bool, True if this was the last replica and server should be cleaned up
        """
        if self.disable_vllm_server_sharing or not self._server_info:
            self._cleanup = True  # Clean up as normal
            return
            
        server_key = self._server_info.get('server_key')
        if not server_key:
            self._cleanup = True
            return
            
        with self._server_sharing_map() as sharing_map:
            if server_key not in sharing_map:
                self._cleanup = True
                return
                
            server_data = sharing_map[server_key]
            if self._llm_identifier in server_data['replica_identifiers']:
                server_data['replica_identifiers'].remove(self._llm_identifier)
                
                if len(server_data['replica_identifiers']) == 0:
                    # Last replica, clean up the server
                    _logger.info(
                        f"🧹 LLM '{self._llm_identifier}' was last replica, cleaning up server"
                    )
                    del sharing_map[server_key]
                    self._cleanup = True
                    return
                else:
                    # Other replicas still using this server
                    _logger.info(
                        f"🔌 LLM '{self._llm_identifier}' disconnecting from shared server "
                        f"({len(server_data['replica_identifiers'])} replicas remaining)"
                    )
                    self._cleanup = False
                    return
        
        self._cleanup = True

    @contextmanager
    def _server_sharing_map(self) -> Generator[Dict[str, Dict], None, None]:
        """Reads the content of the server sharing file with a lock, yields
        the content, and writes the content back to the file after the context manager is
        closed. If the file doesn't exist, an empty dictionary will be yielded.

        Yields:
            The content of the server sharing file.
        """
        _VLLM_SERVER_SHARING_FILE.parent.mkdir(parents=True, exist_ok=True)
        _VLLM_SERVER_SHARING_FILE.touch()
        with portalocker.Lock(
            _VLLM_SERVER_SHARING_FILE,
            "r+",
            flags=portalocker.LockFlags.EXCLUSIVE,
        ) as f:
            try:
                content = json.load(f)
            except json.JSONDecodeError:
                content = {}
            yield content
            f.seek(0)
            f.truncate()
            f.write(json.dumps(content, indent=2))
            f.flush()
            os.fsync(f.fileno())

    def _compute_server_key(self) -> str:
        """Compute a unique key for a vLLM server configuration.
        
        Servers with the same key can be shared. The key is based on:
        - Server ID (replica_id // replicas_per_vllm_server)
        - Model path
        """
        key_data = {
            'server_id': self._replica_id // self.replicas_per_vllm_server,
            'model_path': self.model_path,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _wait_for_server_ready(self, server_key: str, timeout: int = 1500) -> Dict[str, Union[int, None]]:
        """Block until the server owner updates port and pid for the given server key."""
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            with self._server_sharing_map() as sharing_map:
                data = sharing_map.get(server_key) or {}
                port = data.get('port')
                pid = data.get('pid')
                if port is not None and pid is not None:
                    return {'port': port, 'pid': pid}
            time.sleep(1)
        raise RuntimeError(f"vLLM server for key {server_key} did not become ready within {timeout}s")

