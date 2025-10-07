import subprocess
import socket
import random
import time
import os
import signal
import logging

import openai

from distilabel import utils
from distilabel.pydantics import LMConfig

_logger = logging.getLogger('vllm_api')

class vLLMAPI:
    '''vLLM API base class.

    Handles starting and stopping the vLLM server.
    '''
    gpu: int = 0

    def __init__(self, lm_config: LMConfig):
        self.lm_config = lm_config

        self.num_gpus = 1
        self.vllm_server_pid = None

        self.port = self.random_available_port()
        os.makedirs('vllm_logs', exist_ok=True)

    def port_in_use(self, port: int) -> bool:
        '''Check if a port is unavailable for exclusive bind on localhost.'''
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            test_socket.bind(("127.0.0.1", port))
            # No listen; close immediately so the port is released.
            return False
        except OSError:
            return True
        finally:
            try:
                test_socket.close()
            except Exception:
                pass

    def random_available_port(self) -> int:
        '''Get a random available port.'''
        while True:
            port = random.randint(1024, 49151)
            if not self.port_in_use(port):
                return port

    def start_vllm(self, timeout=1500):
        '''Start the asynchronous vLLM server.'''
        # Build the vllm serve command as a list for subprocess
        launch_vllm = [
            "vllm", "serve", self.lm_config.path,
            "--tensor-parallel-size", str(self.lm_config.tp_size),
            "--disable-log-requests",
            "--trust-remote-code",
            "--port", str(self.port),
        ]
        if self.lm_config.pp_size:
            launch_vllm.extend(["--pipeline-parallel-size", str(self.lm_config.pp_size)])

        # Add any extra vllm_args from the config
        for k, v in getattr(self.lm_config, "vllm_kwargs", {}).items():
            flag = f"--{k.replace('_', '-')}"
            launch_vllm.append(flag) if v is None else launch_vllm.extend([flag, str(v)])

        os.makedirs(f"vllm_logs/{self.lm_config.path.replace('/', '-')}", exist_ok=True)
        launch_vllm.extend([
            ">", f"vllm_logs/{self.lm_config.path.replace('/', '-')}/{self.gpu}.log",
            "2>&1",
            "&",
            "echo", "$!",  # retrieve the PID of the actual vllm server
        ])
        
        _logger.info(f'[{self.gpu}] Initializing vLLM Server...')
        time.sleep(self.gpu * 5)
        with utils.suppress_output(debug=False):
            process = subprocess.Popen(  # noqa: S602
                ' '.join(launch_vllm),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()
            self.vllm_server_pid = int(stdout.strip().decode('utf-8'))

            err = (
                f'vllm server process {self.vllm_server_pid} on port {self.port} failed to become ready; '
                f'see vllm_logs/{self.lm_config.path.replace("/", "-")}/{self.gpu}.log'
            )

            # wait patiently for vllm server to start
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < timeout:
                # Early exit if the vllm process crashes during startup wait
                if not self._is_process_running(self.vllm_server_pid):
                    self.cleanup()
                    raise RuntimeError(err)
                try:
                    self.establish_client_vllm()
                    _logger.info(f'[{self.gpu}] vLLM Server Initialized')
                except openai.APIConnectionError:
                    time.sleep(5)
                else:
                    return

        self.cleanup()
        raise RuntimeError(err)

    def establish_client_vllm(self):
        '''Establish a openai client to the vLLM server.'''
        self.client = openai.OpenAI(api_key='empty', base_url=f'http://localhost:{self.port}/v1')
        self.model_name = self.client.models.list().data[0].id

    def _is_process_running(self, pid: int) -> bool:
        '''Return True if a process with the given PID is running.'''
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def wait_for_existing_server(self, port: int, pid: int, timeout=600):
        '''Wait for an existing vLLM server to be ready on the specified port.
        
        This is used when multiple replicas share a single vLLM server.
        One replica starts the server, and others wait for it to be ready.
        
        Args:
            port: The port the vLLM server is running on
            pid: The PID of the vLLM server process
            timeout: Maximum time to wait in seconds
        '''
        self.port = port
        self.vllm_server_pid = pid
        
        _logger.info(f'replica-id [{self.gpu}] Waiting for existing vLLM Server on port {port}...')

        err = (
            f'vllm server process {self.vllm_server_pid} on port {self.port} failed to become ready; '
            f'see vllm_logs/{self.lm_config.path.replace("/", "-")}/{self.gpu}.log'
        )
        
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            # Early exit if the existing server process has exited
            if not self._is_process_running(self.vllm_server_pid):
                raise RuntimeError(err)
            try:
                self.establish_client_vllm()
                _logger.info(f'replica-id [{self.gpu}] Connected to existing vLLM Server')
                return
            except openai.APIConnectionError:
                time.sleep(5)
        raise RuntimeError(err)

    def cleanup(self):
        '''Kill the vLLM server.'''
        if not self.vllm_server_pid:
            return

        pid = self.vllm_server_pid
        if self._is_process_running(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                _logger.info(f'Server with PID {pid} has been sent SIGTERM.')
                time.sleep(10)  # Allow time for cleanup
                if self._is_process_running(pid):
                    os.kill(pid, signal.SIGKILL)
                    _logger.info(f'Server with PID {pid} has been killed.')
            except ProcessLookupError:
                _logger.warning(f'Server with PID {pid} does not exist.')
