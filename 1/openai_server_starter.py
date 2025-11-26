import os
import subprocess
import sys
import threading
import time
from typing import List, Optional

import requests

from clarifai.utils.logging import logger
import psutil
import signal

PYTHON_EXEC = sys.executable


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
  """Kill the process and all its child processes."""
  if parent_pid is None:
    parent_pid = os.getpid()
    include_parent = False

  try:
    itself = psutil.Process(parent_pid)
  except psutil.NoSuchProcess:
    return

  children = itself.children(recursive=True)
  for child in children:
    if child.pid == skip_pid:
        continue
    try:
        child.kill()
    except psutil.NoSuchProcess:
        pass

  if include_parent:
    try:
      itself.kill()

      # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
      # so we send an additional signal to kill them.
      itself.send_signal(signal.SIGQUIT)
    except psutil.NoSuchProcess:
      pass


class OpenAI_APIServer_Watchdog:

  def __init__(self, base_url:str, health_check_interval = 5, server_process = None, server_thread = None):
    self.base_url = base_url
    self.watchdog_enabled = False
    self.watchdog_thread = None
    self.watchdog_stop_event = threading.Event()
    self.health_check_interval = health_check_interval  # seconds
    self.server_process = server_process
    self.server_thread = server_thread
    self.start_watchdog()

  def __del__(self, *exc):
    self.stop_watchdog()
  
  def is_server_healthy(self) -> bool:
    """
    Check if the server is healthy by monitoring both thread and process.
    Override this method to implement custom health checks.
    """
    try:
      
        if self.server_process and self.server_process.poll() is not None:
            logger.warning("Server process is not running")
            return False
        # Make HTTP health check
        health_url = f"{self.base_url}/health"
        try:
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            # Try alternative endpoints
            for endpoint in ["/v1/models", "/docs", "/"]:
                try:
                    url = f"{self.base_url}{endpoint}"
                    response = requests.get(url, timeout=5)
                    if response.status_code in [200, 404]:  # 404 is ok for some endpoints
                        return True
                except requests.RequestException:
                    continue
            return False
            
    except Exception as e:
        logger.error(f"Health check failed with an error: {e}")
        return False
        
  def _watchdog_loop(self):
    """Main watchdog monitoring loop."""
    logger.info(f"Watchdog started with {self.health_check_interval}s interval")
    
    while not self.watchdog_stop_event.is_set():
      try:
        #
        if not self.is_server_healthy():
          logger.warning("Server health check failed, attempting exit the program...")
          break
        else:
          logger.debug("Server health check passed")
              
      except Exception as e:
          logger.error(f"Watchdog error: {e}")
      
      # Wait for next check or stop event
      self.watchdog_stop_event.wait(self.health_check_interval)
    
    os.kill(os.getpid(), signal.SIGTERM)
    if self.server_process:
      kill_process_tree(self.server_process.pid)

  def start_watchdog(
    self, 
    health_check_interval: Optional[int] = None):
    """
    Start the watchdog to monitor server health.
    
    Args:
        health_check_interval: Seconds between health checks
        restart_callback: Function to call after successful restart
    """
    if self.watchdog_enabled:
        logger.warning("Watchdog is already running")
        return
        
    if health_check_interval:
        self.health_check_interval = health_check_interval
        
    self.watchdog_enabled = True
    self.watchdog_stop_event.clear()
    self.watchdog_thread = threading.Thread(
        target=self._watchdog_loop, 
        daemon=True,
        name="ServerWatchdog"
    )
    self.watchdog_thread.start()
    logger.info("Server watchdog started")

  def stop_watchdog(self):
    """Stop the watchdog."""
    if not self.watchdog_enabled:
        return
        
    logger.info("Stopping server watchdog...")
    self.watchdog_enabled = False
    self.watchdog_stop_event.set()
    
    if self.watchdog_thread and self.watchdog_thread.is_alive():
        self.watchdog_thread.join(timeout=5)
        
    logger.info("Server watchdog stopped")


class OpenAI_APIServer:

  def __init__(self, **kwargs):
    self.server_started_event = threading.Event()
    self.process = None
    self.backend = None
    self.server_thread = None

  def __del__(self, *exc):
    # This is important
    # close the server when exit the program
    logger.info("Killing the server.")
    self.close()

  def close(self):
    if self.process:
      logger.info(f"Kill process: {self.process.pid}")
      try:
        kill_process_tree(self.process.pid)
      except:
        self.process.kill()
        self.process.terminate()

  def wait_for_startup(self):
    self.server_started_event.wait()

  def validate_if_server_start(self, line: str):
    line_lower = line.lower()
    if self.backend in ["vllm", "sglang", "lmdeploy"]:
      if self.backend == "vllm":
        return "application startup complete" in line_lower or "vllm api server on" in line_lower
      else:
        return f" running on http://{self.host}:" in line.strip()
    elif self.backend == "llamacpp":
      return "waiting for new tasks" in line_lower
    elif self.backend == "tgi":
      return "Connected" in line.strip()

  def _start_server(self, cmds):
    try:
      env = os.environ.copy()
      env["VLLM_USAGE_SOURCE"] = "production-docker-image"
      self.process = subprocess.Popen(
          cmds,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          text=True,
          encoding="utf-8",
          errors="ignore"
      )
      for line in self.process.stdout:
        logger.info("Server Log:  " + line.strip())
        if self.validate_if_server_start(line):
          self.server_started_event.set()
          # break
    except Exception as e:
      if self.process:
        self.process.terminate()
      raise RuntimeError(f"Failed to start Server server: {e}")

  def start_server_thread(self, cmds: str):
    try:
      # Start the  server in a separate thread
      self.server_thread = threading.Thread(
          target=self._start_server, args=(cmds,), daemon=None)
      self.server_thread.start()

      # Wait for the server to start
      self.wait_for_startup()
    except Exception as e:
      raise Exception(e)

  @classmethod
  def from_lmdeploy_backend(
      cls,
      checkpoints: str,
      backend: str = "turbomind",
      cache_max_entry_count=0.9,
      tool_call_parser: str = None,
      reasoning_parser: str = None,
      tensor_parallel_size=1,
      max_prefill_token_num=4096,
      dtype='auto',
      quantization_format: str = None,
      quant_policy: int = 0,
      chat_template: str = None,
      max_batch_size=16,
      device="cuda",
      server_name="0.0.0.0",
      server_port=23333,
      additional_list_args: List[str] = []
  ):
    """Run lmdeploy OpenAI compatible server

    Args:
        checkpoints (str): model id or path
        backend (str, optional): turbomind or pytorch. Defaults to "turbomind".
        cache_max_entry_count (float, optional): reserved mem for cache. Defaults to 0.9.
        tool_call_parser (str, optional): see [doc](https://lmdeploy.readthedocs.io/en/v0.9.0/llm/api_server_tools.html).
        reasoning_parser (str, optional): see [doc](https://lmdeploy.readthedocs.io/en/v0.9.0/llm/api_server_reasoning.html).
        tensor_parallel_size (int, optional): n gpus. Defaults to 1.
        max_prefill_token_num (int, optional): prefill token, the higher the more GPU mems are used. Defaults to 4096.
        dtype (str, optional): dtype. Defaults to 'auto'.
        quantization_format (str, optional): quantization {awq, gptq}. Defaults to None.
        quant_policy (int, optional): KV cache quant policty {0, 4, 8} bits, 0 means not using quantization. Defaults to 0.
        chat_template (str, optional): Chat template. To see all chatempltes, run `lmdeploy list`. Defaults to None.
        max_batch_size (int, optional): batch size. Defaults to 16.
        device (str, optional): device. Defaults to "cuda".
        server_port (int, optional): port. Defaults to 23333.
        server_name (str, optional): host name. Defaults to "0.0.0.0".
        additional_list_args (List[str], optional): additional args to run subprocess cmd e.g. ["--arg-name", "arg value"]. See more at [github](https://github.com/InternLM/lmdeploy/blob/e8c8e7a019eb67430d7eeea74295813a6de0a780/lmdeploy/cli/serve.py#L83). Defaults to [].

    """
    # lmdeploy serve api_server $MODEL_DIR --backend $LMDEPLOY_BE --server-port 23333
    cmds = [
        PYTHON_EXEC,
        '-m',
        'lmdeploy',
        'serve',
        'api_server',
        checkpoints,
        '--dtype',
        str(dtype),
        '--backend',
        str(backend),
        '--tp',
        str(tensor_parallel_size),
        '--server-port',
        str(server_port),
        '--server-name',
        str(server_name),
        '--cache-max-entry-count',
        str(cache_max_entry_count),
        '--quant-policy',
        str(quant_policy),
        '--device',
        str(device),
    ]

    if quantization_format:
      cmds += ['--model-format', str(quantization_format)]

    if chat_template:
      cmds += ['--chat-template', str(chat_template)]

    if max_batch_size:
      cmds += ['--max-batch-size', str(max_batch_size)]

    if max_prefill_token_num:
      cmds += ['--max-prefill-token-num', str(max_prefill_token_num)]
    
    if tool_call_parser:
      cmds += ['--tool-call-parser', tool_call_parser,]
    if reasoning_parser:
      cmds += ['--reasoning-parser', reasoning_parser,]
    
    cmds += additional_list_args
    logger.info("CMDS to run `lmdeploy` server: " + " ".join(cmds))

    _self = cls()

    _self.host = server_name
    _self.port = server_port
    _self.backend = "lmdeploy"
    _self.start_server_thread(cmds)

    return _self

  @classmethod
  def from_vllm_backend(
      cls,
      checkpoints,
      limit_mm_per_prompt: str = '',
      max_model_len: float = None,
      gpu_memory_utilization: float = 0.9,
      max_num_batched_tokens: int = 8192,
      enable_chunked_prefill: bool = True,
      reasoning_parser: str = None,
      enable_auto_tool_choice: bool = False,
      tool_call_parser: str = None,
      dtype="auto",
      task="auto",
      kv_cache_dtype: str = "auto",
      tensor_parallel_size=1,
      chat_template: str = None,
      cpu_offload_gb: float = 0.,
      quantization: str = None,
      port=23333,
      host="localhost",
      additional_list_args: List[str] = []
  ):
    """Run VLLM OpenAI compatible server

    Args:
      checkpoints (str): model id or path
      limit_mm_per_prompt (str, optional): For each multimodal plugin, limit how many input instances to allow for each prompt. Expects a comma-separated list of items, e.g.: image=16,video=2 allows a maximum of 16 images and 2 videos per prompt. Defaults to 1 for each modality.
      max_model_len (float, optional):Model context length. If unspecified, will be automatically derived from the model config. Defaults to None.
      gpu_memory_utilization (float, optional): The fraction of GPU memory to be used for the model executor, which can range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory utilization. If unspecified, will use the default value of 0.9. This is a per-instance limit, and only applies to the current vLLM instance.It does not matter if you have another vLLM instance running on the same GPU. For example, if you have two vLLM instances running on the same GPU, you can set the GPU memory utilization to 0.5 for each instance. Defaults to 0.9.
      enable_chunked_prefill (bool, optional): Enable chunk prefill.
      max_num_batched_tokens (int, optional): Prefill tokens budeget, use it to tune performance of model. Higher value increase TTFT, lower value (<2048) decreases ITL. For optimal throughput, set it > 8096. see more [doc](https://docs.vllm.ai/en/latest/configuration/optimization.html#chunked-prefill_1).
      reasoning_parser (str, optional): choose parser for reasoning, see [doc](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html)
      enable_auto_tool_choice (bool, optional): mandatory Auto tool choice. tells vLLM that you want to enable the model to generate its own tool calls when it deems appropriate, see [doc](https://docs.vllm.ai/en/latest/features/tool_calling.html)
      tool_call_parser (str, optional): select the tool parser to use. Additional tool parsers will continue to be added in the future, and also can register your own tool parsers in the --tool-parser-plugin, see [doc](https://docs.vllm.ai/en/latest/features/tool_calling.html)
      dtype (str, optional): dtype. Defaults to "float16".
      task (str, optional): The task to use the model for. Each vLLM instance only supports one task, even if the same model can be used for multiple tasks. When the model only supports one task, "auto" can be used to select it; otherwise, you must specify explicitly which task to use. Choices {auto, generate, embedding, embed, classify, score, reward, transcription}. Defaults to "auto".
      kv_cache_dtype (str, optional): Data type for kv cache storage. If “auto”, will use model data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports fp8 (=fp8_e4m3). Defaults to "auto".
      tensor_parallel_size (int, optional): n gpus. Defaults to 1.
      chat_template (str, optional): The file path to the chat template, or the template in single-line form for the specified model. Defaults to None.
      cpu_offload_gb (float, optional): The space in GiB to offload to CPU, per GPU. Default is 0, which means no offloading. Intuitively, this argument can be seen as a virtual way to increase the GPU memory size. For example, if you have one 24 GB GPU and set this to 10, virtually you can think of it as a 34 GB GPU. Then you can load a 13B model with BF16 weight, which requires at least 26GB GPU memory. Note that this requires fast CPU-GPU interconnect, as part of the model is loaded from CPU memory to GPU memory on the fly in each model forward pass. Defaults to 0.
      quantization (str, optional): quantization format {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,hqq,experts_int8,neuron_quant,ipex,quark,moe_wna16,None}. Defaults to None.
      port (int, optional): port. Defaults to 23333.
      host (str, optional): host name. Defaults to "localhost".
      additional_list_args (List[str], optional): additional args to run subprocess cmd e.g. ["--arg-name", "arg value"]. See more at [this document](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#vllm-serve). Defaults to [].

    """
    cmds = [
        PYTHON_EXEC,
        '-m',
        'vllm.entrypoints.openai.api_server',
        '--model',
        checkpoints,
        '--dtype',
        str(dtype),
        '--task',
        str(task),
        '--kv-cache-dtype',
        str(kv_cache_dtype),
        '--tensor-parallel-size',
        str(tensor_parallel_size),
        '--gpu-memory-utilization',
        str(gpu_memory_utilization),
        '--cpu-offload-gb',
        str(cpu_offload_gb),
        '--port',
        str(port),
        '--host',
        str(host),
        "--trust-remote-code"
    ]

    if quantization:
      cmds += ['--quantization', str(quantization),]
    if chat_template:
      cmds += ['--chat-template', str(chat_template),]
    if max_model_len:
      cmds += ['--max-model-len', str(max_model_len),]
    if limit_mm_per_prompt:
      cmds += ['--limit-mm-per-prompt', str(limit_mm_per_prompt),]
    if reasoning_parser:
      cmds += ['--reasoning-parser', str(reasoning_parser),]
      
    if enable_auto_tool_choice or tool_call_parser:
      assert enable_auto_tool_choice and tool_call_parser, ValueError(f"You must set both enable_auto_tool_choice and tool_call_parser to enable tool calling, got {tool_call_parser=} and {enable_auto_tool_choice=}")
      cmds += ['--enable-auto-tool-choice']
      cmds += ['--tool-call-parser', tool_call_parser]
      
    if enable_chunked_prefill:
      cmds += ["--enable-chunked-prefill"]
      
    if max_num_batched_tokens:
      cmds += ["--max-num-batched-tokens", str(max_num_batched_tokens)]
      

    if additional_list_args != []:
      cmds += additional_list_args

    logger.info("CMDS to run vllm server: \n" + ' '.join(cmds))

    _self = cls()

    _self.host = host
    _self.port = port
    _self.backend = "vllm"
    _self.start_server_thread(cmds)
    import time
    time.sleep(5)

    return _self

  @classmethod
  def from_sglang_backend(
      cls,
      checkpoints,
      mem_fraction_static: float = 0.8,
      tool_call_parser: str = None,
      reasoning_parser: str = None,
      context_length: str = None,
      chat_template: str = None,
      dtype: str = "auto",
      kv_cache_dtype: str = "auto",
      tp_size: int = 1,
      quantization: str = None,
      load_format: str = "auto",
      device: str = "cuda",
      port=23333,
      host="0.0.0.0",
      max_running_requests: int = None,
      max_total_tokens: int = None,
      max_prefill_tokens: int = None,

      schedule_policy: str = "fcfs",
      schedule_conservativeness: float = 1.0,
      cpu_offload_gb: int = 0,

      additional_list_args: List[str] = [],
  ):
    """Start SGlang OpenAI compatible server.

    Args:
        checkpoints (str): model id or path.
        dtype (str, optional): Dtype used for the model {"auto", "half", "float16", "bfloat16", "float", "float32"}. Defaults to "auto".
        mem_fraction_static (float, optional): The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors. Defaults to 0.8.
        tool_call_parser (str, optional): see [doc](https://docs.sglang.ai/backend/function_calling.html).
        reasoning_parser (str, optional): see [doc](https://docs.sglang.ai/backend/separate_reasoning.html).
        chat_template (str, optional): The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.. Defaults to None.
        context_length (str, optional): The model's maximum context length. Defaults to None (will use the value from the model's config.json instead). Defaults to None.
        kv_cache_dtype (str, optional): Dtype of the kv cache, defaults to the dtype. Defaults to "auto".
        tp_size (int, optional): The number of GPUs the model weights get sharded over. Mainly for saving memory rather than for high throughput. Defaults to 1.
        quantization (str, optional): Quantization format {"awq","fp8","gptq","marlin","gptq_marlin","awq_marlin","bitsandbytes","gguf","modelopt","w8a8_int8"}. Defaults to None.
        load_format (str, optional): The format of the model weights to load:\n* `auto`: will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available.\n* `pt`: will load the weights in the pytorch bin format. \n* `safetensors`: will load the weights in the safetensors format. \n* `npcache`: will load the weights in pytorch format and store a numpy cache to speed up the loading. \n* `dummy`: will initialize the weights with random values, which is mainly for profiling.\n* `gguf`: will load the weights in the gguf format. \n* `bitsandbytes`: will load the weights using bitsandbytes quantization."\n* `layered`: loads weights layer by layer so that one can quantize a layer before loading another to make the peak memory envelope smaller.\n. Defaults to "auto".\n
        device (str, optional): The device type {"cuda", "xpu", "hpu", "cpu"}. Defaults to "cuda".
        port (int, optional): Port number. Defaults to 23333.
        host (str, optional): Host name. Defaults to "0.0.0.0".
        max_running_requests (int, optional): The maximum number of running requests.. Defaults to None.
        max_total_tokens (int, optional): The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. This option is typically used for development and debugging purposes.. Defaults to None.
        max_prefill_tokens (int, optional): The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length. Defaults to None.
        schedule_policy (str, optional): The scheduling policy of the requests {"lpm", "random", "fcfs", "dfs-weight"}. Defaults to "fcfs".
        schedule_conservativeness (float, optional): How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently. Defaults to 1.0.
        cpu_offload_gb (int, optional): How many GBs of RAM to reserve for CPU offloading. Defaults to 0.
        additional_list_args (List[str], optional): additional args to run subprocess cmd e.g. ["--arg-name", "arg value"]. See more at [doc](https://docs.sglang.ai/backend/server_arguments.html). Defaults to [].

    Returns:
        _type_: _description_
    """
    from sglang.utils import wait_for_server, execute_shell_command

    cmds = [
        PYTHON_EXEC,
        '-m',
        'sglang.launch_server',
        '--model-path',
        checkpoints,
        '--dtype',
        str(dtype),
        '--device',
        str(device),
        '--kv-cache-dtype',
        str(kv_cache_dtype),
        '--tp-size',
        str(tp_size),
        '--load-format',
        str(load_format),
        '--mem-fraction-static',
        str(mem_fraction_static),
        '--schedule-policy',
        str(schedule_policy),
        '--schedule-conservativeness',
        str(schedule_conservativeness),
        '--cpu-offload-gb',
        str(cpu_offload_gb),
        '--port',
        str(port),
        '--host',
        host,
        "--trust-remote-code"
    ]
    if chat_template:
      cmds += [
          "--chat-template", chat_template
      ]
    if quantization:
      cmds += ['--quantization', quantization,]
    if context_length:
      cmds += ['--context-length', context_length,]
    if max_running_requests:
      cmds += ['--max-running-requests', max_running_requests,]
    if max_total_tokens:
      cmds += ['--max-total-tokens', max_total_tokens,]
    if max_prefill_tokens:
      cmds += ['--max-prefill-tokens', max_prefill_tokens,]
    if tool_call_parser:
      cmds += ['--tool-call-parser', tool_call_parser,]
    if reasoning_parser:
      cmds += ['--reasoning-parser', reasoning_parser,]

    if additional_list_args:
      cmds += additional_list_args

    logger.info("CMDS to run `sglang` server: \n" + " ".join(cmds))
    _self = cls()

    _self.host = host
    _self.port = port
    _self.backend = "sglang"
    # _self.start_server_thread(cmds)
    # new_path = os.environ["PATH"] + ":/sbin"
    # _self.process = subprocess.Popen(cmds, text=True, stderr=subprocess.STDOUT, env={**os.environ, "PATH": new_path})
    _self.process = execute_shell_command(" ".join(cmds))

    logger.info("Waiting for " + f"http://{_self.host}:{_self.port}")
    wait_for_server(f"http://{_self.host}:{_self.port}")
    logger.info("Done")

    return _self

  @classmethod
  def from_llamacpp_backend(
      cls,
      checkpoints,
      model,
      chat_template: str = None,
      n_gpu_layers: int = 0,
      ctx_size: int = 4096,
      mmproj: str = None,
      batch_size: int = 2048,
      main_gpu: int = 0,
      tensor_split: float = None,
      ubatch_size: int = 512,
      threads: int = None,
      threads_batch: int = None,
      numa: str = None,
      flash_attn: bool = False,
      no_perf: bool = True,
      port=23333,
      host="localhost",
      cache_type_k: str = "f16",
      cache_type_v: str = "f16",
      no_kv_offload: bool = False,
      parallel: int = 1,
      split_mode: str = "layer",
      additional_list_args: List[str] = []
  ):
    """Start Llamacpp OpenAI server

    Args:
        checkpoints (str): model id or path.
        model (str): GGUF file name.
        chat_template (str, optional): Chat template. Defaults to "chatml".
        n_gpu_layers (int, optional): The number of layers to put on GPU. The rest will be on CPU. Defaults to 0.
        tensor_split (float, optional): fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1. Defaults to None.
        ctx_size (int, optional): size of the prompt context (default: 4096, 0 = loaded from model). Defaults to 4096.
        mmproj (str, optional): additional multimodal projector checkpoint file, see [doc](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
        main_gpu (int, optional): main_gpu interpretation depends on. Defaults to 0.
        batch_size (int, optional): logical maximum batch size. Defaults to 2048.
        ubatch_size (int, optional): physical maximum batch size. Defaults to 512.
        threads (int, optional): number of threads to use during generation. Defaults to None.
        threads_batch (int, optional): umber of threads to use during batch and prompt processing. Defaults to None.
        numa (str, optional): _description_. Defaults to None.
        split_mode (str, optional): how to split the model across multiple GPUs, one of: `none`: use one GPU only, `layer`: split layers and KV across GPUs, `row`: split rows across GPUs. Defaults to `layer`,
        flash_attn (bool, optional): enable Flash Attention. Defaults to False.
        no_perf (bool, optional): disable internal libllama performance timings. Defaults to True.
        port (int, optional): Serving port. Defaults to 23333.
        host (str, optional): Serving host name. Defaults to "localhost".
        cache_type_k (str, optional): KV cache data type for K. Allowed values: {f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1}. Defaults to "f16".
        cache_type_v (str, optional): KV cache data type for V. Allowed values: {f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1}. Defaults to "f16".
        no_kv_offload (bool, optional): disable KV offload. Defaults to False.
        parallel (int, optional): number of parallel sequences to decode. Defaults to 1.
        additional_list_args (List[str], optional): additional args to run subprocess cmd e.g. ["--arg-name", "arg value"]. See more at [github](https://github.com/ggml-org/llama.cpp/tree/master/tools/server). Defaults to [].

    Returns:
        OpenAI_APISever
    """
    from huggingface_hub import hf_hub_download

    checkpoint_path = hf_hub_download(
        repo_id=checkpoints, filename=model)

    if mmproj:
      mmproj = hf_hub_download(
        repo_id=checkpoints, filename=mmproj)
    
    cmds = [
        'llama-server',

        '--model',
        str(checkpoint_path),

        # '--hf-repo',
        # str(checkpoints),

        '--n-gpu-layers',
        str(n_gpu_layers),

        '--port',
        str(port),

        '--host',
        str(host),

        '--main_gpu',
        str(main_gpu),

        '--ctx-size',
        str(ctx_size),

        '--batch-size',
        str(batch_size),

        '--ubatch-size',
        str(ubatch_size),

        '--cache-type-k',
        str(cache_type_k),

        '--cache-type-v',
        str(cache_type_v),

        '--parallel',
        str(parallel),

        '--split-mode',
        str(split_mode),

        "--verbose",
        "--no-webui",

    ]

    if tensor_split:
      cmds += ['--tensor_split', str(tensor_split),]

    if threads_batch:
      cmds += ['--threads-batch', str(threads_batch),]

    if threads:
      cmds += ['--threads', str(threads),]

    if numa:
      cmds += ['--numa', str(numa)]

    if flash_attn is not None:
      cmds += ['--flash-attn']

    if no_perf is not None:
      cmds += ['--no-perf']

    if chat_template is not None:
      cmds += ['--chat-template', str(chat_template),]

    if no_kv_offload is not None:
      cmds += ['--no-kv-offload']
    
    if mmproj is not None:
      cmds += ['--mmproj', str(mmproj)]

    if additional_list_args != []:
      cmds += additional_list_args

    logger.info("CMDS to run `llamacpp` server: \n" + " ".join(cmds))

    _self = cls()

    _self.host = host
    _self.port = port
    _self.backend = "llamacpp"
    _self.start_server_thread(cmds)
    import time
    time.sleep(5)

    return _self

  @classmethod
  def from_tgi_backend(
      cls,
      checkpoints,
      port=23333,
      hostname="0.0.0.0",
      hf_token=None,
      additional_list_args: List[str] = []
  ):
    """Start TGI OpenAI server

    Args:
        checkpoints (str): model id or path.
        port(int, optional): Serving port. Defaults to 23333.
        host(str, optional): Serving host name. Defaults to "localhost".
        additional_list_args (List[str], optional): additional args to run subprocess cmd e.g. ["--arg-name", "arg value"]. See more at [github](https://huggingface.co/docs/text-generation-inference/reference/launcher). Defaults to [].

    Returns:
        OpenAI_APISever
    """
    if hf_token:
      os.environ["HF_TOKEN"] = str(hf_token)
    os.environ["RUST_BACKTRACE"] = "full"
    os.environ["METRICS_ADDRESS"] = "localhost:34388"

    cmds = [
        'text-generation-launcher',

        '--model-id',
        str(checkpoints),

        '--port',
        str(port),

        '--hostname',
        str(hostname),

        "--trust-remote-code",

        "--usage-stats",
        "off",

        "-e"

    ]

    if additional_list_args != []:
      cmds += additional_list_args

    logger.info("CMDS to run `tgi` server: \n" + " ".join(cmds))

    _self = cls()

    _self.host = hostname
    _self.port = port
    _self.backend = "tgi"
    _self.start_server_thread(cmds)
    import time
    time.sleep(5)

    return _self


def analyze_multiple_streams(
    stream_iterators,
):
    """
    Analyze multiple streaming responses to compute average metrics:
    - Time to first token
    - Throughput (output/blended tokens/sec)
    - End-to-end latency
    - Requests per minute

    Args:
        stream_iterators: list of iterators yielding streaming chunks
    Returns:
        dict with average metrics
    """
    all_metrics = []

    for idx, iterator in enumerate(stream_iterators):
        start_time = time.time()
        first_token_time = None
        output_tokens = 0
        for chunk in iterator:
            current_time = time.time()
            if first_token_time is None:
                first_token_time = current_time
            # Determine text content
            if isinstance(chunk, str):
                output_tokens += 1
        end_time = time.time()
        time_to_first_token = (
            first_token_time - start_time if first_token_time else None
        )
        e2e_latency = end_time - start_time
        output_throughput = (
            output_tokens / e2e_latency if e2e_latency > 0 else 0
        )
        requests_per_minute = (
            60 / e2e_latency if e2e_latency > 0 else 0
        )
        all_metrics.append({
            "time_to_first_token": time_to_first_token,
            "e2e_latency": e2e_latency,
            "output_tokens": output_tokens,
            "output_throughput": output_throughput,
            "requests_per_minute": requests_per_minute,
        })

    # Compute averages
    def avg(key):
        vals = [m[key] for m in all_metrics if m[key] is not None]
        return sum(vals) / len(vals) if vals else None

    averaged_metrics = {
        "avg_time_to_first_token": avg("time_to_first_token"),
        "avg_e2e_latency": avg("e2e_latency"),
        "avg_output_tokens": avg("output_tokens"),
        "avg_output_throughput": avg("output_throughput"),
        "avg_requests_per_minute": avg("requests_per_minute"),
        "all_metrics": all_metrics
    }

    logger.info("\n=== Average Metrics Across Requests ===")
    logger.info(f"Average Time to First Token: {averaged_metrics['avg_time_to_first_token']:.4f} sec")
    #logger.info(f"Average End-to-End Latency: {averaged_metrics['avg_e2e_latency']:.4f} sec")
    logger.info(f"Average Output Tokens: {averaged_metrics['avg_output_tokens']:.2f}")
    logger.info(f"Average Output Throughput: {averaged_metrics['avg_output_throughput']:.2f} tokens/sec")
    logger.info(f"Average Requests per Minute: {averaged_metrics['avg_requests_per_minute']:.2f}")

    return averaged_metrics
  
