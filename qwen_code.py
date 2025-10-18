import subprocess
import sys
import json
import re
import os
import socket
import time
import gc
from typing import List, Dict, Any, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# also try limiting the cache size (e.g., to 512MB) to force more frequent returns to the OS.
"""max_split_size_mb:512: This parameter sets the maximum size (in megabytes) of a free memory block that the allocator will split 
when a smaller allocation is requested. If a free block is larger than 512MB, and a smaller allocation is needed, 
the allocator will split the block, using only the necessary portion and keeping the remainder available as a smaller free block. 
This helps to prevent large contiguous blocks of memory from being entirely consumed by small allocations, 
thus improving the chances of finding large contiguous blocks for future large allocations."""

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"


def check_internet():
    """Check if internet connection is available."""
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5)
        return True
    except (socket.timeout, socket.error, OSError):
        return False

def find_local_model(model_name):
    """Find cached model in common HuggingFace cache locations."""
    # Check local directory first
    local_name = model_name.split('/')[-1]
    if os.path.exists(local_name) and validate_model_files(local_name):
        return local_name
    
    # Check HuggingFace cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    model_path = os.path.join(cache_dir, model_dir_name)
    
    if os.path.exists(model_path):
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            for snapshot in snapshots:
                snapshot_path = os.path.join(snapshots_dir, snapshot)
                if validate_model_files(snapshot_path):
                    return snapshot_path
    
    return None

def validate_model_files(model_path):
    """Check if model directory has required files."""
    if not os.path.exists(model_path):
        return False
    
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    model_files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors'))]
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            return False
    
    return len(model_files) > 0

def load_model(model_name="Qwen/Qwen2.5-Coder-7B-Instruct", force_offline=False):
    """Load model and tokenizer with offline support."""
    #check if we have the model downloaded already
    print(f"Checking for model {model_name}...")
    _ = download_model(model_name)  # Attempt to download if not present

    # Check if we should use online or offline mode
    if force_offline or not check_internet():
        print("Using offline mode...")
        local_path = find_local_model(model_name)
        if not local_path:
            raise FileNotFoundError(
                f"Model {model_name} not found locally. "
                f"Please run with internet connection first to download the model."
            )
        
        print(f"Loading model from: {local_path}")
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            dtype="auto",
            device_map="auto", # This instructs PyTorch to use all available GPUs
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            local_files_only=True
        )
    else:
        print(f"Loading {model_name} (will download if needed)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def download_model(model_name="Qwen/Qwen2.5-Coder-7B-Instruct"):
    """Download model for offline use."""
    import multiprocessing
    print(f"Ensuring model {model_name} is downloaded...")
    #do this to put in a separate process to avoid memory bloat in the main process
    #will use more total memory but avoid fragmentation in the main process
    #auto deletes when process ends
    download_process = multiprocessing.Process(target=download_thread, args=(model_name,))
    download_process.start()
    
    # Wait for the process to complete
    download_process.join()
    return True

def download_thread(model_name):
    """Thread target for downloading model."""
    #auto downloads the model if we don't have it.
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return  # just download and cache in huggingface cache
  
class SimpleQwen:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-7B-Instruct", force_offline=False):
        self.model, self.tokenizer = load_model(model_name, force_offline)
        self.torch = torch#__import__("torch")  # Lazy import to avoid unnecessary memory usage if not needed
        self.messages = [{"role": "system", "content": "You are a helpful developer coding assistant."}]
        self.tools = {}
        self.available_tools = []

        # Get the data type of the first parameter in the model
        first_param_dtype = next(self.model.parameters()).dtype

        # Print the result
        print(f"The model's data type is: {first_param_dtype}")

        # Ensure model is in evaluation mode
        self.model.eval()
    
    def register_tool(self, func: Callable, name: str = None, description: str = None):
        """Register a tool function."""
        if name is None:
            name = func.__name__
        
        self.tools[name] = func
        
        # Simple tool definition
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or func.__doc__ or f"Function {name}",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        # Update tools list
        self.available_tools = [t for t in self.available_tools if t["function"]["name"] != name]
        self.available_tools.append(tool_def)
    
    def _parse_tool_calls(self, content: str) -> Dict[str, Any]:
        """Parse tool calls from model output."""
        tool_calls = []
        
        # Look for tool call blocks: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
        for match in re.finditer(r"<tool_call>\n(.+?)\n</tool_call>", content, re.DOTALL):
            try:
                tool_call_json = json.loads(match.group(1).strip())
                tool_calls.append({
                    "type": "function", 
                    "function": tool_call_json
                })
            except json.JSONDecodeError:
                continue
        
        if tool_calls:
            # Extract content before tool calls
            offset = content.find("<tool_call>")
            content_text = content[:offset].strip() if offset > 0 else ""
            return {
                "role": "assistant",
                "content": content_text,
                "tool_calls": tool_calls
            }
        else:
            return {
                "role": "assistant",
                "content": content.strip()
            }
    
    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls and return results."""
        tool_results = []
        
        for tool_call in tool_calls:
            function_call = tool_call.get("function")
            if function_call:
                function_name = function_call["name"]
                function_args = function_call.get("arguments", {})
                
                if function_name in self.tools:
                    try:
                        result = self.tools[function_name](function_args)
                        tool_results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(result) if not isinstance(result, str) else result
                        })
                    except Exception as e:
                        tool_results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error: {str(e)}"
                        })
                else:
                    tool_results.append({
                        "role": "tool",
                        "name": function_name,
                        "content": f"Function {function_name} not found"
                    })
        
        return tool_results
    
    def _run_generation_and_cleanup(self, text: str, max_new_tokens: int = 4500):
        """Helper function to run generation, decode, and aggressively clean memory."""
        
        # Use torch.no_grad() for inference to save memory used by backprop structures
        with self.torch.no_grad():
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_ids_len = model_inputs.input_ids.shape[1]

            # START timing the generation
            start_time = time.time()

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8
            )
            generation_time = time.time() - start_time

        # Explicitly delete input tensors before cleanup
        del model_inputs
        
        # Extract new tokens
        output_tokens = [
            output_ids[input_ids_len:] for output_ids in generated_ids
        ]
        output_tokens_count = len(output_tokens[0])
        tps = output_tokens_count / generation_time
        print(f"[Timing Check] Generated {output_tokens_count} tokens in {generation_time:.3f}s. Observed GPU TPS: {tps:.2f}")
        
        response_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        
        # Explicitly delete generated tensor after decoding
        del generated_ids
        del output_tokens
        
        # Aggressive memory cleanup *after* inference and tensor deletion
        gc.collect()
        
        return response_text
    
    def token_count(self, text: str) -> int:
        """Count tokens in text."""
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def chat(self, user_input: str, file_contents: list = None) -> str:
        """Generate response with tool support."""
        # Add user message
        self.messages.append({"role": "user", "content": user_input})

        # Copy messages to avoid modifying original during processing, file contents are dynamically added
        #user could add or delete files between calls so we don't want to cache them in self.messages since we can't track the index easily
        messages_copy = self.messages.copy()

        #Insert file contents if provided
        if file_contents and len(file_contents) > 0:
            for file in file_contents:
                content = file.get("content", "")
                filename = file.get("filename", "unknown")
                messages_copy.insert(1, {  # Insert after system prompt
                    "role": "user",
                    "content": f"File '{filename}' has been loaded. Here is its content:\n ### START### \n{content}\n### END ###\n"
                })
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages_copy,
            tools=self.available_tools if self.available_tools else None,
            tokenize=False,
            add_generation_prompt=True
        )
        
        #First Generation Attempt
        response_text = self._run_generation_and_cleanup(text)
        
        # Parse response for tool calls
        parsed_response = self._parse_tool_calls(response_text)
        self.messages.append(parsed_response)
        
        # Execute tools if present
        if tool_calls := parsed_response.get("tool_calls"):
            tool_results = self._execute_tool_calls(tool_calls)

            self.messages.extend(tool_results)
            
            # Generate final response after tool execution
            text = self.tokenizer.apply_chat_template(
                self.messages,
                tools=self.available_tools if self.available_tools else None,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 2. Second Generation Attempt ( cleanup applied)
            final_response = self._run_generation_and_cleanup(text)

            final_parsed = self._parse_tool_calls(final_response)
            self.messages.append(final_parsed)
            
            return final_parsed["content"]
        
        return parsed_response["content"]
  
# Example usage
if __name__ == "__main__":
    import time
    
    # Limit to specific GPU because this is a 7B model and fits on one 16GB GPU. Adjust as needed.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Initialize chat, Load model (will download if not present)
    chat = SimpleQwen(model_name="Qwen/Qwen2.5-Coder-7B-Instruct",force_offline=True)
    
    # Example tool
    def get_weather(args):
        """Get weather information for a location."""
        location = args.get("location", "unknown")
        return f"The weather in {location} is sunny and 75°F"
    
    # Register tool
    chat.register_tool(get_weather, description="Get current weather for a location")
    
    # Chat loop
    print("Chat started! Type 'quit' or 'exit' to end.")
   
    
    while True:
        user_input = input("\nYou: ").strip()
        
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        if user_input:
            loop_start = time.time()  # Start timing the loop
            response = chat.chat(user_input)
            print(f"Assistant: {response}")
            loop_end = time.time()
            elapsed = loop_end - loop_start
            print(f"response {elapsed:.2f} seconds")