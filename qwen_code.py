import subprocess
import sys
import json
import re
import os
import socket
import time
import gc
from typing import List, Dict, Any, Callable


# also try limiting the cache size (e.g., to 512MB) to force more frequent returns to the OS.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

def install_dependencies():
    """Install required packages if not available.
    LEGACY: This is a legacy function, prefer using requirements.txt and a virtual environment."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error dependencies missing, run requirements.txt ...")
        
    return torch, AutoModelForCausalLM, AutoTokenizer

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
    torch, AutoModelForCausalLM, AutoTokenizer = install_dependencies()
    
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
            device_map="auto",
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
    
    return model, tokenizer, torch

class SimpleQwen:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-7B-Instruct", force_offline=False):
        self.model, self.tokenizer, self.torch = load_model(model_name, force_offline)
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self.tools = {}
        self.available_tools = []

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
    
    def _run_generation_and_cleanup(self, text: str, max_new_tokens: int = 8000):
        """Helper function to run generation, decode, and aggressively clean memory."""
        
        # Clear unused cached memory *before* allocating inputs and running inference
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            gc.collect()
            # No need for time.sleep(1) if we are immediately proceeding with the load/run

        # Use torch.no_grad() for inference to save memory used by backprop structures
        with self.torch.no_grad():
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_ids_len = model_inputs.input_ids.shape[1]

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8
            )

        # Explicitly delete input tensors before cleanup
        del model_inputs
        
        # Extract new tokens
        output_tokens = [
            output_ids[input_ids_len:] for output_ids in generated_ids
        ]
        
        response_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        
        # Explicitly delete generated tensor after decoding
        del generated_ids
        del output_tokens
        
        # Aggressive memory cleanup *after* inference and tensor deletion
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            gc.collect()
        
        return response_text
    
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
        
        # 1. First Generation Attempt
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
            
            # 2. Second Generation Attempt (Crucial cleanup applied)
            final_response = self._run_generation_and_cleanup(text)

            final_parsed = self._parse_tool_calls(final_response)
            self.messages.append(final_parsed)
            
            return final_parsed["content"]
        
        return parsed_response["content"]

def download_model(model_name="Qwen/Qwen2.5-Coder-7B-Instruct"):
    """Download model for offline use."""
    save_path = f"./{model_name.split('/')[-1]}"
    
    print(f"Downloading {model_name} for offline use...")
    torch, AutoModelForCausalLM, AutoTokenizer = install_dependencies()
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"Model downloaded to: {save_path}")

# Example usage
if __name__ == "__main__":
    import time
    # Handle command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-Coder-7B-Instruct"
        download_model(model_name)
        sys.exit()
    
    force_offline = len(sys.argv) > 1 and sys.argv[1] == "offline"
    
    # Initialize chat
    chat = SimpleQwen(force_offline=force_offline)
    
    # Example tool
    def get_weather(args):
        """Get weather information for a location."""
        location = args.get("location", "unknown")
        return f"The weather in {location} is sunny and 75°F"
    
    # Register tool
    chat.register_tool(get_weather, description="Get current weather for a location")
    
    # Chat loop
    print("Chat started! Type 'quit' to exit.")
    print("Commands:")
    print("  python script.py download - Download model for offline use")
    print("  python script.py offline - Force offline mode")
    
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