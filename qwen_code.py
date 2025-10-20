import json
import re
import os
import socket
import time
import gc
import random
import string
from typing import List, Dict, Any, Callable, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
logging.basicConfig(level=logging.DEBUG)

# also try limiting the cache size (e.g., to 512MB) to force more frequent returns to the OS.
"""max_split_size_mb:256: This parameter sets the maximum size (in megabytes) of a free memory block that the allocator will split 
when a smaller allocation is requested. If a free block is larger than 512MB, and a smaller allocation is needed, 
the allocator will split the block, using only the necessary portion and keeping the remainder available as a smaller free block. 
This helps to prevent large contiguous blocks of memory from being entirely consumed by small allocations, 
thus improving the chances of finding large contiguous blocks for future large allocations."""

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"


class SimpleQwen:
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-7B-Instruct"):
        self.model_name = model_name
        self.local_files_only = self.check_internet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                                       local_files_only=self.local_files_only)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          dtype="auto",
                                                          device_map="auto", 
                                                          local_files_only=self.local_files_only)
        self.model.eval()
        self.messages = [{"role": "system", "content": "You are a helpful developer coding assistant.When calling tools, always use this exact format: <tool_call>{'name': '...', 'arguments': {...}}</tool_call>"}]
        self.tools = {}
        self.available_tools = []
        self.streaming_tool_break_flag = "BREAK_HERE_TOOLS_WERE_USED"
        first_param_dtype = next(self.model.parameters()).dtype
        logging.debug(f"The model's data type is: {first_param_dtype}")
        self.model.eval()

    def check_internet(self):
        """Check if internet connection is available."""
        try:
            socket.create_connection(("huggingface.co", 443), timeout=5)
            return True
        except (socket.timeout, socket.error, OSError):
            return False

    def register_tool(self, func: Callable, name: str = None, description: str = None):
        """Register a tool function."""
        if name is None:
            name = func.__name__
        
        self.tools[name] = func
        
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
        
        self.available_tools = [t for t in self.available_tools if t["function"]["name"] != name]
        self.available_tools.append(tool_def)
    
    def _generate_tool_call_id(self):
        """Generate a unique tool call ID."""
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(9))
    
    def _parse_tool_calls(self, content: str) -> Dict[str, Any]:
        """Parse tool calls from model output - handles multiple tools and various formats.
        Qwen 7B was NOT consistent with tool call formats so have 3 observed patterns for parsing"""
        tool_calls = []
        logging.debug(f"TOOL ARGS: {content}")
        # First, strip markdown code blocks if present (```xml ... ``` or ```json ... ```)
        # This handles cases where the model wraps tool calls in code blocks
        content = re.sub(r'```(?:xml|json)?\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'```\s*', '', content)
        
        # Pattern 1: <tools>, <tool>, <response>, <function-calls>, <function-call>, <functions>, or <function> with JSON content
        # Matches: <tools>...</tools>, <response>...</response>, <function-calls>...</function-calls>, <function-call>...</function-call> etc.
        block_pattern = r"<(?:tools?|tool_calls?|response?|xml?|functions?|function-call?|function-calls?)>\s*(.+?)\s*</(?:tools?|tool_calls?|response?|xml?|functions?|function-call?|function-calls?)>"
        for match in re.finditer(block_pattern, content, re.DOTALL | re.IGNORECASE):
            tools_block = match.group(1).strip()
            
            # Try to parse the entire block as a single JSON object first
            try:
                tool_call_json = json.loads(tools_block)
                logging.debug(f"Tool Call Json (single): {tool_call_json}")
                
                # Create tool call with ID
                tool_calls.append({
                    "id": self._generate_tool_call_id(),
                    "type": "function",
                    "function": {
                        "name": tool_call_json["name"],
                        "arguments": tool_call_json.get("arguments", {})
                    }
                })
            except json.JSONDecodeError:
                # If that fails, try parsing line by line (original behavior)
                for line in tools_block.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        tool_call_json = json.loads(line)
                        logging.debug(f"Tool Call Json (line): {tool_call_json}")
                        
                        # Create tool call with ID
                        tool_calls.append({
                            "id": self._generate_tool_call_id(),
                            "type": "function",
                            "function": {
                                "name": tool_call_json["name"],
                                "arguments": tool_call_json.get("arguments", {})
                            }
                        })
                    except json.JSONDecodeError as e:
                        logging.debug(f"Error parsing tool call: {e}")
                        continue
        
        #If we found tools already don't run:
        if not tool_calls:
            # Pattern 2: Self-closing XML tags like <function name="..." arguments='...' />
            # Matches: <function name="get_weather" arguments='{"location": "boston"}' />
            xml_pattern = r'<(?:function|function-call|tool)\s+name=["\']([^"\']+)["\']\s+arguments=["\']([^"\']+)["\']\s*/>'
            for match in re.finditer(xml_pattern, content, re.IGNORECASE):
                function_name = match.group(1)
                arguments_str = match.group(2)
                
                try:
                    # Parse the arguments JSON
                    arguments = json.loads(arguments_str)
                    logging.debug(f"Tool Call (XML): {function_name} with args {arguments}")
                    
                    tool_calls.append({
                        "id": self._generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments
                        }
                    })
                except json.JSONDecodeError as e:
                    logging.debug(f"Error parsing XML-style tool call arguments: {e}")
                    continue

        #If we found tools already don't run:
        if not tool_calls:    
            # Pattern 3: Markdown code blocks with JSON containing tool calls
            # Matches: ```json\n{"name": "get_movies", ...}\n``` or ```xml\n{"name": ...}\n```
            markdown_pattern = r'```(?:json|xml)?\s*(\{[^`]+\})\s*```'
            for match in re.finditer(markdown_pattern, content, re.DOTALL | re.IGNORECASE):
                json_block = match.group(1).strip()
                
                try:
                    tool_call_json = json.loads(json_block)
                    
                    # Check if this JSON has a "name" field that matches one of our registered tools
                    if "name" in tool_call_json and tool_call_json["name"] in self.tools:
                        logging.debug(f"Tool Call (Markdown): {tool_call_json}")
                        
                        arguments = tool_call_json.get("arguments", {})
                        if not isinstance(arguments, dict):
                            logging.debug(f"Warning: arguments is not a dict, converting: {arguments}")
                            arguments = {}
                        
                        tool_calls.append({
                            "id": self._generate_tool_call_id(),
                            "type": "function",
                            "function": {
                                "name": tool_call_json["name"],
                                "arguments": arguments
                            }
                        })
                except json.JSONDecodeError as e:
                    logging.debug(f"Error parsing markdown code block: {e}")
                    continue

        #If we found tools already don't run:
        if not tool_calls:    
            # Pattern 4: <function_call>, <tool_call>, <tool>, or <function> with nested <name> and <arguments> tags
            # Matches: <function_call>\n    <name>get_movies</name>\n    <arguments>{"location": "boston"}</arguments>\n</function_call>
            function_call_pattern = r'<(?:function_call|tool_call|tool|function)>\s*<name>([^<]+)</name>\s*<arguments>([^<]+)</arguments>\s*</(?:function_call|tool_call|tool|function)>'
            for match in re.finditer(function_call_pattern, content, re.DOTALL | re.IGNORECASE):
                function_name = match.group(1).strip()
                arguments_str = match.group(2).strip()
                
                try:
                    # Parse the arguments JSON
                    arguments = json.loads(arguments_str)
                    logging.debug(f"Tool Call (nested XML): {function_name} with args {arguments}")
                    
                    tool_calls.append({
                        "id": self._generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments
                        }
                    })
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing nested XML tool call arguments: {e}")
                    continue

        #If we found tools already don't run:
        if not tool_calls:    
            # Pattern 5: Standalone JSON that looks like a tool call
            # Catches: {"name": "get_weather", "arguments": {"location": "boston"}}
            # MUST validate 'name' matches a registered tool to avoid false positives
            standalone_json_pattern = r'(?:^|\n)\s*(\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})'
            for match in re.finditer(standalone_json_pattern, content, re.DOTALL | re.IGNORECASE):
                json_block = match.group(1).strip()
                
                try:
                    tool_call_json = json.loads(json_block)
                    
                    # CRITICAL: Only treat as tool call if name matches a registered tool
                    if "name" in tool_call_json and tool_call_json["name"] in self.tools:
                        logging.debug(f"Tool Call (Standalone JSON): {tool_call_json}")
                        
                        arguments = tool_call_json.get("arguments", {})
                        if not isinstance(arguments, dict):
                            arguments = {}
                        
                        tool_calls.append({
                            "id": self._generate_tool_call_id(),
                            "type": "function",
                            "function": {
                                "name": tool_call_json["name"],
                                "arguments": arguments
                            }
                        })
                    else:
                        # This is just regular JSON output, not a tool call
                        pass
                        
                except json.JSONDecodeError as e:
                    logging.debug(f"Error parsing standalone JSON: {e}")
                    continue

        #If we found tools already don't run:
        if not tool_calls:    
            # Pattern 6: <xml> wrapper with <toolCall> containing nested <name> and <arguments>
            # Matches: <xml><toolCall><name>get_weather</name><arguments>{"location": "boston"}</arguments></toolCall></xml>
            xml_toolcall_pattern = r'<xml>\s*<toolCall>\s*<name>([^<]+)</name>\s*<arguments>([^<]+)</arguments>\s*</toolCall>\s*</xml>'
            for match in re.finditer(xml_toolcall_pattern, content, re.DOTALL | re.IGNORECASE):
                function_name = match.group(1).strip()
                arguments_str = match.group(2).strip()
                
                try:
                    # Parse the arguments JSON
                    arguments = json.loads(arguments_str)
                    logging.debug(f"Tool Call (XML wrapper with toolCall): {function_name} with args {arguments}")
                    
                    tool_calls.append({
                        "id": self._generate_tool_call_id(),
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments
                        }
                    })
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing XML toolCall arguments: {e}")
                    continue


        if tool_calls:
            
            toolpkg = {
                "role": "assistant",
                "content": {
                        "role": "assistant",
                        "tool_calls": tool_calls
                        },
                "tool_calls": tool_calls
            }
            logging.debug(f"\nTOOLS DETECTED: {toolpkg}\n\n")
            return toolpkg
        else:
            return {
                "role": "assistant",
                "content": content.strip()
            }
    
    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Execute tool calls and return results."""
        tool_results = []
        
        for tool_call in tool_calls:
            tool_call_id = tool_call.get("id")
            function_call = tool_call.get("function")
            
            if function_call:
                function_name = function_call["name"]
                function_args = function_call.get("arguments", {})
                
                if function_name in self.tools:
                    try:
                        result = self.tools[function_name](function_args)
                        x={
                            "role": "tool",
                            "name": str(function_name),
                            "content": json.dumps(result) if not isinstance(result, str) else result
                        }
                        logging.debug(x)
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
    
    def _run_generation_and_cleanup(self, text: str, max_new_tokens: int = 2048):
        """Helper function to run generation, decode, and aggressively clean memory."""
        
        with torch.no_grad():
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            input_ids_len = model_inputs.input_ids.shape[1]

            start_time = time.time()

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8
            )
            generation_time = time.time() - start_time

        del model_inputs
        
        output_tokens = [
            output_ids[input_ids_len:] for output_ids in generated_ids
        ]
        output_tokens_count = len(output_tokens[0])
        tps = output_tokens_count / generation_time
        logging.debug(f"[Timing Check] Generated {output_tokens_count} tokens in {generation_time:.3f}s. Observed GPU TPS: {tps:.2f}")
        
        response_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        
        del generated_ids
        del output_tokens
        
        gc.collect()
        
        return response_text
    
    def _generate_streaming(self, model_inputs, max_new_tokens: int = 4500) -> Generator[str, None, str]:
        """
        Generate tokens one at a time and yield them as strings.
        
        Args:
            model_inputs: Tokenized input tensors from the tokenizer
            max_new_tokens: Maximum number of new tokens to generate
            
        Yields:
            str: Each decoded token/chunk as it's generated
            
        Returns:
            str: The complete generated response text
        """
        # Track how many input tokens we started with
        input_ids_len = model_inputs.input_ids.shape[1]
        
        # Store all generated token IDs
        generated_token_ids = []
        
        # Buffer for handling multi-byte characters (like emojis)
        # We accumulate tokens here until they decode cleanly
        decode_buffer = []
        last_output_length = 0  # Track how much text we've already yielded
        
        # Cache for efficient generation (stores attention key/value pairs)
        past_key_values = None
        
        # Start with the input IDs
        input_ids = model_inputs.input_ids
        attention_mask = model_inputs.attention_mask if hasattr(model_inputs, 'attention_mask') else None
        
        start_time = time.time()
        
        # Generate one token at a time
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Prepare inputs for this generation step
                if past_key_values is None:
                    # First step: use full input
                    current_input_ids = input_ids
                    current_attention_mask = attention_mask
                else:
                    # Subsequent steps: only use the last generated token
                    # This is more efficient due to caching
                    current_input_ids = input_ids[:, -1:]
                    if attention_mask is not None:
                        # Extend attention mask by one position
                        current_attention_mask = torch.cat([
                            attention_mask,
                            torch.ones((attention_mask.shape[0], 1), 
                                     dtype=attention_mask.dtype, 
                                     device=attention_mask.device)
                        ], dim=1)
                    else:
                        current_attention_mask = None
                
                # Forward pass through the model
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True  # Enable KV caching for speed
                )
                
                # Get logits for the next token position
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                
                # Apply temperature scaling
                logits = logits / 0.7
                
                # Apply top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > 0.8
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                
                # Sample the next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check if we hit end of sequence
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Add token to our buffers
                generated_token_ids.append(next_token.item())
                decode_buffer.append(next_token.item())
                
                # Try to decode the buffer into text
                token_text = ""
                try:
                    # Decode all tokens in the buffer
                    full_decoded = self.tokenizer.decode(decode_buffer, 
                                                        skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False)
                    
                    # Extract only the new text we haven't yielded yet
                    new_text = full_decoded[last_output_length:]
                    
                    # Check if the text is valid (no replacement characters)
                    if new_text and '�' not in new_text:
                        token_text = new_text
                        last_output_length = len(full_decoded)
                        
                        # Reset buffer periodically to prevent memory buildup
                        if len(decode_buffer) > 20:
                            decode_buffer = []
                            last_output_length = 0
                    else:
                        # Text contains incomplete unicode, keep buffering
                        # But don't let buffer grow too large
                        if len(decode_buffer) > 50:
                            # Force output and reset
                            clean_text = full_decoded[last_output_length:].rstrip('�')
                            if clean_text:
                                token_text = clean_text
                            decode_buffer = []
                            last_output_length = 0
                            
                except UnicodeDecodeError:
                    # Decoding failed, keep buffering
                    if len(decode_buffer) > 50:
                        decode_buffer = decode_buffer[-10:]
                        last_output_length = 0
                
                # Yield the decoded text if we have any
                if token_text:
                    yield token_text
                
                # Update input_ids for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if attention_mask is not None:
                    attention_mask = current_attention_mask
        
        # Handle any remaining buffered tokens at the end
        if decode_buffer:
            try:
                remaining_text = self.tokenizer.decode(decode_buffer, 
                                                      skip_special_tokens=False,
                                                      clean_up_tokenization_spaces=False)
                final_new_text = remaining_text[last_output_length:].rstrip('�')
                if final_new_text:
                    yield final_new_text
            except:
                pass
        
        generation_time = time.time() - start_time
        tps = len(generated_token_ids) / generation_time if generation_time > 0 else 0
        #logging.debug(f"\n[Streaming] Generated {len(generated_token_ids)} tokens in {generation_time:.3f}s. TPS: {tps:.2f}")
        
        # Decode the full response
        if generated_token_ids:
            full_response = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        else:
            full_response = ""
        
        return full_response
    
    def token_count(self, text: str) -> int:
        """Count tokens in text."""
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def strip_response_tags(self,text: str) -> str:
        """Extract content between response tags, or return original if no tags."""
        match = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def chat(self, user_input: str, file_contents: list = None, max_tool_iterations: int = 5) -> str:
        """
        Generate response with recursive tool support.
        Continues calling tools until model stops requesting them or max iterations reached.
        """
        self.messages.append({"role": "user", "content": user_input})

        messages_copy = self.messages.copy()

        # Add file contents if provided
        if file_contents and len(file_contents) > 0:
            for file in file_contents:
                content = file.get("content", "")
                filename = file.get("filename", "unknown")
                messages_copy.insert(1, {
                    "role": "user",
                    "content": f"File '{filename}' has been loaded. Here is its content:\n ### START### \n{content}\n### END ###\n"
                })
        
        iteration = 0
        while iteration < max_tool_iterations:
            iteration += 1
            logging.debug(f"\n[Generation iteration {iteration}]")
            logging.debug(f"\nMessages:{messages_copy}\n")
            
            # Apply chat template
            # On first iteration: add_generation_prompt=True
            # On subsequent iterations after tool results: continue_final_message=True
            if iteration == 1:
                text = self.tokenizer.apply_chat_template(
                    messages_copy,
                    tools=self.available_tools if self.available_tools else None,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # After tool execution, continue the conversation
                text = self.tokenizer.apply_chat_template(
                    messages_copy,
                    tools=self.available_tools if self.available_tools else None,
                    tokenize=False,
                    add_generation_prompt=True,  # Still need generation prompt for new assistant turn
                    #continue_final_message=False  # Start fresh assistant response after tool result
                )
            
            # Generate response
            response_text = self._run_generation_and_cleanup(text)
            
            # Parse for tool calls
            parsed_response = self._parse_tool_calls(response_text)
            
            
            # Check if model wants to use tools
            if parsed_response.get("tool_calls",False):
                tool_calls = parsed_response.get("tool_calls")
                #Format the messages for the fact model called tools
                tool_calls_format =  parsed_response.get("content","")
                messages_copy.append(tool_calls_format)
                
                logging.debug(f"post parse: {tool_calls}")
                logging.debug(f"[Found {len(tool_calls)} tool call(s)]")
                
                tool_results = self._execute_tool_calls(tool_calls)
                messages_copy.extend(tool_results)
                
                # Continue loop to generate response with results from tools
                continue
            else:
                #strip the <response> tags if model used them
                parsed_response["content"] = self.strip_response_tags(parsed_response["content"])
                # No more tool calls, we're done
                messages_copy.append(parsed_response)
                logging.debug("[No tool calls - finishing]")
                break
        
        if iteration >= max_tool_iterations:
            logging.debug(f"[Warning: Reached max tool iterations ({max_tool_iterations})]")
        
        # Update the actual message history with all the interactions
        self.messages = messages_copy
        
        # Return the final assistant response
        return parsed_response["content"]
    
    def chat_streaming(self, user_input: str, file_contents: list = None, max_tool_iterations: int = 5) -> Generator[str, None, str]:
        """
        Generate streaming response with recursive tool support.
        
        Args:
            user_input: The user's message
            file_contents: Optional list of file dictionaries with 'content' and 'filename' keys
            max_tool_iterations: Maximum number of tool call iterations to prevent infinite loops
            
        Yields:
            str: Each token/chunk as it's generated
            
        Returns:
            str: The complete response content
        """
        # Add user message to conversation history
        self.messages.append({"role": "user", "content": user_input})

        messages_copy = self.messages.copy()

        # Insert file contents after system prompt if provided
        if file_contents and len(file_contents) > 0:
            for file in file_contents:
                content = file.get("content", "")
                filename = file.get("filename", "unknown")
                messages_copy.insert(1, {
                    "role": "user",
                    "content": f"File '{filename}' has been loaded. Here is its content:\n ### START### \n{content}\n### END ###\n"
                })
        
        iteration = 0
        tools_were_used = False
        while iteration < max_tool_iterations:
            iteration += 1
            if tools_were_used:
                yield self.streaming_tool_break_flag
                
            logging.debug(f"[Iteration {iteration}]")
            # Apply chat template with tools
            text = self.tokenizer.apply_chat_template(
                messages_copy,
                tools=self.available_tools if self.available_tools else None,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize the input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # Stream the generation
            full_response = ""
            for token_text in self._generate_streaming(model_inputs):
                full_response += token_text
                yield token_text
               
            # Clean up input tensors
            del model_inputs
            gc.collect()
            
            # Parse for tool calls
            parsed_response = self._parse_tool_calls(full_response)
            # Check if model wants to use tools
            if parsed_response.get("tool_calls",False):
                tools_were_used = True
                tool_calls = parsed_response.get("tool_calls")
                #Format the messages for the fact model called tools
                tool_calls_format =  parsed_response.get("content","")
                messages_copy.append(tool_calls_format)
                
                logging.debug(f"post parse: {tool_calls}")
                logging.debug(f"[Found {len(tool_calls)} tool call(s)]")
                
                tool_results = self._execute_tool_calls(tool_calls)
                messages_copy.extend(tool_results)
                
                # Continue loop to generate response with results from tools
                continue
            else:
                #strip the <response> tags if model used them
                parsed_response["content"] = self.strip_response_tags(parsed_response["content"])
                # No more tool calls, we're done
                messages_copy.append(parsed_response)
                logging.debug("[No tool calls - finishing]")
                break
        
        if iteration >= max_tool_iterations:
            logging.debug(f"\n[Warning: Reached max tool iterations ({max_tool_iterations})]")
        
        # Update the actual message history
        self.messages = messages_copy
        
        return parsed_response["content"]

# Example usage
if __name__ == "__main__":
    import time
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    chat = SimpleQwen(model_name="Qwen/Qwen2.5-Coder-7B-Instruct")
    
    def get_weather(args):
        """Get weather information for a location."""
        location = args.get("location", "unknown")
        return f"The weather in Boston is sunny and 75°F"
    
    def get_movies(args):
        """Get movies information for a location."""
        location = args.get("location", "unknown")
        logging.debug(f"GET MOVIES: {args}")
        return f"The movies playing in Boston are Back To The Future 2 at 8pm"
    
    chat.register_tool(get_weather, description="Get current weather for a location")
    chat.register_tool(get_movies, description="Get current movies playing in a location")
    
    logging.debug("Chat started! Type 'quit' or 'exit' to end.")
    logging.debug("Type 'stream' to toggle streaming mode.\n")
    
    use_streaming = False
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        if user_input.lower() == 'stream':
            use_streaming = not use_streaming
            print(f"Streaming mode: {'ON' if use_streaming else 'OFF'}")
            continue
        
        if user_input:
            loop_start = time.time()
            print("Assistant: ", end='', flush=True)
            
            if use_streaming:
                full_response_text = "" 
                # Use streaming mode - capture the return value
                generator = chat.chat_streaming(user_input)
                for token in generator:
                    print(token, end='', flush=True)
                    full_response_text += token
                print()  # Newline after streaming
                logging.debug(f"FINAL POST STREAM: {full_response_text}")
            else:
                # Use regular mode
                response = chat.chat(user_input)
                print(response)
            
            loop_end = time.time()
            elapsed = loop_end - loop_start
            logging.debug(f"\n[Response took {elapsed:.2f} seconds]")