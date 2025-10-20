# bart_lg.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import socket
import re
import gc
from mxbai_txt_embed import MxBaiEmbedder
import logging
logging.basicConfig(level=logging.INFO)

MAX_CHUNK_TOKENS = 950 #model max is 1024 input
MAX_RESPONSE_TOKENS = 500
OVERLAP_SIZE = 125  # Number of tokens to overlap between chunks
MIN_SIMILARITY_REJECTION = 0.5 #when doing a similarity based return, below this is not even included in pool for ranking
TOP_k_RATIO = 0.75 #percential threshold
class TextSummarizer:
    #The bart-large-cnn model is a fine-tuned version of the BART (large-sized) model, 
    # which was specifically trained on the CNN/Daily Mail dataset. 
    # This model is designed for abstractive text summarization tasks,
    # *** cnn *** is not convelutional neural network, but rather a reference to the fine tuning dataset consisting of news articles and their corresponding summaries.
    def __init__(self, model_name="facebook/bart-large-cnn",keep_loaded_in_memory=False):
        self.model_name = model_name
        self.local_files_only = self.check_internet()
        #Set to None because assignment is dynamic based on the keep_loaded_in_memory option.
        self.device = None
        self.tokenizer = None
        self.model = None
        self.keep_loaded_in_memory = keep_loaded_in_memory
        self.embedder = MxBaiEmbedder() # Initialize the embedder


    def check_internet(self):
        """Check if internet connection is available."""
        try:
            socket.create_connection(("huggingface.co", 443), timeout=5)
            #yes internet, so we do NOT want local files only
            return False
        except (socket.timeout, socket.error, OSError):
            #no internet, so we DO want local files only
            return True
        
    def _load_model(self):
        self.device = torch.device(self.use_gpu_with_most_memory())
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _unload_model(self):
        """Removes the model and clears memory caches."""
        if self.model is not None:
            print(f"Unloading model '{self.model_name}' and clearing memory...")
            
            # Move the model to CPU and delete it
            self.model.to("cpu")
            del self.model
            
            # Delete tokenizer
            del self.tokenizer
            
            # Reset attributes to None
            self.model = None
            self.tokenizer = None
            
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch device cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            print("Model unloaded and memory caches cleared.")
    
    def use_gpu_with_most_memory(self):
        """To help prevent OOM errors when using cuda, find device with most space"""
        #check for CUDA
        if not torch.cuda.is_available():
            #check for MPS
            if torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        
        devices = torch.cuda.device_count()
        max_memory = -1
        best_device = None
        #Using CUDA, if multiple devices use the one with most mem availible.
        for i in range(devices):
            memory = torch.cuda.memory_allocated(i)
            if memory > max_memory:
                max_memory = memory
                best_device = i
        
        return f"cuda:{best_device}"
    
    def clean_text_for_bart_cnn(self,text):
        """
        Cleans text, keeping only English alphanumeric characters, standard 
        punctuation, parentheses, and hyphens.

        Args:
            text (str): The input text containing various scripts/symbols.

        Returns:
            str: The strictly cleaned text.
        """
        
        # Define the allowed set of characters:
        # a-zA-Z0-9 : English alphanumeric
        # \s        : Whitespace (space, tab, newline)
        # .,?!:;'"  : Standard punctuation
        # ()        : Parentheses
        # \-        : Literal hyphen (must be escaped or placed at the start/end)
        # &%$/@     : Adding a few common ASCII symbols often safely included (optional, but practical)
        
        allowed_pattern = r'[^a-zA-Z0-9\s.,?!:;\'"()\[\]\-\&]' 

        cleaned_text = re.sub(allowed_pattern, ' ', text)
        # Replace multiple spaces/newlines with a single space
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text

    def _generate_summary(self, chunk):
        #clean chunk
        clean_chunk = self.clean_text_for_bart_cnn(chunk)
        #Feed the raw text chunk to the model, no extra prompt needed for BART
        inputs = self.tokenizer(clean_chunk, return_tensors="pt")
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(inputs['input_ids'], max_length=MAX_RESPONSE_TOKENS, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _split_into_token_chunks(self, text, chunk_size=MAX_CHUNK_TOKENS, overlap_size=OVERLAP_SIZE):
        """
        Splits text into chunks based on exact token count (IDs), Not splitting on words or characters,
        ensuring compliance with the model's context limit.
        """
        # 1. Encode the entire text into token IDs
        # We use add_special_tokens=False because the generation step 
        # (in _generate_summary) will add <s> and </s> automatically.
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
        total_length = len(token_ids)
        chunks = []
        
        start = 0
        while start < total_length:
            # Determine the end of the current chunk
            end = min(start + chunk_size, total_length)
            
            # Extract the token slice
            chunk_ids = token_ids[start:end]
            
            # Decode the token slice back into a string chunk
            # skip_special_tokens=True ensures the output is clean text
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            
            # Check for termination
            if end == total_length:
                break
                
            # Calculate the next starting point, moving backward by the overlap amount
            # The overlap must also be based on tokens!
            start = end - overlap_size
            
            # Safety check: ensure start is not negative
            if start < 0:
                start = 0
                
        return chunks
    
    def summarize_text(self, text, query_match=False, top_k_ratio=TOP_k_RATIO):
        if not self.keep_loaded_in_memory or self.model is None:
            self._load_model()

        chunks = self._split_into_token_chunks(text)
        logging.debug(f"Total chunks created: {len(chunks)}")
        final_response =""
        if query_match:
            # Collect ALL summaries with their similarity scores
            summary_scores = []
            
            for i, chunk in enumerate(chunks):
                logging.debug(f"Generating summary for chunk {i+1}/{len(chunks)}")
                summary = self._generate_summary(chunk)
                similarity = self.embedder.compare_strings(summary, query_match)
                
                # Only include summaries with similarity above minimum threshold
                if similarity > MIN_SIMILARITY_REJECTION:
                    summary_scores.append((summary, similarity))
                    logging.debug(f"Chunk {i+1} similarity: {similarity:.2f} - ACCEPTED")
                else:
                    logging.debug(f"Chunk {i+1} similarity: {similarity:.2f} - REJECTED (below 0.3)")
            
            if summary_scores:
                # Sort by similarity (descending)
                summary_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Calculate how many to keep (top 1-top_k_ratio proportion)
                num_to_keep = max(1, int(len(summary_scores) * (1 - top_k_ratio)))
                
                # Keep only the top N most relevant summaries
                summaries = [summary for summary, score in summary_scores[:num_to_keep]]
                
                logging.info(f"Kept top {num_to_keep}/{len(summary_scores)} summaries")
                logging.info(f"Similarity range: {summary_scores[0][1]:.2f} to {summary_scores[num_to_keep-1][1]:.2f}")

                final_response = ' '.join(summaries)
                logging.info(f"\n\nFinal summarized response generated: {final_response}\n\n")
                
            else:
                logging.info(f"\n\nFinal summarized response generated NO SIMILAR text: {final_response}\n\n")

        else:
            # No query matching, keep all summaries
            summaries = []
            for i, chunk in enumerate(chunks):
                logging.debug(f"Generating summary for chunk {i+1}/{len(chunks)}")
                summary = self._generate_summary(chunk)
                summaries.append(summary)

            final_response = ' '.join(summaries)
            logging.info(f"\n\nFinal summarized response generated: {final_response}\n\n")
        
        if not self.keep_loaded_in_memory:
            self._unload_model()

        return final_response

# Example usage:
if __name__ == "__main__":
    summarizer = TextSummarizer()
    test_string = """The myths given in this paper are part of a large body of material
collected among the Cherokee, chiefly in successive field seasons
from 1887 to 1890, inclusive, and comprising more or less extensive
notes, together with original Cherokee manuscripts, relating to the
history, archeology, geographic nomenclature, personal names, botany,
medicine, arts, home life, religion, songs, ceremonies, and language
of the tribe. It is intended that this material shall appear from time
to time in a series of papers which, when finally brought together,
shall constitute a monograph upon the Cherokee Indians. This paper may
be considered the first of the series, all that has hitherto appeared
being a short paper upon the sacred formulas of the tribe, published
in the Seventh Annual Report of the Bureau in 1891 and containing a
synopsis of the Cherokee medico-religious theory, with twenty-eight
specimens selected from a body of about six hundred ritual formulas
written down in the Cherokee language and alphabet by former doctors
of the tribe and constituting altogether the largest body of aboriginal
American literature in existence.
""" 
    # A long string for testing
    summarizer.summarize_text(test_string)