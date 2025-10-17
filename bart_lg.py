
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import socket
import re

MAX_CHUNK_TOKENS = 650
MAX_RESPONSE_TOKENS = 500
OVERLAP_SIZE = 100  # Number of tokens to overlap between chunks

class Summarizer:
    #The bart-large-cnn model is a fine-tuned version of the BART (large-sized) model, 
    # which was specifically trained on the CNN/Daily Mail dataset. 
    # This model is designed for abstractive text summarization tasks,
    # *** cnn *** is not convelutional neural network, but rather a reference to the fine tuning dataset consisting of news articles and their corresponding summaries.
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.model_name = model_name
        self.local_files_only = self.check_internet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        self.model = self.model.to(self.device)

    def check_internet(self):
        """Check if internet connection is available."""
        try:
            socket.create_connection(("huggingface.co", 443), timeout=5)
            #yes internet, so we do NOT want local files only
            return False
        except (socket.timeout, socket.error, OSError):
            #no internet, so we DO want local files only
            return True


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


    def generate_summary(self, chunk):
        #clean chunk
        clean_chunk = self.clean_text_for_bart_cnn(chunk)
        #Feed the raw text chunk to the model, no extra prompt needed for BART
        inputs = self.tokenizer(clean_chunk, return_tensors="pt")
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(inputs['input_ids'], max_length=MAX_RESPONSE_TOKENS, num_return_sequences=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def split_into_chunks_with_overlap(self, text, chunk_size=MAX_CHUNK_TOKENS, overlap_size=OVERLAP_SIZE):
        """
        Split text into overlapping chunks.
        overlap_size: number of tokens to overlap between chunks (typically 10-20% of chunk_size)
        """
        words = text.split()
        chunks = []
        start_idx = 0
        
        while start_idx < len(words):
            # Calculate end of current chunk
            end_idx = min(start_idx + chunk_size, len(words))
            
            # Extract chunk
            chunk = ' '.join(words[start_idx:end_idx])
            chunks.append(chunk)
            
            # Move start index forward, accounting for overlap
            # For the last chunk, we break to avoid duplicating
            if end_idx >= len(words):
                break
            start_idx = end_idx - overlap_size
        
        return chunks

    def summarize_text(self, text):
        # Split the text into chunks
        chunks = self.split_into_chunks_with_overlap(text)
        print(f"Total chunks created: {len(chunks)}")

        # Initialize an empty list to store summaries
        summaries = []

        # Generate summaries for each chunk
        for i, chunk in enumerate(chunks):
            #print(f"Generating summary for chunk {i+1}/{len(chunks)}")
            summary = self.generate_summary(chunk)
            print(f"\n\nChunk {i+1} summary: {summary}\n\n")
            summaries.append(summary)

        # Combine summaries into a single response
        final_response = ' '.join(summaries)

        # return the final response
        return final_response

# Example usage:
if __name__ == "__main__":
    summarizer = Summarizer()
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
"""  # A long string for testing
    summarizer.summarize_text(test_string)