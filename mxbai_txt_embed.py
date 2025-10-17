# mxbai_txt_embed.py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import socket

class MxBaiEmbedder:
    def __init__(self, model_name = "mixedbread-ai/mxbai-embed-large-v1"):
        self.tokenizer = None
        self.model = None
        self.local_files_only = self.check_internet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings_store = {}  # Dict to store embeddings with UUID keys
        self.metadata_store = {}    # Store original text and other metadata
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=self.local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=self.local_files_only)
        self.model = self.model.to(self.device)
        self.model.eval()


    def check_internet(self):
        """Check if internet connection is available."""
        try:
            socket.create_connection(("huggingface.co", 443), timeout=5)
            #yes internet, so we do NOT want local files only
            return False
        except (socket.timeout, socket.error, OSError):
            #no internet, so we DO want local files only
            return True
    
    def embed_text_string(self, text: str) -> np.ndarray:
        """
        Create an embedding for a given text string
        
        Args:
            text (str): The input text to embed
            
        Returns:
            np.ndarray: The embedding vector
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                               padding=True, max_length=512)
        
        # Move input tensors to the correct device (GPU/CUDA) 
        inputs = {k: v.to(self.device) for k, v in inputs.items()}


        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use mean pooling of the last hidden state
            # This is a common approach for sentence embeddings
            embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            
            # Apply attention mask and mean pooling
            masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
            summed_embeddings = torch.sum(masked_embeddings, dim=1)
            summed_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = summed_embeddings / summed_mask.unsqueeze(-1)
            
            if self.device.type == 'cuda':
                # Convert to numpy array (must move back to CPU first if on GPU)
                embedding_vector = mean_pooled.squeeze().cpu().numpy()
            else:
                # Convert to numpy array
                embedding_vector = mean_pooled.squeeze().numpy()
            
            # Normalize the embedding (optional but often helpful for similarity search)
            embedding_vector = embedding_vector / np.linalg.norm(embedding_vector)
            
            return embedding_vector
        
    def compare_strings(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between embeddings of two text strings
        
        Args:
            text1 (str): First input text
            text2 (str): Second input text
            
        Returns:
            float: Cosine similarity score
        """
        emb1 = self.embed_text_string(text1)
        emb2 = self.embed_text_string(text2)
        return self.compare_embeddings(emb1, emb2)
    
    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            emb1 (np.ndarray): First embedding vector
            emb2 (np.ndarray): Second embedding vector
            
        Returns:
            float: Cosine similarity score
        """
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)
        return similarity[0][0]