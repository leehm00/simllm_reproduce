"""
Inter-Task KV Reuse Manager for Sim-LLM
Implements global KV cache management with LSH-based similarity search
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import hashlib


class LSHIndex:
    """Simple LSH implementation using Random Projection"""
    
    def __init__(self, embedding_dim: int, num_hyperplanes: int = 16, seed: int = 42):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_hyperplanes: Number of random hyperplanes for hashing
            seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.num_hyperplanes = num_hyperplanes
        np.random.seed(seed)
        
        # Generate random hyperplanes for LSH
        self.hyperplanes = np.random.randn(num_hyperplanes, embedding_dim)
        self.hyperplanes /= np.linalg.norm(self.hyperplanes, axis=1, keepdims=True)
    
    def compute_hash(self, embedding: np.ndarray) -> str:
        """
        Compute LSH hash for an embedding using random projection
        
        Args:
            embedding: Input embedding vector (embedding_dim,)
            
        Returns:
            Hash string representing the bucket
        """
        # Project embedding onto hyperplanes
        projections = np.dot(self.hyperplanes, embedding)
        
        # Create binary hash based on sign of projections
        binary_hash = (projections > 0).astype(int)
        
        # Convert to string for easy bucketing
        hash_str = ''.join(map(str, binary_hash))
        return hash_str
    
    def compute_hash_batch(self, embeddings: np.ndarray) -> List[str]:
        """Compute hashes for a batch of embeddings"""
        projections = np.dot(embeddings, self.hyperplanes.T)
        binary_hashes = (projections > 0).astype(int)
        return [''.join(map(str, h)) for h in binary_hashes]


class TaskCacheEntry:
    """Represents a cached task with its embedding and KV pairs"""
    
    def __init__(
        self,
        task_id: str,
        task_embedding: torch.Tensor,
        lsh_hash: str,
        top_layer_kv: Tuple[torch.Tensor, torch.Tensor],
        timestamp: int,
        penultimate_hidden_states: Optional[torch.Tensor] = None
    ):
        """
        Args:
            task_id: Unique identifier for the task
            task_embedding: Mean-pooled embedding of prompt tokens
            lsh_hash: LSH hash of the embedding
            top_layer_kv: (key, value) tensors from last layer
            timestamp: Access timestamp for LRU
            penultimate_hidden_states: Hidden states from layer N-2 (for layer skipping)
        """
        self.task_id = task_id
        self.task_embedding = task_embedding  # (embedding_dim,)
        self.lsh_hash = lsh_hash
        self.top_layer_kv = top_layer_kv  # (key, value) each (batch, num_heads, seq_len, head_dim)
        self.timestamp = timestamp
        self.access_count = 1
        self.penultimate_hidden_states = penultimate_hidden_states  # (batch, seq_len, hidden_dim)


class InterTaskKVManager:
    """
    Global KV Manager for Inter-Task KV Reuse
    Implements LSH-based similarity search and LRU eviction
    """
    
    def __init__(
        self,
        embedding_dim: int = 4096,
        max_cache_size: int = 100,
        similarity_threshold: float = 0.8,
        num_hyperplanes: int = 16,
        device: str = 'cuda'
    ):
        """
        Args:
            embedding_dim: Dimension of task embeddings
            max_cache_size: Maximum number of cached tasks
            similarity_threshold: Cosine similarity threshold for cache hit
            num_hyperplanes: Number of hyperplanes for LSH
            device: Device to store tensors
        """
        self.embedding_dim = embedding_dim
        self.max_cache_size = max_cache_size
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # LSH index for fast similarity search
        self.lsh_index = LSHIndex(embedding_dim, num_hyperplanes)
        
        # Storage: hash_bucket -> list of TaskCacheEntry
        self.hash_buckets: Dict[str, List[TaskCacheEntry]] = {}
        
        # LRU tracking: task_id -> TaskCacheEntry
        self.cache_entries: OrderedDict[str, TaskCacheEntry] = OrderedDict()
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
        self.current_timestamp = 0
    
    def _compute_task_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute mean-pooled embedding from hidden states
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            
        Returns:
            task_embedding: (hidden_dim,)
        """
        # Mean pool across sequence length
        # hidden_states shape: (batch, seq_len, hidden_dim)
        task_embedding = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        
        # If batch size is 1, squeeze
        if task_embedding.dim() > 1 and task_embedding.shape[0] == 1:
            task_embedding = task_embedding.squeeze(0)  # (hidden_dim,)
        
        # Ensure it's on the right device and detached
        task_embedding = task_embedding.detach()
        
        return task_embedding
    
    def _cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings"""
        # Ensure both tensors are on the same device and detached
        emb1 = emb1.detach().float()
        emb2 = emb2.detach().float()
        
        # Move to same device if needed
        if emb1.device != emb2.device:
            emb2 = emb2.to(emb1.device)
        
        # Flatten if needed
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        
        # Normalize
        emb1_norm = emb1 / (emb1.norm() + 1e-8)
        emb2_norm = emb2 / (emb2.norm() + 1e-8)
        
        similarity = torch.dot(emb1_norm, emb2_norm).item()
        return similarity
    
    def _evict_lru(self):
        """Evict least recently used entry when cache is full"""
        if len(self.cache_entries) >= self.max_cache_size:
            # Remove oldest entry (first item in OrderedDict)
            lru_task_id, lru_entry = self.cache_entries.popitem(last=False)
            
            # Remove from hash bucket
            bucket = self.hash_buckets.get(lru_entry.lsh_hash, [])
            self.hash_buckets[lru_entry.lsh_hash] = [
                e for e in bucket if e.task_id != lru_task_id
            ]
            
            # Clean up empty buckets
            if not self.hash_buckets[lru_entry.lsh_hash]:
                del self.hash_buckets[lru_entry.lsh_hash]
            
            print(f"[KVManager] Evicted LRU entry: {lru_task_id}")
    
    def _validate_kv(self, top_layer_kv: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[bool, str]:
        """
        Validate KV tensors before saving
        
        Returns:
            (is_valid, reason)
        """
        if top_layer_kv is None:
            return False, "top_layer_kv is None"
        
        if not isinstance(top_layer_kv, (tuple, list)) or len(top_layer_kv) != 2:
            return False, f"top_layer_kv should be tuple of 2, got {type(top_layer_kv)}"
        
        key, value = top_layer_kv
        
        if key is None or value is None:
            return False, "key or value is None"
        
        # Check shapes
        if key.dim() < 3 or value.dim() < 3:
            return False, f"Invalid dimensions: key.dim={key.dim()}, value.dim={value.dim()}"
        
        # Check seq_len (shape[2])
        if key.shape[2] == 0 or value.shape[2] == 0:
            return False, f"seq_len is 0: key.shape={key.shape}, value.shape={value.shape}"
        
        # Check for NaN/Inf
        if torch.isnan(key).any() or torch.isinf(key).any():
            return False, "key contains NaN or Inf"
        
        if torch.isnan(value).any() or torch.isinf(value).any():
            return False, "value contains NaN or Inf"
        
        return True, "OK"
    
    def search_similar_task(
        self,
        query_embedding: torch.Tensor,
        task_id: str = "unknown"
    ) -> Optional[TaskCacheEntry]:
        """
        Search for similar cached task using LSH and cosine similarity
        
        Args:
            query_embedding: Task embedding to search for (embedding_dim,)
            task_id: Task ID for logging
            
        Returns:
            Matched TaskCacheEntry if found, None otherwise
        """
        self.total_queries += 1
        
        # Compute query norm
        query_norm = query_embedding.norm().item()
        
        # Compute LSH hash
        query_emb_np = query_embedding.detach().cpu().numpy().astype(np.float32)
        query_hash = self.lsh_index.compute_hash(query_emb_np)
        
        print(f"\n{'='*70}")
        print(f"[KVManager] REQUEST: task_id={task_id}, query_hash={query_hash[:12]}..., query_norm={query_norm:.4f}")
        print(f"[KVManager] Cache size: {len(self.cache_entries)}, Buckets: {len(self.hash_buckets)}")
        
        # Get candidate tasks from same bucket
        candidates = self.hash_buckets.get(query_hash, [])
        
        # Also get all candidates for fallback
        all_candidates = []
        for bucket_hash, bucket_entries in self.hash_buckets.items():
            all_candidates.extend(bucket_entries)
        
        # List all candidate task_ids
        bucket_candidate_ids = [c.task_id for c in candidates]
        all_candidate_ids = [c.task_id for c in all_candidates]
        
        print(f"[KVManager] Bucket candidates: {bucket_candidate_ids}, Total cached entries: {len(all_candidates)}")
        
        # Search in all candidates (not just same bucket) for better hit rate
        search_candidates = all_candidates if len(candidates) == 0 else candidates
        
        if not search_candidates:
            self.cache_misses += 1
            print(f"[KVManager]  Cache MISS - No candidates available")
            print(f"{'='*70}\n")
            return None
        
        # Verify with cosine similarity
        best_match = None
        best_similarity = -1.0
        
        for candidate in search_candidates:
            candidate_norm = candidate.task_embedding.norm().item()
            similarity = self._cosine_similarity(query_embedding, candidate.task_embedding)
            print(f"[KVManager]   Candidate {candidate.task_id} sim={similarity:.4f} norm={candidate_norm:.4f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = candidate
        
        # Check if best match exceeds threshold
        if best_similarity >= self.similarity_threshold:
            self.cache_hits += 1
            
            # Update LRU: move to end
            self.cache_entries.move_to_end(best_match.task_id)
            best_match.timestamp = self.current_timestamp
            best_match.access_count += 1
            self.current_timestamp += 1
            
            print(f"[KVManager]  Cache HIT! Similarity: {best_similarity:.4f} >= {self.similarity_threshold}")
            print(f"[KVManager] Matched task: {best_match.task_id}")
            print(f"{'='*70}\n")
            return best_match
        else:
            self.cache_misses += 1
            print(f"[KVManager]  Cache MISS. Best similarity: {best_similarity:.4f} < {self.similarity_threshold}")
            print(f"{'='*70}\n")
            return None
    
    def add_task(
        self,
        task_id: str,
        task_embedding: torch.Tensor,
        top_layer_kv: Tuple[torch.Tensor, torch.Tensor],
        penultimate_hidden_states: Optional[torch.Tensor] = None
    ) -> bool:
        """
        Add a new task to the cache
        
        Args:
            task_id: Unique identifier for the task
            task_embedding: Mean-pooled embedding (embedding_dim,)
            top_layer_kv: (key, value) from last layer
            penultimate_hidden_states: Hidden states from layer N-2 (for layer skipping)
            
        Returns:
            True if task was added successfully, False otherwise
        """
        # Determine KV info for logging
        kv_present = top_layer_kv is not None
        if kv_present and isinstance(top_layer_kv, (tuple, list)) and len(top_layer_kv) == 2:
            key_shape = top_layer_kv[0].shape if top_layer_kv[0] is not None else None
            value_shape = top_layer_kv[1].shape if top_layer_kv[1] is not None else None
        else:
            key_shape = None
            value_shape = None
        
        penult_shape = penultimate_hidden_states.shape if penultimate_hidden_states is not None else None
        
        print(f"\n{'='*70}")
        print(f"[KVManager] ADD ATTEMPT: task_id={task_id}, kv_present={kv_present}, key.shape={key_shape}, value.shape={value_shape}")
        print(f"[KVManager] Embedding shape: {task_embedding.shape}, norm={task_embedding.norm().item():.4f}")
        print(f"[KVManager] Penultimate hidden states shape: {penult_shape}")
        
        # Validate KV
        is_valid, reason = self._validate_kv(top_layer_kv)
        if not is_valid:
            print(f"[KVManager] ⚠️ INVALID KV - Not adding task. Reason: {reason}")
            print(f"{'='*70}\n")
            return False
        
        # Check if task already exists
        if task_id in self.cache_entries:
            print(f"[KVManager] Task {task_id} already exists, updating...")
            # Update existing entry
            entry = self.cache_entries[task_id]
            entry.task_embedding = task_embedding.detach().clone()
            entry.top_layer_kv = (top_layer_kv[0].detach().clone(), top_layer_kv[1].detach().clone())
            entry.penultimate_hidden_states = penultimate_hidden_states.detach().clone() if penultimate_hidden_states is not None else None
            entry.timestamp = self.current_timestamp
            self.current_timestamp += 1
            self.cache_entries.move_to_end(task_id)
            print(f"[KVManager]  Task updated successfully")
            print(f"[KVManager] Cache size: {len(self.cache_entries)}. Buckets: {len(self.hash_buckets)}")
            print(f"{'='*70}\n")
            return True
        
        # Evict if necessary
        self._evict_lru()
        
        # Compute LSH hash
        emb_np = task_embedding.detach().cpu().numpy().astype(np.float32)
        lsh_hash = self.lsh_index.compute_hash(emb_np)
        print(f"[KVManager] LSH hash: {lsh_hash[:12]}...")
        
        # Create new entry - clone tensors to avoid reference issues
        entry = TaskCacheEntry(
            task_id=task_id,
            task_embedding=task_embedding.detach().clone(),
            lsh_hash=lsh_hash,
            top_layer_kv=(top_layer_kv[0].detach().clone(), top_layer_kv[1].detach().clone()),
            timestamp=self.current_timestamp,
            penultimate_hidden_states=penultimate_hidden_states.detach().clone() if penultimate_hidden_states is not None else None
        )
        self.current_timestamp += 1
        
        # Add to cache
        self.cache_entries[task_id] = entry
        
        # Add to hash bucket
        if lsh_hash not in self.hash_buckets:
            self.hash_buckets[lsh_hash] = []
        self.hash_buckets[lsh_hash].append(entry)
        
        print(f"[KVManager]  Task added. Cache size: {len(self.cache_entries)}. Buckets: {len(self.hash_buckets)}")
        print(f"{'='*70}\n")
        return True
    
    def dump_cache_state(self, top_n: int = 10):
        """
        Dump current cache state for debugging
        
        Args:
            top_n: Number of entries to show details for
        """
        print(f"\n{'='*70}")
        print(f"[KVManager] CACHE STATE DUMP")
        print(f"{'='*70}")
        print(f"Cache size: {len(self.cache_entries)}")
        print(f"Number of buckets: {len(self.hash_buckets)}")
        print(f"Total queries: {self.total_queries}")
        print(f"Cache hits: {self.cache_hits}")
        print(f"Cache misses: {self.cache_misses}")
        hit_rate = self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0
        print(f"Hit rate: {hit_rate:.2%}")
        
        # Print bucket info
        print(f"\nBuckets:")
        for bucket_hash, entries in self.hash_buckets.items():
            print(f"  {bucket_hash[:12]}...: {len(entries)} entries")
        
        # Print top_n entries
        print(f"\nTop {top_n} entries:")
        for i, (task_id, entry) in enumerate(self.cache_entries.items()):
            if i >= top_n:
                break
            key_shape = entry.top_layer_kv[0].shape if entry.top_layer_kv else None
            value_shape = entry.top_layer_kv[1].shape if entry.top_layer_kv else None
            print(f"  [{i+1}] task_id={task_id}, timestamp={entry.timestamp}, access_count={entry.access_count}")
            print(f"       key.shape={key_shape}, value.shape={value_shape}")
        
        print(f"{'='*70}\n")
    
    def get_statistics(self) -> Dict:
        """Get cache statistics"""
        hit_rate = self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_queries': self.total_queries,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache_entries),
            'num_buckets': len(self.hash_buckets)
        }
    
    def reset(self):
        """Reset the cache"""
        self.hash_buckets.clear()
        self.cache_entries.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
        self.current_timestamp = 0
        print("[KVManager] Cache reset")
    
    def generate_task_id(self, input_text: str) -> str:
        """Generate a unique task ID from input text"""
        return hashlib.md5(input_text.encode()).hexdigest()[:16]
