"""
VideoGraph implementation ported from original M3-Agent.
Core data structure for memory storage and retrieval.
"""

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import json
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class VideoGraph:
    """
    VideoGraph class for managing video memory nodes and relationships.
    Ported from /n/fs/scratch/sb6870/m3-agent-fixed/mmagent/videograph.py
    """
    
    def __init__(self, max_img_embeddings=10, max_audio_embeddings=20, 
                 img_matching_threshold=0.3, audio_matching_threshold=0.6):
        """Initialize a video graph with nodes for faces, voices and text events."""
        self.nodes = {}  # node_id -> node object
        self.edges = {}  # (node_id1, node_id2) -> edge weight
        
        # Maintain ordered text nodes
        self.text_nodes = []  # List of text node IDs in insertion order
        self.text_nodes_by_clip = {}
        self.event_sequence_by_clip = {}
        
        self.max_img_embeddings = max_img_embeddings
        self.max_audio_embeddings = max_audio_embeddings
        
        self.img_matching_threshold = img_matching_threshold
        self.audio_matching_threshold = audio_matching_threshold
        
        self.next_node_id = 0
        
        # Character tracking
        self.reverse_character_mappings = {}  # For character ID translation
        self.character_mappings = {}
    
    class Node:
        """Node class for storing embeddings and metadata."""
        def __init__(self, node_id, node_type):
            self.id = node_id
            self.type = node_type  # 'img', 'voice', 'episodic' or 'semantic'
            self.embeddings = []
            self.metadata = {}
    
    def _average_similarity(self, embeddings1, embeddings2):
        """Calculate average cosine similarity between two lists of embeddings."""
        if not embeddings1 or not embeddings2:
            return 0
            
        # Convert lists to numpy arrays
        emb1_array = np.array(embeddings1)
        emb2_array = np.array(embeddings2)
        
        # Calculate pairwise cosine similarities between all embeddings
        similarities = cosine_similarity(emb1_array, emb2_array)
        
        # Return mean of all pairwise similarities
        return np.mean(similarities)
    
    def add_img_node(self, imgs):
        """Add a new image node with embeddings and content."""
        node = self.Node(self.next_node_id, 'img')
        
        img_embeddings = imgs['embeddings']
        node.embeddings.extend(img_embeddings[:self.max_img_embeddings])
        node.metadata['contents'] = imgs['contents']
        
        self.nodes[self.next_node_id] = node
        node_id = self.next_node_id
        self.next_node_id += 1
        
        logger.debug(f"Image node added with ID {node_id}")
        return node_id
    
    def add_voice_node(self, audios):
        """Add a new voice node with embeddings and content."""
        node = self.Node(self.next_node_id, 'voice')
        
        audio_embeddings = audios['embeddings']
        node.embeddings.extend(audio_embeddings[:self.max_audio_embeddings])
        node.metadata['contents'] = audios['contents']
        
        self.nodes[self.next_node_id] = node
        node_id = self.next_node_id
        self.next_node_id += 1
        
        logger.debug(f"Voice node added with ID {node_id}")
        return node_id
    
    def add_text_node(self, memory, clip_id, memory_type='episodic'):
        """Add a new text node (episodic or semantic memory)."""
        node = self.Node(self.next_node_id, memory_type)
        
        # Handle different memory formats
        if isinstance(memory, dict):
            contents = memory.get('contents', [])
            embeddings = memory.get('embeddings', [])
        else:
            contents = [str(memory)]
            embeddings = []
        
        node.metadata['contents'] = contents
        node.metadata['timestamp'] = clip_id
        node.embeddings = embeddings
        
        self.nodes[self.next_node_id] = node
        node_id = self.next_node_id
        self.next_node_id += 1
        
        # Track text nodes by clip
        if clip_id not in self.text_nodes_by_clip:
            self.text_nodes_by_clip[clip_id] = []
        self.text_nodes_by_clip[clip_id].append(node_id)
        
        # Add to ordered text nodes list
        self.text_nodes.append(node_id)
        
        logger.debug(f"Text node ({memory_type}) added with ID {node_id} for clip {clip_id}")
        return node_id
    
    def update_node(self, node_id, new_data):
        """Update an existing node with new data."""
        if node_id not in self.nodes:
            logger.warning(f"Node {node_id} not found for update")
            return
        
        node = self.nodes[node_id]
        
        # Update embeddings
        if 'embeddings' in new_data:
            if node.type == 'img':
                node.embeddings.extend(new_data['embeddings'][:self.max_img_embeddings - len(node.embeddings)])
            elif node.type == 'voice':
                node.embeddings.extend(new_data['embeddings'][:self.max_audio_embeddings - len(node.embeddings)])
        
        # Update contents
        if 'contents' in new_data:
            if 'contents' not in node.metadata:
                node.metadata['contents'] = []
            node.metadata['contents'].extend(new_data['contents'])
        
        logger.debug(f"Node {node_id} updated")
    
    def add_edge(self, node_id1, node_id2, weight=1.0):
        """Add an edge between two nodes."""
        if node_id1 in self.nodes and node_id2 in self.nodes:
            self.edges[(node_id1, node_id2)] = weight
            self.edges[(node_id2, node_id1)] = weight  # Undirected edge
            logger.debug(f"Edge added between {node_id1} and {node_id2} with weight {weight}")
    
    def search_img_nodes(self, img_info):
        """Search for similar image nodes."""
        query_embeddings = img_info['embeddings']
        
        matched_nodes = []
        for node_id, node in self.nodes.items():
            if node.type == 'img':
                similarity = self._average_similarity(query_embeddings, node.embeddings)
                if similarity > self.img_matching_threshold:
                    matched_nodes.append((node_id, similarity))
        
        # Sort by similarity (descending)
        matched_nodes.sort(key=lambda x: x[1], reverse=True)
        return matched_nodes
    
    def search_voice_nodes(self, voice_info):
        """Search for similar voice nodes."""
        query_embeddings = voice_info['embeddings']
        
        matched_nodes = []
        for node_id, node in self.nodes.items():
            if node.type == 'voice':
                similarity = self._average_similarity(query_embeddings, node.embeddings)
                if similarity > self.audio_matching_threshold:
                    matched_nodes.append((node_id, similarity))
        
        # Sort by similarity (descending)
        matched_nodes.sort(key=lambda x: x[1], reverse=True)
        return matched_nodes
    
    def search_text_nodes(self, query_embeddings, range_nodes=[], mode="max"):
        """Search for text nodes using text embeddings.

        Args:
            query_embeddings: Single embedding or list of embeddings
            range_nodes: Optional list of nodes to restrict search to
            mode: Similarity calculation mode ('max' for character translation)

        Returns:
            List of (node_id, similarity_score) tuples sorted by score
        """
        # Handle single embedding (backward compatibility)
        if not isinstance(query_embeddings, list):
            query_embeddings = [query_embeddings]

        if not query_embeddings or not query_embeddings[0]:
            return []

        # Get target nodes - restrict to range_nodes if provided
        if range_nodes:
            target_nodes = []
            for node_id in range_nodes:
                if node_id in self.nodes:
                    connected = self.get_connected_nodes(node_id, type=['episodic', 'semantic'])
                    target_nodes.extend(connected)
            target_nodes = list(set(target_nodes))  # Remove duplicates
        else:
            target_nodes = [node_id for node_id, node in self.nodes.items()
                           if node.type in ['episodic', 'semantic'] and node.embeddings]

        # Calculate similarities for all query embeddings
        all_similarities = []

        for query_emb in query_embeddings:
            if not query_emb:
                continue
            query_array = np.array(query_emb).reshape(1, -1)

            for node_id in target_nodes:
                node = self.nodes[node_id]
                if not node.embeddings:
                    continue

                # Calculate similarity with node embeddings
                node_emb = np.array(node.embeddings[0]).reshape(1, -1)
                similarity = cosine_similarity(query_array, node_emb)[0][0]
                all_similarities.append((node_id, similarity))

        # Aggregate by node_id (take max similarity across all queries)
        node_similarities = {}
        for node_id, similarity in all_similarities:
            if node_id not in node_similarities or similarity > node_similarities[node_id]:
                node_similarities[node_id] = similarity

        # Sort by aggregated similarity
        sorted_similarities = sorted(node_similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_similarities
    
    def get_clip_memories(self, clip_id):
        """Get all memories for a specific clip."""
        if clip_id not in self.text_nodes_by_clip:
            return []
        
        memories = []
        for node_id in self.text_nodes_by_clip[clip_id]:
            node = self.nodes[node_id]
            memories.extend(node.metadata.get('contents', []))
        
        return memories
    
    def truncate_memory_by_clip(self, max_clip_id, inclusive=True):
        """Remove memories after a certain clip ID."""
        nodes_to_remove = []
        
        for node_id, node in self.nodes.items():
            if node.type in ['episodic', 'semantic']:
                timestamp = node.metadata.get('timestamp', 0)
                if (inclusive and timestamp > max_clip_id) or (not inclusive and timestamp >= max_clip_id):
                    nodes_to_remove.append(node_id)
        
        # Remove nodes
        for node_id in nodes_to_remove:
            del self.nodes[node_id]
            if node_id in self.text_nodes:
                self.text_nodes.remove(node_id)
        
        # Update text_nodes_by_clip
        clips_to_update = []
        for clip_id, node_list in self.text_nodes_by_clip.items():
            updated_nodes = [n for n in node_list if n in self.nodes]
            if len(updated_nodes) != len(node_list):
                clips_to_update.append((clip_id, updated_nodes))
        
        for clip_id, updated_nodes in clips_to_update:
            if updated_nodes:
                self.text_nodes_by_clip[clip_id] = updated_nodes
            else:
                del self.text_nodes_by_clip[clip_id]
        
        logger.info(f"Truncated {len(nodes_to_remove)} nodes after clip {max_clip_id}")
    
    def refresh_equivalences(self):
        """Build character mappings from equivalence statements using Union-Find.

        EXACT port from m3-agent-fixed videograph.py lines 442-511.
        """
        # Union-Find data structure
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y

        # Process equivalence statements from semantic memories
        for node_id, node in self.nodes.items():
            if (node.type == 'semantic' and 'contents' in node.metadata):
                for content in node.metadata['contents']:
                    if content.lower().startswith("equivalence: "):
                        # Import here to avoid circular import
                        from .memory_builder import parse_video_caption
                        entities = parse_video_caption(self, content)
                        if len(entities) >= 2:
                            anchor_node = entities[0][1]
                            for entity in entities[1:]:
                                union(anchor_node, entity[1])

        # Group nodes by their representative (character)
        character_mappings = {}
        character_count = 0
        root_to_character = {}

        for node_id, node in self.nodes.items():
            if node.type not in ['img', 'voice']:
                continue
            root = find(node_id)
            tag = f"face_{node_id}" if node.type == 'img' else f"voice_{node_id}"
            if root not in root_to_character:
                root_to_character[root] = f"character_{character_count}"
                character_count += 1
            character = root_to_character[root]
            if character not in character_mappings:
                character_mappings[character] = []
            character_mappings[character].append(tag)

        # Create reverse mapping
        reverse_character_mappings = {}
        for character, tags in character_mappings.items():
            for tag in tags:
                reverse_character_mappings[tag] = character

        self.character_mappings = character_mappings
        self.reverse_character_mappings = reverse_character_mappings

        logger.info(f"refresh_equivalences: Found {character_count} characters")
    
    def get_stats(self):
        """Get statistics about the video graph."""
        stats = {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': {},
            'clips_processed': len(self.text_nodes_by_clip),
            'text_nodes': len(self.text_nodes)
        }
        
        # Count nodes by type
        for node in self.nodes.values():
            node_type = node.type
            if node_type not in stats['node_types']:
                stats['node_types'][node_type] = 0
            stats['node_types'][node_type] += 1
        
        return stats
    
    def get_memory_size_estimate(self):
        """Summarise graph state by counting embeddings, floats, and content items."""

        def _count_numeric_elements(obj):
            if obj is None:
                return 0
            if isinstance(obj, (int, float)):
                return 1
            if isinstance(obj, np.ndarray):
                return int(obj.size)
            if hasattr(obj, 'numel') and callable(obj.numel):
                try:
                    return int(obj.numel())
                except TypeError:
                    return 0
            if isinstance(obj, (list, tuple)):
                return sum(_count_numeric_elements(item) for item in obj)
            return 0

        def _normalise_contents(raw_contents):
            if raw_contents is None:
                return []
            if isinstance(raw_contents, (list, tuple, set)):
                return list(raw_contents)
            return [raw_contents]

        node_counts: Dict[str, int] = {}
        embedding_stats: Dict[str, Dict[str, int]] = {}
        content_stats: Dict[str, Dict[str, int]] = {}

        total_embeddings = 0
        total_floats = 0
        total_string_items = 0
        total_string_chars = 0
        total_binary_items = 0
        total_binary_bytes = 0
        total_other_items = 0

        for node in self.nodes.values():
            node_type = node.type
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

            vectors = node.embeddings or []
            vector_count = len(vectors)
            float_count = sum(_count_numeric_elements(vec) for vec in vectors)

            total_embeddings += vector_count
            total_floats += float_count

            type_embed_stats = embedding_stats.setdefault(node_type, {'vectors': 0, 'floats': 0})
            type_embed_stats['vectors'] += vector_count
            type_embed_stats['floats'] += float_count

            contents = _normalise_contents(node.metadata.get('contents'))
            type_content_stats = content_stats.setdefault(
                node_type,
                {
                    'string_items': 0,
                    'string_characters': 0,
                    'binary_items': 0,
                    'binary_bytes': 0,
                    'other_items': 0,
                },
            )

            for content in contents:
                if isinstance(content, str):
                    length = len(content)
                    total_string_items += 1
                    total_string_chars += length
                    type_content_stats['string_items'] += 1
                    type_content_stats['string_characters'] += length
                elif isinstance(content, (bytes, bytearray)):
                    length = len(content)
                    total_binary_items += 1
                    total_binary_bytes += length
                    type_content_stats['binary_items'] += 1
                    type_content_stats['binary_bytes'] += length
                elif content is None:
                    continue
                else:
                    total_other_items += 1
                    type_content_stats['other_items'] += 1

        undirected_edges = len({tuple(sorted(edge)) for edge in self.edges.keys()})
        total_edge_floats = sum(1 for weight in self.edges.values() if isinstance(weight, (int, float)))

        return {
            'node_counts': node_counts,
            'embedding_stats': {
                'total_vectors': total_embeddings,
                'total_floats': total_floats,
                'by_type': embedding_stats,
            },
            'content_stats': {
                'string_items': total_string_items,
                'string_characters': total_string_chars,
                'binary_items': total_binary_items,
                'binary_bytes': total_binary_bytes,
                'other_items': total_other_items,
                'by_type': content_stats,
            },
            'edge_stats': {
                'directed_edges': len(self.edges),
                'undirected_edges': undirected_edges,
                'weight_scalars': total_edge_floats,
            },
        }
    
    def _remove_node(self, node_id: str):
        """Remove node and all associated edges."""
        if node_id not in self.nodes:
            return
            
        # Remove all edges involving this node
        edges_to_remove = []
        for edge_key in self.edges.keys():
            if node_id in edge_key:
                edges_to_remove.append(edge_key)
        
        for edge_key in edges_to_remove:
            del self.edges[edge_key]
        
        # Remove node
        del self.nodes[node_id]

    def get_connected_nodes(self, node_id, type=['img', 'voice', 'episodic', 'semantic']):
        """Get all nodes connected to given node."""
        connected = set()
        for (n1, n2), _ in self.edges.items():
            if n1 == node_id and n2 in self.nodes and self.nodes[n2].type in type:
                connected.add(n2)
            elif n2 == node_id and n1 in self.nodes and self.nodes[n1].type in type:
                connected.add(n1)
        return list(connected)

    def reinforce_node(self, node_id, delta_weight=1):
        """Reinforce all edges connected to the given node."""
        if node_id not in self.nodes:
            return 0
        reinforced_count = 0
        for (n1, n2) in list(self.edges.keys()):
            if n1 == node_id or n2 == node_id:
                current_weight = self.edges.get((n1, n2), 1.0)
                self.edges[(n1, n2)] = current_weight + delta_weight
                self.edges[(n2, n1)] = current_weight + delta_weight
                reinforced_count += 1
        return reinforced_count

    def weaken_node(self, node_id, delta_weight=1):
        """Weaken all edges connected to the given node."""
        if node_id not in self.nodes:
            return 0
        weakened_count = 0
        for (n1, n2) in list(self.edges.keys()):
            if n1 == node_id or n2 == node_id:
                current_weight = self.edges.get((n1, n2), 1.0)
                new_weight = max(0.1, current_weight - delta_weight)
                self.edges[(n1, n2)] = new_weight
                self.edges[(n2, n1)] = new_weight
                weakened_count += 1
        return weakened_count
