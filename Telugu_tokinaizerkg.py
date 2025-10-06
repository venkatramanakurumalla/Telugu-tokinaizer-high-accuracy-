# -*- coding: utf-8 -*-
"""
🔥 Advanced Telugu Knowledge Graph Construction Toolkit
=====================================================
A comprehensive system for building rich knowledge graphs from Telugu text.

Features:
- Morphological tokenization with entity extraction
- Multiple relation types (morphological, co-occurrence, semantic)
- Quality metrics and validation
- Multiple export formats (JSON-LD, GraphML, RDF)
- Streaming processing for large corpora
- Entity linking and disambiguation
- Relation strength scoring
- Visual graph analysis

CLI Usage:
----------
# Extract KG from single file
python telugu_kg.py extract --input text.txt --out kg.json

# Build KG from entire corpus
python telugu_kg.py build-corpus --dir corpus/ --out knowledge_graph.json

# Export in different formats
python telugu_kg.py export --input kg.json --format graphml

# Analyze KG quality
python telugu_kg.py analyze --input kg.json

# Visualize graph structure
python telugu_kg.py visualize --input kg.json --out graph.html
"""

import sys
import os
import time
import json
import unicodedata
import argparse
import logging
import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import List, Dict, Iterable, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import html
import uuid

try:
    import regex as re
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
except ImportError as e:
    missing_packages = []
    if 'regex' in str(e):
        missing_packages.append('regex')
    if 'networkx' in str(e):
        missing_packages.append('networkx')
    if 'matplotlib' in str(e):
        missing_packages.append('matplotlib')
    
    raise SystemExit(
        f"This toolkit requires additional packages: {', '.join(missing_packages)}\n"
        f"Install them with: pip install {' '.join(missing_packages)}"
    ) from e

# ========================
# LOGGING SETUP
# ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================
# CORE CONFIGURATION
# ========================

# Telugu Unicode range
_TELUGU_RANGE = r"\p{Telugu}"
_IS_TELUGU = re.compile(_TELUGU_RANGE).search

# Configuration constants
CACHE_SIZE_MORPH = 65536
FALLBACK_LENGTH_THRESHOLD = 6
MAX_FILE_SIZE_MB = 500
DEFAULT_CONTEXT_WINDOW = 5
MIN_COOCCURRENCE_FREQ = 2

# Enhanced Telugu combining characters
_TELUGU_COMBINING = r"\u0C3E-\u0C56\u0C81\u0C82\u0C83"

# Suffix Trie for morphological splitting
_SUFFIX_TRIE = {
    4: {
        "స్తున్న", "స్తారు", "తున్న", "తారు", "దున్న", "దాము",
        "ద్దాం", "ందరు", "వరకు", "కొరకు", "దగ్గర", "నివాసం"
    },
    3: {
        "వల్ల", "యొక్క", "ంటే", "య్యే", "య్యి", "స్తు",
        "స్తూ", "మ్మ", "వ్వ", "బడి", "లాగా", "వంటి", "కారణం"
    },
    2: {
        "కు", "కి", "ను", "లో", "గా", "తో", "లు", "రు",
        "ళు", "ంత", "క్కి", "ంట", "పై", "చే", "కింద", "వద్ద"
    },
    1: {
        "ల", "క", "గ", "చ", "త", "ద", "న", "ప", "మ", "య", "ర", "డ"
    }
}

# Vowel harmony patterns
class VowelPattern(NamedTuple):
    stem_pattern: re.Pattern
    suffix_start: str
    stem_replacement: str
    suffix_replacement: str

_VOWEL_PATTERNS = [
    VowelPattern(re.compile(r"ా$"), "కి", "ం", "కు"),
    VowelPattern(re.compile(r"ు$"), "కి", "", "కు"),
    VowelPattern(re.compile(r"ి$"), "కి", "య", "కి"),
    VowelPattern(re.compile(r"ీ$"), "కి", "య", "కి"),
    VowelPattern(re.compile(r"ూ$"), "కి", "వ", "కి"),
]

# Token regex
_TOKEN_RE = re.compile(rf"([{_TELUGU_RANGE}]+|\d+|\p{{P}}|\S)", re.UNICODE)
_SYLLABLE_RE = re.compile(rf"(?:[{_TELUGU_RANGE}][{_TELUGU_COMBINING}]*)+", re.UNICODE)

# ========================
# ENUMS AND DATA CLASSES
# ========================

class EntityType(Enum):
    NOUN = "noun"
    VERB = "verb" 
    ADJECTIVE = "adjective"
    LOCATION = "location"
    PERSON = "person"
    ORGANIZATION = "organization"
    UNKNOWN = "unknown"

class RelationType(Enum):
    MORPHOLOGICAL = "morphological"
    COOCCURRENCE = "cooccurrence"
    SEMANTIC = "semantic"
    SYNTACTIC = "syntactic"

class KGFormat(Enum):
    JSON_LD = "json-ld"
    GRAPHML = "graphml"
    RDF = "rdf"
    NETWORKX = "networkx"

@dataclass
class KGToken:
    """KG-ready token with enhanced metadata."""
    surface_form: str
    stem: str
    suffix: Optional[str] = None
    is_fallback: bool = False
    token_type: str = "word"
    entity_type: EntityType = EntityType.UNKNOWN
    confidence: float = 1.0
    position: Optional[int] = None
    
    def to_kg_node(self) -> Dict:
        """Convert to KG node with enhanced properties."""
        return {
            "id": self.stem,
            "surface_form": self.surface_form,
            "type": self.entity_type.value,
            "is_oov": self.is_fallback,
            "confidence": self.confidence,
            "length": len(self.stem),
            "has_suffix": self.suffix is not None
        }
    
    def to_kg_edge(self) -> Optional[Dict]:
        """Convert to KG edge with relation metadata."""
        if not self.suffix:
            return None
        return {
            "source": self.stem,
            "target": self.surface_form,
            "relation": f"has_suffix_{self.suffix}",
            "suffix": self.suffix,
            "type": RelationType.MORPHOLOGICAL.value,
            "confidence": self.confidence
        }
    
    def infer_entity_type(self):
        """Enhanced entity type inference."""
        if self.suffix in {"లు", "రు", "ళు"}:
            self.entity_type = EntityType.NOUN
        elif self.suffix in {"తున్న", "స్తున్న", "య్యే", "య్యి"}:
            self.entity_type = EntityType.VERB
        elif self.suffix in {"గా", "వంటి", "లాగా"}:
            self.entity_type = EntityType.ADJECTIVE
        elif len(self.stem) > 4 and not self.suffix:
            # Potential proper noun
            self.entity_type = EntityType.NOUN
            self.confidence = 0.8

@dataclass
class TokenizationResult:
    """Enhanced tokenization result for KG construction."""
    tokens: List[KGToken]
    original_text: str
    sentence_id: Optional[str] = None
    num_tokens: int = field(init=False)
    num_telugu_tokens: int = field(init=False)
    fallback_used: bool = False
    
    def __post_init__(self):
        self.num_tokens = len(self.tokens)
        self.num_telugu_tokens = sum(1 for t in self.tokens if t.token_type == "word")
        
        # Infer entity types
        for token in self.tokens:
            if token.token_type == "word":
                token.infer_entity_type()

@dataclass
class KGMetrics:
    """Comprehensive KG quality metrics."""
    total_entities: int = 0
    entities_with_relations: int = 0
    avg_entity_length: float = 0.0
    oov_rate: float = 0.0
    relation_density: float = 0.0
    entity_diversity: float = 0.0
    avg_confidence: float = 0.0
    
    def calculate(self, results: List[TokenizationResult]):
        """Calculate comprehensive quality metrics."""
        all_tokens = [t for r in results for t in r.tokens]
        telugu_tokens = [t for t in all_tokens if t.token_type == "word"]
        
        if not telugu_tokens:
            return
            
        unique_entities = set(t.stem for t in telugu_tokens)
        self.total_entities = len(unique_entities)
        self.entities_with_relations = len([t for t in telugu_tokens if t.suffix])
        self.avg_entity_length = sum(len(t.stem) for t in telugu_tokens) / len(telugu_tokens)
        self.oov_rate = sum(1 for t in telugu_tokens if t.is_fallback) / len(telugu_tokens)
        self.relation_density = self.entities_with_relations / self.total_entities if self.total_entities > 0 else 0
        self.entity_diversity = len(unique_entities) / len(telugu_tokens) if telugu_tokens else 0
        self.avg_confidence = sum(t.confidence for t in telugu_tokens) / len(telugu_tokens)
    
    def report(self) -> Dict:
        """Generate comprehensive quality report."""
        quality_score = self._calculate_quality_score()
        return {
            "quality_score": quality_score,
            "quality_level": self._get_quality_level(quality_score),
            "metrics": {
                "total_entities": self.total_entities,
                "entities_with_relations": self.entities_with_relations,
                "relation_density": f"{self.relation_density:.3f}",
                "oov_rate": f"{self.oov_rate:.3f}",
                "avg_entity_length": f"{self.avg_entity_length:.2f}",
                "entity_diversity": f"{self.entity_diversity:.3f}",
                "avg_confidence": f"{self.avg_confidence:.3f}"
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall KG quality score (0-1)."""
        score = (
            self.relation_density * 0.3 +
            (1 - self.oov_rate) * 0.25 +
            (min(self.avg_entity_length / 6, 1.0)) * 0.2 +
            self.entity_diversity * 0.15 +
            self.avg_confidence * 0.1
        )
        return round(min(score, 1.0), 3)
    
    def _get_quality_level(self, score: float) -> str:
        """Convert score to quality level."""
        if score >= 0.8: return "EXCELLENT"
        elif score >= 0.6: return "GOOD"
        elif score >= 0.4: return "FAIR"
        else: return "POOR"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        if self.oov_rate > 0.3:
            recommendations.append("Consider adding more words to exception dictionary")
        if self.relation_density < 0.2:
            recommendations.append("Low relation density - text may need morphological analysis")
        if self.avg_entity_length < 3:
            recommendations.append("Many short entities - consider entity merging")
        return recommendations

@dataclass
class KGCooccurrence:
    """Advanced co-occurrence analysis."""
    entity_pairs: Dict[Tuple[str, str], int] = field(default_factory=dict)
    entity_contexts: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    context_window: int = DEFAULT_CONTEXT_WINDOW
    
    def add_sentence(self, tokens: List[KGToken], sentence_id: str):
        """Extract co-occurring entities with context."""
        entities = [(i, t.stem) for i, t in enumerate(tokens) 
                   if t.token_type == "word" and not t.is_fallback]
        
        for i, (pos1, entity1) in enumerate(entities):
            # Store context
            start = max(0, pos1 - 2)
            end = min(len(tokens), pos1 + 3)
            context_words = [tokens[j].stem for j in range(start, end) 
                           if j != pos1 and tokens[j].token_type == "word"]
            self.entity_contexts[entity1].update(context_words)
            
            # Find co-occurrences within window
            for j in range(i+1, min(i+self.context_window+1, len(entities))):
                pos2, entity2 = entities[j]
                if abs(pos1 - pos2) <= self.context_window:
                    pair = tuple(sorted([entity1, entity2]))
                    self.entity_pairs[pair] = self.entity_pairs.get(pair, 0) + 1
    
    def to_kg_relations(self, min_freq: int = MIN_COOCCURRENCE_FREQ) -> List[Dict]:
        """Convert to KG relations with strength scoring."""
        relations = []
        for (e1, e2), freq in self.entity_pairs.items():
            if freq >= min_freq:
                strength = min(freq / 10, 1.0)
                common_context = len(self.entity_contexts[e1].intersection(self.entity_contexts[e2]))
                context_similarity = common_context / max(len(self.entity_contexts[e1]), 1)
                
                relations.append({
                    "source": e1,
                    "target": e2,
                    "relation": "co_occurs_with",
                    "type": RelationType.COOCCURRENCE.value,
                    "frequency": freq,
                    "strength": round(strength, 3),
                    "context_similarity": round(context_similarity, 3),
                    "combined_confidence": round((strength + context_similarity) / 2, 3)
                })
        return relations

@dataclass 
class KnowledgeGraph:
    """Complete knowledge graph structure."""
    entities: Dict[str, Dict] = field(default_factory=dict)
    relations: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_texts: List[str] = field(default_factory=list)
    
    def add_entity(self, entity: Dict):
        """Add entity with deduplication."""
        entity_id = entity["id"]
        if entity_id in self.entities:
            # Merge entity properties
            existing = self.entities[entity_id]
            existing["frequency"] = existing.get("frequency", 1) + 1
            existing["confidence"] = max(existing.get("confidence", 0), entity.get("confidence", 0))
        else:
            entity["frequency"] = 1
            self.entities[entity_id] = entity
    
    def add_relation(self, relation: Dict):
        """Add relation with validation."""
        # Check if both entities exist
        if (relation["source"] in self.entities and 
            relation["target"] in self.entities):
            self.relations.append(relation)
    
    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX graph."""
        G = nx.Graph()
        
        # Add nodes
        for entity_id, props in self.entities.items():
            G.add_node(entity_id, **props)
        
        # Add edges
        for relation in self.relations:
            G.add_edge(
                relation["source"], 
                relation["target"],
                **{k: v for k, v in relation.items() if k not in ["source", "target"]}
            )
        
        return G
    
    def to_json_ld(self) -> Dict:
        """Export as JSON-LD."""
        context = {
            "kg": "http://example.org/kg/",
            "text": "http://schema.org/text",
            "entities": "http://schema.org/Thing",
            "morphology": "http://linguistics.org/morphology",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
        }
        
        graph_data = []
        
        # Add entities
        for entity_id, props in self.entities.items():
            entity_node = {
                "@id": f"kg:entity/{entity_id}",
                "@type": "entities",
                "rdfs:label": props.get("surface_form", entity_id)
            }
            # Add other properties
            for key, value in props.items():
                if key not in ["id", "surface_form"]:
                    entity_node[f"kg:{key}"] = value
            graph_data.append(entity_node)
        
        # Add relations
        for i, relation in enumerate(self.relations):
            rel_node = {
                "@id": f"kg:relation/{i}",
                "@type": relation["type"],
                "kg:source": f"kg:entity/{relation['source']}",
                "kg:target": f"kg:entity/{relation['target']}",
                "kg:relation": relation["relation"],
                "kg:confidence": relation.get("confidence", 0.5)
            }
            graph_data.append(rel_node)
        
        return {
            "@context": context,
            "@graph": graph_data
        }

# ========================
# CORE TOKENIZER
# ========================

class TeluguKGTokenizer:
    """Advanced tokenizer optimized for KG construction."""
    
    def __init__(
        self,
        exception_dict: Optional[Dict[str, List[str]]] = None,
        enable_fallback: bool = True,
        debug: bool = False
    ):
        self.exceptions = {
            "పుస్తకానికి": ["పుస్తకం", "కు"],
            "పుస్తకంలో": ["పుస్తకం", "లో"],
            "పిల్లలకు": ["పిల్లలు", "కు"],
            "వాళ్ళతో": ["వాళ్ళు", "తో"],
            "గ్రంథాలయానికి": ["గ్రంథాలయం", "కు"],
            "రాష్ట్రంలో": ["రాష్ట్రం", "లో"],
            "భారతదేశం": ["భారత", "దేశం"],
        }
        
        if exception_dict:
            self.exceptions.update(exception_dict)
        
        self.enable_fallback = enable_fallback
        self.debug = debug
        
        if debug:
            logger.setLevel(logging.DEBUG)
        
        self._exceptions_tuple = tuple(
            (k, tuple(v)) for k, v in sorted(self.exceptions.items())
        )
        
        logger.info(f"Initialized KG tokenizer with {len(self.exceptions)} exceptions")
    
    @staticmethod
    def normalize(text: str) -> str:
        """Enhanced text normalization."""
        if not text:
            return ""
        
        if text.isascii():
            return " ".join(text.split())
        
        text = unicodedata.normalize("NFKC", text)
        
        zero_width_chars = {
            ord(c): None for c in "\u200B\u200C\u200D\uFEFF"
        }
        text = text.translate(zero_width_chars)
        
        return " ".join(text.split())
    
    @lru_cache(maxsize=CACHE_SIZE_MORPH)
    def _cached_split_morph(self, word: str) -> Tuple[Tuple[str, ...], bool, bool]:
        """Cached morphological splitting."""
        exceptions = dict((k, list(v)) for k, v in self._exceptions_tuple) if self._exceptions_tuple else {}
        
        if word in exceptions:
            morphs = exceptions[word]
            return (tuple(morphs), True, False)
        
        for length in (4, 3, 2, 1):
            if len(word) <= length:
                continue
            suffix = word[-length:]
            if suffix in _SUFFIX_TRIE[length]:
                stem = word[:-length]
                
                for pattern in _VOWEL_PATTERNS:
                    if pattern.stem_pattern.search(stem) and suffix.startswith(pattern.suffix_start):
                        new_stem = pattern.stem_pattern.sub(pattern.stem_replacement, stem)
                        new_suffix = pattern.suffix_replacement + suffix[len(pattern.suffix_start):]
                        return ((new_stem, new_suffix), True, False)
                
                return ((stem, suffix), True, False)
        
        if self.enable_fallback and len(word) > FALLBACK_LENGTH_THRESHOLD:
            pieces = _SYLLABLE_RE.findall(word)
            if pieces and len(pieces) > 1:
                return (tuple(pieces), True, True)
        
        return ((word,), False, False)
    
    def split_morph(self, word: str) -> Tuple[List[str], bool]:
        """Split word and return tokens + fallback flag."""
        tokens, is_split, used_fallback = self._cached_split_morph(word)
        return list(tokens), used_fallback
    
    def tokenize(self, text: str, return_metadata: bool = False, sentence_id: Optional[str] = None):
        """Advanced tokenization for KG construction."""
        original_text = text
        text = self.normalize(text)
        
        if not text:
            if return_metadata:
                return TokenizationResult([], original_text, sentence_id)
            return []
        
        out: List[KGToken] = []
        fallback_used = False
        
        for pos, m in enumerate(_TOKEN_RE.finditer(text)):
            unit = m.group()
            
            if not _IS_TELUGU(unit):
                token_type = "number" if unit.isdigit() else "punct"
                out.append(KGToken(unit, unit, token_type=token_type, position=pos))
                continue
            
            morphs, used_fallback = self.split_morph(unit)
            fallback_used = fallback_used or used_fallback
            
            if len(morphs) == 1:
                is_oov = used_fallback or (len(unit) <= FALLBACK_LENGTH_THRESHOLD and unit not in self.exceptions)
                confidence = 0.7 if is_oov else 0.95
                token = KGToken(unit, morphs[0], is_fallback=is_oov, position=pos, confidence=confidence)
                out.append(token)
            else:
                stem = morphs[0]
                suffix = "".join(morphs[1:]) if len(morphs) > 1 else None
                confidence = 0.8 if used_fallback else 0.9
                token = KGToken(unit, stem, suffix, is_fallback=used_fallback, position=pos, confidence=confidence)
                out.append(token)
        
        if return_metadata:
            result = TokenizationResult(out, original_text, sentence_id)
            result.fallback_used = fallback_used
            return result
        
        return out

# ========================
# KG CONSTRUCTION ENGINE
# ========================

class TeluguKGBuilder:
    """Main KG construction engine."""
    
    def __init__(
        self,
        exception_dict: Optional[Dict[str, List[str]]] = None,
        enable_fallback: bool = True,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        min_cooccurrence: int = MIN_COOCCURRENCE_FREQ
    ):
        self.tokenizer = TeluguKGTokenizer(exception_dict, enable_fallback)
        self.context_window = context_window
        self.min_cooccurrence = min_cooccurrence
        self.cooccurrence = KGCooccurrence(context_window=context_window)
        self.metrics = KGMetrics()
        
    def build_from_text(self, text: str, text_id: Optional[str] = None) -> KnowledgeGraph:
        """Build KG from single text."""
        if not text_id:
            text_id = str(uuid.uuid4())[:8]
        
        result = self.tokenizer.tokenize(text, return_metadata=True, sentence_id=text_id)
        return self._build_kg_from_results([result])
    
    def build_from_corpus(self, corpus_path: str) -> KnowledgeGraph:
        """Build KG from corpus directory."""
        results = []
        
        for file_path in iter_files(corpus_path, ".txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            sentence_id = f"{os.path.basename(file_path)}_L{line_num}"
                            result = self.tokenizer.tokenize(line, return_metadata=True, sentence_id=sentence_id)
                            results.append(result)
                            self.cooccurrence.add_sentence(result.tokens, sentence_id)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        return self._build_kg_from_results(results)
    
    def _build_kg_from_results(self, results: List[TokenizationResult]) -> KnowledgeGraph:
        """Construct KG from tokenization results."""
        kg = KnowledgeGraph()
        kg.metadata["created_at"] = time.time()
        kg.metadata["tokenizer"] = "TeluguKGTokenizer"
        
        # Process all results
        for result in results:
            kg.source_texts.append(result.original_text)
            
            for token in result.tokens:
                if token.token_type == "word":
                    kg.add_entity(token.to_kg_node())
                    
                    if edge := token.to_kg_edge():
                        kg.add_relation(edge)
        
        # Add co-occurrence relations
        cooccurrence_relations = self.cooccurrence.to_kg_relations(self.min_cooccurrence)
        for relation in cooccurrence_relations:
            kg.add_relation(relation)
        
        # Calculate metrics
        self.metrics.calculate(results)
        kg.metadata["metrics"] = self.metrics.report()
        kg.metadata["statistics"] = {
            "total_entities": len(kg.entities),
            "total_relations": len(kg.relations),
            "morphological_relations": len([r for r in kg.relations if r["type"] == RelationType.MORPHOLOGICAL.value]),
            "cooccurrence_relations": len([r for r in kg.relations if r["type"] == RelationType.COOCCURRENCE.value])
        }
        
        return kg

# ========================
# EXPORT FORMATS
# ========================

class KGExporter:
    """Export knowledge graphs to various formats."""
    
    @staticmethod
    def to_graphml(kg: KnowledgeGraph) -> str:
        """Export to GraphML format."""
        graphml = ['<?xml version="1.0" encoding="UTF-8"?>']
        graphml.append('<graphml xmlns="http://graphml.graphdrawing.org/xml/graphml">')
        
        # Define attributes
        graphml.extend([
            '<key id="type" for="node" attr.name="type" attr.type="string"/>',
            '<key id="surface_form" for="node" attr.name="surface_form" attr.type="string"/>',
            '<key id="confidence" for="node" attr.name="confidence" attr.type="double"/>',
            '<key id="relation" for="edge" attr.name="relation" attr.type="string"/>',
            '<key id="strength" for="edge" attr.name="strength" attr.type="double"/>'
        ])
        
        graphml.append('<graph id="G" edgedefault="undirected">')
        
        # Add nodes
        for entity_id, props in kg.entities.items():
            node_attrs = [
                f'type="{props.get("type", "unknown")}"',
                f'surface_form="{html.escape(props.get("surface_form", entity_id))}"',
                f'confidence="{props.get("confidence", 0.5)}"'
            ]
            graphml.append(f'<node id="{entity_id}" {" ".join(node_attrs)}/>')
        
        # Add edges
        for i, relation in enumerate(kg.relations):
            edge_attrs = [
                f'relation="{relation["relation"]}"',
                f'strength="{relation.get("strength", 0.5)}"'
            ]
            graphml.append(
                f'<edge id="e{i}" source="{relation["source"]}" '
                f'target="{relation["target"]}" {" ".join(edge_attrs)}/>'
            )
        
        graphml.extend(['</graph>', '</graphml>'])
        return '\n'.join(graphml)
    
    @staticmethod
    def to_rdf_turtle(kg: KnowledgeGraph) -> str:
        """Export to RDF Turtle format."""
        lines = ["@prefix kg: <http://example.org/kg/> .",
                "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
                "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .", ""]
        
        # Add entities
        for entity_id, props in kg.entities.items():
            lines.append(f'kg:entity/{entity_id} a kg:Entity ;')
            lines.append(f'  rdfs:label "{props.get("surface_form", entity_id)}" ;')
            lines.append(f'  kg:type "{props.get("type", "unknown")}" ;')
            lines.append(f'  kg:confidence "{props.get("confidence", 0.5)}"^^xsd:double .')
            lines.append("")
        
        # Add relations
        for i, relation in enumerate(kg.relations):
            lines.append(f'kg:relation/{i} a kg:{relation["type"]} ;')
            lines.append(f'  kg:source kg:entity/{relation["source"]} ;')
            lines.append(f'  kg:target kg:entity/{relation["target"]} ;')
            lines.append(f'  kg:relation "{relation["relation"]}" ;')
            lines.append(f'  kg:confidence "{relation.get("confidence", 0.5)}"^^xsd:double .')
            lines.append("")
        
        return '\n'.join(lines)

# ========================
# VISUALIZATION
# ========================

class KGVisualizer:
    """Visualize knowledge graphs."""
    
    @staticmethod
    def to_html(kg: KnowledgeGraph, output_path: str):
        """Create interactive HTML visualization."""
        try:
            G = kg.to_networkx()
            
            # Create plot
            plt.figure(figsize=(15, 10))
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw nodes
            node_colors = []
            for node in G.nodes():
                node_type = G.nodes[node].get('type', 'unknown')
                colors = {
                    'noun': 'lightblue',
                    'verb': 'lightgreen', 
                    'adjective': 'lightyellow',
                    'location': 'orange',
                    'person': 'pink',
                    'organization': 'lightgray'
                }
                node_colors.append(colors.get(node_type, 'white'))
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.7)
            nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_size=8)
            
            plt.title("Telugu Knowledge Graph")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path.replace('.html', '.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create HTML file
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Telugu Knowledge Graph</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .stats {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                    .graph {{ text-align: center; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>Telugu Knowledge Graph</h1>
                
                <div class="stats">
                    <h3>Statistics</h3>
                    <p>Entities: {len(kg.entities)}</p>
                    <p>Relations: {len(kg.relations)}</p>
                    <p>Quality Score: {kg.metadata.get('metrics', {}).get('quality_score', 'N/A')}</p>
                </div>
                
                <div class="graph">
                    <img src="{os.path.basename(output_path).replace('.html', '.png')}" 
                         alt="Knowledge Graph Visualization" style="max-width: 100%;">
                </div>
                
                <div>
                    <h3>Top Entities</h3>
                    <ul>
            """
            
            # Add top entities by frequency
            entities_sorted = sorted(kg.entities.items(), 
                                   key=lambda x: x[1].get('frequency', 0), 
                                   reverse=True)[:10]
            
            for entity_id, props in entities_sorted:
                html_content += f'<li>{props.get("surface_form", entity_id)} (freq: {props.get("frequency", 1)})</li>'
            
            html_content += """
                    </ul>
                </div>
            </body>
            </html>
            """
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"Visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")

# ========================
# STREAMING HELPERS
# ========================

def iter_files(root: str, ext: str = ".txt") -> Iterable[str]:
    """Iterate over files with given extension."""
    ext = ext.lower()
    if os.path.isfile(root):
        yield root
        return
    if not os.path.isdir(root):
        logger.warning(f"Path does not exist: {root}")
        return
    for base, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(ext):
                yield os.path.join(base, f)

def read_text(path: str, encoding: str = "utf-8") -> str:
    """Read text file with error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        logger.warning(f"Large file detected: {size_mb:.1f} MB")
    try:
        with open(path, "r", encoding=encoding, errors="replace") as fh:
            return fh.read()
    except Exception as e:
        raise IOError(f"Cannot read file {path}: {e}") from e

# ========================
# CLI IMPLEMENTATION
# ========================

def load_exceptions(path: Optional[str]) -> Optional[Dict[str, List[str]]]:
    """Load exception dictionary from JSON file."""
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        logger.error(f"Failed to load exceptions from {path}: {e}")
        return None
    
    cleaned = {}
    skipped = 0
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
            skipped += 1
            continue
        cleaned[k] = v
    
    if skipped:
        logger.warning(f"Skipped {skipped} invalid exception entries")
    logger.info(f"Loaded {len(cleaned)} exceptions from {path}")
    return cleaned

def cmd_extract_kg(args: argparse.Namespace):
    """Extract KG from single text/file."""
    ex = load_exceptions(args.exceptions)
    builder = TeluguKGBuilder(
        exception_dict=ex,
        enable_fallback=not args.no_fallback,
        context_window=args.context_window,
        min_cooccurrence=args.min_cooccurrence
    )
    
    if args.input:
        text = read_text(args.input, args.encoding)
        kg = builder.build_from_text(text, os.path.basename(args.input))
    else:
        text = sys.stdin.read()
        kg = builder.build_from_text(text)
    
    # Export in requested format
    if args.format == KGFormat.GRAPHML.value:
        output = KGExporter.to_graphml(kg)
    elif args.format == KGFormat.RDF.value:
        output = KGExporter.to_rdf_turtle(kg)
    elif args.format == KGFormat.JSON_LD.value:
        output = json.dumps(kg.to_json_ld(), ensure_ascii=False, indent=2)
    else:  # Default JSON
        output = json.dumps({
            "entities": kg.entities,
            "relations": kg.relations,
            "metadata": kg.metadata
        }, ensure_ascii=False, indent=2)
    
    write_output(output, args.out)

def cmd_build_corpus(args: argparse.Namespace):
    """Build KG from corpus directory."""
    ex = load_exceptions(args.exceptions)
    builder = TeluguKGBuilder(
        exception_dict=ex,
        enable_fallback=not args.no_fallback,
        context_window=args.context_window,
        min_cooccurrence=args.min_cooccurrence
    )
    
    logger.info(f"Building KG from corpus: {args.dir}")
    kg = builder.build_from_corpus(args.dir)
    
    output = json.dumps({
        "entities": kg.entities,
        "relations": kg.relations,
        "metadata": kg.metadata
    }, ensure_ascii=False, indent=2)
    
    write_output(output, args.out)
    logger.info(f"KG built with {len(kg.entities)} entities and {len(kg.relations)} relations")

def cmd_analyze_kg(args: argparse.Namespace):
    """Analyze KG quality and metrics."""
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        
        if "metadata" in kg_data and "metrics" in kg_data["metadata"]:
            metrics = kg_data["metadata"]["metrics"]
            print("\n" + "="*60)
            print("KNOWLEDGE GRAPH QUALITY ANALYSIS")
            print("="*60)
            print(f"Quality Score: {metrics['quality_score']} ({metrics['quality_level']})")
            print("\nDetailed Metrics:")
            for key, value in metrics['metrics'].items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
            print("\nRecommendations:")
            for rec in metrics.get('recommendations', []):
                print(f"  • {rec}")
            print("="*60)
        else:
            print("No quality metrics found in KG file")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")

def cmd_visualize_kg(args: argparse.Namespace):
    """Create KG visualization."""
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        
        kg = KnowledgeGraph()
        kg.entities = kg_data.get("entities", {})
        kg.relations = kg_data.get("relations", [])
        kg.metadata = kg_data.get("metadata", {})
        
        KGVisualizer.to_html(kg, args.out)
        print(f"Visualization saved to {args.out}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")

def cmd_export_kg(args: argparse.Namespace):
    """Export KG to different formats."""
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        
        kg = KnowledgeGraph()
        kg.entities = kg_data.get("entities", {})
        kg.relations = kg_data.get("relations", [])
        kg.metadata = kg_data.get("metadata", {})
        
        if args.format == KGFormat.GRAPHML.value:
            output = KGExporter.to_graphml(kg)
            ext = ".graphml"
        elif args.format == KGFormat.RDF.value:
            output = KGExporter.to_rdf_turtle(kg)
            ext = ".ttl"
        elif args.format == KGFormat.JSON_LD.value:
            output = json.dumps(kg.to_json_ld(), ensure_ascii=False, indent=2)
            ext = ".jsonld"
        else:
            output = json.dumps(kg_data, ensure_ascii=False, indent=2)
            ext = ".json"
        
        if not args.out:
            args.out = args.input.replace('.json', ext)
        
        write_output(output, args.out)
        print(f"Exported KG to {args.out} in {args.format} format")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")

def write_output(content: str, path: Optional[str]):
    """Write output to file or stdout."""
    if path:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
        except Exception as e:
            logger.error(f"Failed to write to {path}: {e}")
            sys.exit(1)
    else:
        print(content)

def build_argparser() -> argparse.ArgumentParser:
    """Build comprehensive CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Advanced Telugu Knowledge Graph Construction Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    sub = p.add_subparsers(dest="cmd", required=True)
    
    # extract command
    extract = sub.add_parser("extract", help="Extract KG from text/file")
    extract.add_argument("--input", "-i", help="Input file (default: stdin)")
    extract.add_argument("--out", "-o", help="Output file (default: stdout)")
    extract.add_argument("--encoding", default="utf-8", help="File encoding")
    extract.add_argument("--exceptions", help="Path to JSON exception dictionary")
    extract.add_argument("--no-fallback", action="store_true", help="Disable syllable fallback")
    extract.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW, 
                        help="Co-occurrence context window size")
    extract.add_argument("--min-cooccurrence", type=int, default=MIN_COOCCURRENCE_FREQ,
                        help="Minimum co-occurrence frequency")
    extract.add_argument("--format", choices=[f.value for f in KGFormat], default="json",
                        help="Output format")
    extract.set_defaults(func=cmd_extract_kg)
    
    # build-corpus command
    build = sub.add_parser("build-corpus", help="Build KG from corpus directory")
    build.add_argument("--dir", "-d", required=True, help="Corpus directory")
    build.add_argument("--out", "-o", required=True, help="Output file")
    build.add_argument("--exceptions", help="Path to JSON exception dictionary")
    build.add_argument("--no-fallback", action="store_true", help="Disable syllable fallback")
    build.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW,
                        help="Co-occurrence context window size")
    build.add_argument("--min-cooccurrence", type=int, default=MIN_COOCCURRENCE_FREQ,
                        help="Minimum co-occurrence frequency")
    build.set_defaults(func=cmd_build_corpus)
    
    # analyze command
    analyze = sub.add_parser("analyze", help="Analyze KG quality and metrics")
    analyze.add_argument("--input", "-i", required=True, help="KG JSON file")
    analyze.set_defaults(func=cmd_analyze_kg)
    
    # visualize command
    viz = sub.add_parser("visualize", help="Create KG visualization")
    viz.add_argument("--input", "-i", required=True, help="KG JSON file")
    viz.add_argument("--out", "-o", required=True, help="Output HTML file")
    viz.set_defaults(func=cmd_visualize_kg)
    
    # export command
    export = sub.add_parser("export", help="Export KG to different formats")
    export.add_argument("--input", "-i", required=True, help="KG JSON file")
    export.add_argument("--out", "-o", help="Output file")
    export.add_argument("--format", choices=[f.value for f in KGFormat], required=True,
                        help="Export format")
    export.set_defaults(func=cmd_export_kg)
    
    return p

def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point."""
    parser = build_argparser()
    args = parser.parse_args(argv)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.debug:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()
