"""
Codebase Analyzer for MOTHER

This module analyzes MOTHER's codebase to understand:
- Architecture and components
- Function definitions and their purposes
- Data flow (how data is stored/retrieved)
- Memory system implementations
- File structure and organization
"""

import ast
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import inspect

logger = logging.getLogger(__name__)


class CodebaseAnalyzer:
    """Analyzes MOTHER's codebase to extract architectural knowledge"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.analyzed_files: Dict[str, Dict[str, Any]] = {}
        self.architecture: Dict[str, Any] = {
            "modules": {},
            "classes": {},
            "functions": {},
            "data_flow": {},
            "memory_systems": {},
            "file_structure": {}
        }
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze the entire codebase"""
        logger.info("[Codebase Analyzer] Starting codebase analysis...")
        
        # Analyze key directories
        key_dirs = [
            "memory",
            "processing",
            "personality",
            "reflection",
            "utils",
            "routes.py",
            "startup.py"
        ]
        
        for item in key_dirs:
            path = self.project_root / item
            if path.is_file():
                self._analyze_file(path)
            elif path.is_dir():
                self._analyze_directory(path)
        
        # Extract architecture patterns
        self._extract_architecture()
        self._extract_data_flow()
        self._extract_memory_systems()
        
        logger.info(f"[Codebase Analyzer] Analysis complete. Analyzed {len(self.analyzed_files)} files.")
        return self.architecture
    
    def _analyze_directory(self, directory: Path):
        """Analyze all Python files in a directory"""
        for py_file in directory.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                self._analyze_file(py_file)
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            file_info = {
                "path": str(file_path.relative_to(self.project_root)),
                "module_name": file_path.stem,
                "classes": [],
                "functions": [],
                "imports": [],
                "docstring": ast.get_docstring(tree),
                "file_purpose": self._extract_file_purpose(content, file_path)
            }
            
            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node, content)
                    file_info["classes"].append(class_info)
                    self.architecture["classes"][class_info["name"]] = class_info
                
                elif isinstance(node, ast.FunctionDef):
                    if not any(node.name == c["name"] for c in file_info["classes"] for node in ast.walk(ast.parse(content)) if isinstance(node, ast.FunctionDef)):
                        func_info = self._analyze_function(node, content)
                        file_info["functions"].append(func_info)
                        full_name = f"{file_info['module_name']}.{func_info['name']}"
                        self.architecture["functions"][full_name] = func_info
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._extract_import(node)
                    if import_info:
                        file_info["imports"].append(import_info)
            
            self.analyzed_files[str(file_path)] = file_info
            module_key = file_info["module_name"]
            self.architecture["modules"][module_key] = file_info
            
        except Exception as e:
            logger.error(f"[Codebase Analyzer] Error analyzing {file_path}: {e}")
    
    def _analyze_class(self, node: ast.ClassDef, source_code: str) -> Dict[str, Any]:
        """Extract information about a class"""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item, source_code, is_method=True)
                methods.append(method_info)
        
        return {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "methods": methods,
            "base_classes": [base.id for base in node.bases if isinstance(base, ast.Name)],
            "purpose": self._extract_docstring_summary(ast.get_docstring(node))
        }
    
    def _analyze_function(self, node: ast.FunctionDef, source_code: str, is_method: bool = False) -> Dict[str, Any]:
        """Extract information about a function"""
        args = [arg.arg for arg in node.args.args]
        if is_method and args and args[0] == "self":
            args = args[1:]  # Remove self
        
        # Extract function body summary
        body_summary = self._extract_function_summary(node, source_code)
        
        return {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "args": args,
            "purpose": self._extract_docstring_summary(ast.get_docstring(node)),
            "summary": body_summary,
            "is_method": is_method
        }
    
    def _extract_function_summary(self, node: ast.FunctionDef, source_code: str) -> str:
        """Extract a summary of what the function does by analyzing its body"""
        lines = source_code.split('\n')
        func_start = node.lineno - 1
        func_end = node.end_lineno if hasattr(node, 'end_lineno') else func_start + 20
        
        func_body = '\n'.join(lines[func_start:func_end])
        
        # Look for key patterns
        summary_parts = []
        
        # Check for file operations
        if any(op in func_body for op in ['open(', 'save_to_file', 'load_from_file', 'json.dump', 'json.load']):
            summary_parts.append("file I/O operations")
        
        # Check for memory operations
        if any(op in func_body for op in ['add_memory', 'get_memory', 'search_memory', 'store', 'retrieve']):
            summary_parts.append("memory operations")
        
        # Check for graph operations
        if any(op in func_body for op in ['add_node', 'add_edge', 'get_node', 'graph']):
            summary_parts.append("knowledge graph operations")
        
        # Check for LLM calls
        if any(op in func_body for op in ['get_response', 'llm', 'groq', 'chat.completions']):
            summary_parts.append("LLM API calls")
        
        # Check for database operations
        if any(op in func_body for op in ['query', 'execute', 'commit', 'session']):
            summary_parts.append("database operations")
        
        return ", ".join(summary_parts) if summary_parts else "general processing"
    
    def _extract_import(self, node: ast.Import | ast.ImportFrom) -> Optional[Dict[str, str]]:
        """Extract import information"""
        if isinstance(node, ast.Import):
            return {"type": "import", "module": node.names[0].name if node.names else ""}
        elif isinstance(node, ast.ImportFrom):
            return {
                "type": "from_import",
                "module": node.module or "",
                "imports": [alias.name for alias in node.names]
            }
        return None
    
    def _extract_file_purpose(self, content: str, file_path: Path) -> str:
        """Extract the purpose of a file from its content and path"""
        # Check module docstring
        try:
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)
            if docstring:
                return docstring.split('\n')[0]  # First line
        except:
            pass
        
        # Infer from filename and content
        filename = file_path.stem
        
        if "memory" in filename.lower():
            if "vector" in filename.lower():
                return "Vector memory storage and semantic search"
            elif "structured" in filename.lower():
                return "Structured facts and knowledge graph storage"
            elif "episodic" in filename.lower():
                return "Episodic memory (conversation logs)"
            elif "knowledge" in filename.lower():
                return "Knowledge graph implementation"
            else:
                return "Memory system component"
        
        elif "processing" in filename.lower():
            if "llm" in filename.lower():
                return "LLM API handler (Groq)"
            elif "cognitive" in filename.lower():
                return "Cognitive agent orchestrator"
            elif "interpreter" in filename.lower():
                return "Universal interpreter for language understanding"
            else:
                return "Processing component"
        
        elif "personality" in filename.lower():
            return "Personality and identity formation"
        
        elif "reflection" in filename.lower():
            return "Reflection engine for self-awareness"
        
        elif filename == "routes":
            return "Flask API routes and chat endpoint"
        
        elif filename == "startup":
            return "System initialization and startup"
        
        return "System component"
    
    def _extract_docstring_summary(self, docstring: Optional[str]) -> str:
        """Extract a one-line summary from a docstring"""
        if not docstring:
            return ""
        first_line = docstring.split('\n')[0].strip()
        # Remove common prefixes
        for prefix in ["Returns", "Args", "Parameters"]:
            if first_line.startswith(prefix):
                return ""
        return first_line[:200]  # Limit length
    
    def _extract_architecture(self):
        """Extract high-level architecture patterns"""
        # Identify main components
        components = {
            "memory_layer": [],
            "processing_layer": [],
            "personality_layer": [],
            "api_layer": []
        }
        
        for module_name, module_info in self.architecture["modules"].items():
            path = module_info.get("path", "")
            
            if "memory" in path:
                components["memory_layer"].append(module_name)
            elif "processing" in path:
                components["processing_layer"].append(module_name)
            elif "personality" in path:
                components["personality_layer"].append(module_name)
            elif "routes" in path or "startup" in path:
                components["api_layer"].append(module_name)
        
        self.architecture["components"] = components
    
    def _extract_data_flow(self):
        """Extract how data flows through the system"""
        data_flow = {
            "storage": {},
            "retrieval": {},
            "processing": {}
        }
        
        # Analyze memory modules
        for module_name, module_info in self.architecture["modules"].items():
            if "memory" not in module_info.get("path", "").lower():
                continue
            
            # Find storage functions
            for func in module_info.get("functions", []):
                func_name = func["name"].lower()
                if any(keyword in func_name for keyword in ["save", "store", "add", "set", "write"]):
                    data_flow["storage"][f"{module_name}.{func['name']}"] = {
                        "purpose": func.get("purpose", ""),
                        "file_location": self._extract_file_location(func, module_info)
                    }
                elif any(keyword in func_name for keyword in ["get", "load", "read", "retrieve", "search", "find"]):
                    data_flow["retrieval"][f"{module_name}.{func['name']}"] = {
                        "purpose": func.get("purpose", ""),
                        "file_location": self._extract_file_location(func, module_info)
                    }
        
        self.architecture["data_flow"] = data_flow
    
    def _extract_file_location(self, func: Dict, module_info: Dict) -> str:
        """Extract where data is stored/retrieved from"""
        # This would require deeper analysis of the function body
        # For now, infer from module name
        path = module_info.get("path", "")
        
        if "vector" in path.lower():
            return "data/vector_memory/"
        elif "structured" in path.lower():
            return "data/structured_memory/"
        elif "episodic" in path.lower():
            return "data/episodic_memory/"
        elif "journal" in path.lower():
            return "data/journal/"
        elif "reflection" in path.lower():
            return "data/reflections/"
        elif "usage" in path.lower():
            return "data/usage_tracking/"
        else:
            return "data/"
    
    def _extract_memory_systems(self):
        """Extract detailed information about memory systems"""
        memory_systems = {}
        
        # Vector Memory
        if "vector_store" in self.architecture["modules"]:
            memory_systems["vector_memory"] = {
                "module": "memory.vector_store",
                "storage": "data/vector_memory/memories.json, embeddings.npy, vectorizer.pkl",
                "purpose": "Semantic search using TF-IDF vectorization",
                "key_functions": ["add_memory", "search_memory", "get_recent_memories"]
            }
        
        # Structured Memory
        if "structured_store" in self.architecture["modules"]:
            memory_systems["structured_memory"] = {
                "module": "memory.structured_store",
                "storage": "data/structured_memory/facts.json, knowledge_graph.json",
                "purpose": "Structured facts and knowledge graph relationships",
                "key_functions": ["set_fact", "get_fact", "all_facts", "get_knowledge_graph"]
            }
        
        # Episodic Memory
        if "episodic_logger" in self.architecture["modules"]:
            memory_systems["episodic_memory"] = {
                "module": "memory.episodic_logger",
                "storage": "data/episodic_memory/YYYY-MM-DD.json files",
                "purpose": "Daily conversation logs organized by date",
                "key_functions": ["log_event", "get_log_for_date", "get_all_conversation_dates"]
            }
        
        # Knowledge Graph
        if "knowledge_graph" in self.architecture["modules"]:
            memory_systems["knowledge_graph"] = {
                "module": "memory.knowledge_graph",
                "storage": "data/structured_memory/knowledge_graph.json",
                "purpose": "NetworkX-based graph of concepts and relationships",
                "key_functions": ["add_node", "add_edge", "get_node_by_name", "has_edge_between_names"]
            }
        
        self.architecture["memory_systems"] = memory_systems
    
    def get_architecture_summary(self) -> str:
        """Get a human-readable summary of the architecture"""
        summary_parts = []
        
        summary_parts.append("MOTHER Architecture:")
        summary_parts.append("\n1. Memory Layer:")
        for system_name, system_info in self.architecture.get("memory_systems", {}).items():
            summary_parts.append(f"   - {system_name}: {system_info.get('purpose', '')}")
            summary_parts.append(f"     Storage: {system_info.get('storage', '')}")
            summary_parts.append(f"     Module: {system_info.get('module', '')}")
        
        summary_parts.append("\n2. Processing Layer:")
        for module in self.architecture.get("components", {}).get("processing_layer", []):
            module_info = self.architecture["modules"].get(module, {})
            summary_parts.append(f"   - {module}: {module_info.get('file_purpose', '')}")
        
        summary_parts.append("\n3. Data Flow:")
        storage_funcs = list(self.architecture.get("data_flow", {}).get("storage", {}).keys())[:5]
        retrieval_funcs = list(self.architecture.get("data_flow", {}).get("retrieval", {}).keys())[:5]
        summary_parts.append(f"   Storage functions: {', '.join(storage_funcs)}")
        summary_parts.append(f"   Retrieval functions: {', '.join(retrieval_funcs)}")
        
        return "\n".join(summary_parts)

