"""
MOTHER Knowledge Graph Visualizer (Interactive HTML)

Visualizes MOTHER's knowledge graph with an interactive network visualization.
Similar to Axiom's visualize_brain.py but adapted for MOTHER's structure.

Features:
- Interactive HTML visualization (drag, zoom, hover)
- Color-coded nodes by type
- Node size based on connectivity
- Edge width based on confidence/weight
- Detailed tooltips for nodes and edges
- Performance optimization for large graphs
- Physics simulation toggle
"""

import json
import logging
from pathlib import Path
from typing import Optional

try:
    import networkx as nx
    from pyvis.network import Network
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("pyvis not available. Install with: pip install pyvis")

from memory.knowledge_graph import KnowledgeGraph

# Default knowledge graph file location
DEFAULT_GRAPH_FILE = Path("data/structured_memory/knowledge_graph.json")
OUTPUT_DIR = Path("visualizations")
OUTPUT_FILE = OUTPUT_DIR / "mother_knowledge_graph.html"

# Node type color mapping
COLOR_MAP = {
    "concept": "#ff66cc",
    "noun": "#ffcc00",
    "verb": "#00d8ff",
    "person": "#90ee90",
    "pet": "#ffaa00",
    "location": "#66b3ff",
    "module": "#ff6b6b",
    "function": "#4ecdc4",
    "file_location": "#95e1d3",
    "purpose": "#f38181",
    "word": "#a8e6cf",
    "unknown": "#888888",
}

# Default color for unknown types
DEFAULT_COLOR = "#888888"


def visualize_knowledge_graph(
    graph_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    max_nodes: int = 750,
    show_physics: bool = False,
) -> None:
    """
    Load MOTHER's knowledge graph and generate an interactive HTML visualization.
    
    Args:
        graph_file: Path to knowledge graph JSON file. Defaults to data/structured_memory/knowledge_graph.json
        output_file: Path for output HTML file. Defaults to visualizations/mother_knowledge_graph.html
        max_nodes: Maximum number of nodes to display (for performance). Shows most connected nodes.
        show_physics: Whether to enable physics simulation by default.
    """
    if not VISUALIZATION_AVAILABLE:
        print("ERROR: Visualization libraries not available.")
        print("   Install with: pip install pyvis networkx")
        return
    
    graph_file = graph_file or DEFAULT_GRAPH_FILE
    output_file = output_file or OUTPUT_FILE
    
    print(f"Loading knowledge graph from: {graph_file}")
    
    if not graph_file.exists():
        raise FileNotFoundError(
            f"ERROR: Knowledge graph file not found at:\n   {graph_file}\n"
            f"   Run MOTHER first to generate the knowledge graph."
        )
    
    # Load knowledge graph
    try:
        graph = KnowledgeGraph.load_from_file(str(graph_file))
    except Exception as e:
        raise ValueError(f"ERROR: Failed to load knowledge graph: {e}")
    
    nodes = graph.get_all_node_names()
    edges = graph.get_all_edges()
    
    print(f"Loaded knowledge graph: {len(nodes)} nodes, {len(edges)} edges")
    
    if not nodes:
        raise ValueError("WARNING: Knowledge graph is empty - nothing to visualize!")
    
    # Build NetworkX graph
    g = nx.MultiDiGraph()
    
    # Add nodes
    for node_name in nodes:
        node = graph.get_node_by_name(node_name)
        if node:
            g.add_node(
                node.id,
                name=node.name,
                node_type=node.type or "concept",
                value=node.value,
                activation=node.activation,
                properties=node.properties or {},
            )
    
    # Add edges
    for edge in edges:
        g.add_edge(
            edge.source,
            edge.target,
            relation_type=edge.type,
            weight=edge.weight,
            properties=edge.properties or {},
        )
    
    # Cull graph if too large
    if len(g.nodes) > max_nodes:
        print(
            f"WARNING: Graph is large ({len(g.nodes)} nodes). "
            f"Culling to the {max_nodes} most connected nodes for performance."
        )
        degrees = dict(g.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
        core_node_ids = {node_id for node_id, _ in sorted_nodes[:max_nodes]}
        
        core_g = nx.MultiDiGraph()
        for node_id in core_node_ids:
            if node_id in g.nodes:
                core_g.add_node(node_id, **g.nodes[node_id])
        
        for u, v, data in g.edges(data=True):
            if u in core_node_ids and v in core_node_ids:
                core_g.add_edge(u, v, **data)
        
        g = core_g
        print(f"Culled graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
    
    # Create pyvis network
    vis = Network(
        height="100vh",
        width="100%",
        bgcolor="#0d1117",
        font_color="#ffffff",
        notebook=False,
        directed=True,
    )
    
    # Configure physics and interaction
    options = {
        "physics": {
            "enabled": show_physics,
            "barnesHut": {
                "gravity": -40000,
                "centralGravity": 0.4,
                "springLength": 200,
                "springConstant": 0.02,
            },
            "maxVelocity": 50,
            "minVelocity": 0.75,
            "stabilization": {"enabled": True, "iterations": 1000, "fit": True},
        },
        "interaction": {
            "tooltipDelay": 200,
            "hideEdgesOnDrag": True,
            "hover": True,
        },
        "configure": {"enabled": False},
    }
    vis.set_options(json.dumps(options))
    
    # Calculate node sizes based on connectivity
    degrees = dict(g.degree())
    min_size, max_size = 15, 40
    min_degree = min(degrees.values()) if degrees else 0
    max_degree = max(degrees.values()) if degrees else 0
    
    def scale_node_size(degree):
        if max_degree == min_degree:
            return (min_size + max_size) / 2
        return min_size + (degree - min_degree) / (max_degree - min_degree) * (
            max_size - min_size
        )
    
    # Add nodes to visualization
    for node_id, data in g.nodes(data=True):
        node_type = data.get("node_type", "unknown")
        degree = degrees.get(node_id, 1)
        name = data.get("name", node_id)
        value = data.get("value", "")
        activation = data.get("activation", 0.0)
        properties = data.get("properties", {})
        
        # Build tooltip HTML
        props_str = json.dumps(properties, indent=2) if properties else "None"
        title_html = (
            f"<b>Name:</b> {name}<br>"
            f"<b>Type:</b> {node_type}<br>"
            f"<b>Connections:</b> {degree}<br>"
        )
        if value:
            title_html += f"<b>Value:</b> {value}<br>"
        title_html += f"<b>Activation:</b> {activation:.2f}<br>"
        title_html += f"<hr><b>Properties:</b><br><pre>{props_str}</pre>"
        
        vis.add_node(
            node_id,
            label=name,
            color=COLOR_MAP.get(node_type, DEFAULT_COLOR),
            title=title_html,
            size=scale_node_size(degree),
            group=node_type,
        )
    
    # Add edges to visualization
    for u, v, data in g.edges(data=True):
        relation_type = data.get("relation_type", "related_to")
        weight = float(data.get("weight", 0.5))
        properties = data.get("properties", {})
        
        # Edge styling
        edge_width = 1 + (weight * 4)  # Scale width by confidence
        is_negated = properties.get("negated", False)
        edge_color = "#e06c75" if is_negated else "#555555"
        dashes = is_negated
        
        # Build tooltip HTML
        props_str = json.dumps(properties, indent=2) if properties else "None"
        title_html = (
            f"<b>Relation:</b> {relation_type}<br>"
            f"<b>Confidence:</b> {weight:.2f}<br>"
            f"<hr><b>Properties:</b><br><pre>{props_str}</pre>"
        )
        
        vis.add_edge(
            u,
            v,
            title=title_html,
            label=relation_type,
            width=edge_width,
            color=edge_color,
            dashes=dashes,
        )
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Save HTML
    vis.write_html(str(output_file))
    
    # Inject physics toggle button
    html = output_file.read_text(encoding="utf-8")
    injection = """
    <style>
    #physicsToggle {
        position: absolute; top: 15px; left: 15px; z-index: 1000;
        background: #161b22; color: #c9d1d9; padding: 8px 12px;
        border: 1px solid #30363d; border-radius: 6px; cursor: pointer;
        font-size: 14px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }
    #physicsToggle:hover { background: #21262d; }
    </style>
    <button id="physicsToggle">Enable Physics</button>
    <script type="text/javascript">
        document.addEventListener("DOMContentLoaded", function() {
            const btn = document.getElementById('physicsToggle');
            let physicsEnabled = false;
            btn.onclick = function() {
                physicsEnabled = !physicsEnabled;
                network.setOptions({ physics: { enabled: physicsEnabled } });
                this.innerText = physicsEnabled ? 'Disable Physics' : 'Enable Physics';
            };
        });
    </script>
    """
    html = html.replace("</body>", injection + "\n</body>")
    output_file.write_text(html, encoding="utf-8")
    
    print("SUCCESS: Interactive visualization generated successfully!")
    print(f"File saved at:\n   {output_file.resolve()}")
    print("Open it in your browser to explore MOTHER's knowledge graph interactively.")


def main():
    """Entry point for the visualization script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize MOTHER's knowledge graph interactively"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_GRAPH_FILE,
        help=f"Path to knowledge graph file (default: {DEFAULT_GRAPH_FILE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output HTML file path (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=750,
        help="Maximum number of nodes to display (default: 750)",
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Enable physics simulation by default",
    )
    
    args = parser.parse_args()
    
    try:
        visualize_knowledge_graph(
            graph_file=args.file,
            output_file=args.output,
            max_nodes=args.max_nodes,
            show_physics=args.physics,
        )
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

