import os
import seaborn as sns
import networkx as nx
from typing import List
import matplotlib.pyplot as plt
from pyvis.network import Network

from hats.utils.const import HATS_ROOT
from hats.topology.connection import Connection

def generate_color_map(node_ids):
    color_palette = sns.color_palette("husl", len(node_ids)).as_hex()
    color_map = {node_id: color_palette[i % len(color_palette)] for i, node_id in enumerate(node_ids)}
    return color_map

def visualize_topology(
    system,
    experiment_name: str, 
    record_time: str, 
    mask_decision:bool=False, 
    style="pyvis"
):
    assert len(system.agents) != 0, "please organize the system"
    assert hasattr(system, 'edgelist'), "please give the edgelist of the system"

    run_path = os.path.join(HATS_ROOT, "vis")
    experiment_path = os.path.join(run_path, experiment_name)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    if style == 'pyvis':
        file_path = os.path.join(experiment_path, record_time) + ".html"
    else:
        file_path = os.path.join(experiment_path, record_time) + ".png"

    agent_ids = list(system.agents.keys())
    if not mask_decision:
        agent_ids.append(system.decision_agent.id)
    node2color = generate_color_map(agent_ids)

    edge2color = {
        "Sequential_Connection": "#2E54A1",
        "Decentralized_Connection": "#F2664D"
    }

    if style == "pyvis":
        net = Network(notebook=True, height="750px", width="100%", bgcolor="#FFFFFF", font_color="black", directed=True)
        for agent_id, agent in system.agents.items():
            net.add_node(agent_id, label=f"{agent.agent_name}\n(ID: {agent_id})", color=node2color[agent_id])
        if not mask_decision:
            color_key = system.decision_agent.id
            net.add_node(color_key, label=f"FinalDecision\n(ID: {color_key})", color=node2color[color_key])

        for edge in system.edgelist:
            src, dst, connection_type = edge.src, edge.dst, edge.connection_type
            if mask_decision:
                if dst == system.decision_agent.id:
                    continue
            net.add_edge(src, dst, color=edge2color[connection_type])
            
        net.show(file_path)

    else:
        G = nx.DiGraph()
        for agent_id, agent in system.agents.items():
            G.add_node(agent_id, label=f"{agent.agent_name}\n(ID: {agent_id})")
        if not mask_decision:
            G.add_node(system.decision_agent.id, label=f"FinalDecision\n(ID: {system.decision_agent.id})")
        
        edge_colors = []
        for edge in system.edgelist:
            src, dst, connection_type = edge.src, edge.dst, edge.connection_type
            if mask_decision:
                if dst == system.decision_agent.id:
                    continue
            G.add_edge(src, dst)
            edge_colors.append(edge2color[connection_type])
        
        pos = nx.spring_layout(G, k=.3, iterations=30)
        node_colors = [node2color[node] for node in G.nodes()]
        node_sizes = [2500 for node in G.nodes()]
        plt.figure(figsize=(12, 12))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.93)

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.6, node_shape='o',
                               edgecolors='black', linewidths=1.5)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrowstyle='-|>', node_size=node_sizes, arrowsize=20)
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, font_family='sans-serif', font_weight='bold',
                                font_color='blue')

        plt.title(f"System", size=20, color='darkblue', fontweight='bold', fontfamily='sans-serif')
        plt.axis('off')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.savefig(file_path)

