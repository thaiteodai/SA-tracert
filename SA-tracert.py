#!/usr/bin/env python3
"""
Enhanced Traceroute + Simulated Annealing Script with CSV Export, Graph Plotting, Detailed Logging

Chuc nang:
- Chay tracert/traceroute nhieu lan de tong hop du lieu latency
- Builds a weighted directed graph (average latency per edge)
- Simulated Annealing to find minimal-latency path
- CSV export of edge statistics & optimal path (using csv module)
- Plotting network graph with optimal path highlighted
- Detailed logging

Usage:
  python enhanced_traceroute_sa.py TARGET [--runs R] [--max-ttl T] [--timeout S] [--sa-* options] [--csv-dir DIR] [--no-plot]

Requires: scapy, networkx, matplotlib
Run on Windows as Administrator.
"""
import time
import random
import math
import argparse
import logging
import csv
import socket
from pathlib import Path

# Scapy imports and suppress ARP warnings
from scapy.all import IP, ICMP, sr1
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

import networkx as nx
import matplotlib.pyplot as plt

# Configure main logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def traceroute_scapy(dst_ip, max_ttl=30, timeout=2.0):
    logging.info(f"Traceroute to {dst_ip}: max_ttl={max_ttl}, timeout={timeout}s")
    path = []
    for ttl in range(1, max_ttl+1):
        pkt = IP(dst=dst_ip, ttl=ttl) / ICMP()
        start = time.time()
        reply = sr1(pkt, verbose=0, timeout=timeout)
        if reply:
            rtt = (time.time() - start) * 1000
            hop = reply.src
            logging.debug(f"TTL={ttl} hop={hop} rtt={rtt:.2f}ms")
            path.append((hop, round(rtt, 2)))
            if getattr(reply, 'type', None) == 0:
                logging.info("Destination reached")
                break
        else:
            logging.debug(f"TTL={ttl} no reply")
            path.append((None, None))
    return path


def aggregate_paths(runs, dst_ip, max_ttl, timeout):
    edge_data = {}
    for i in range(runs):
        logging.info(f"Run {i+1}/{runs}")
        path = traceroute_scapy(dst_ip, max_ttl, timeout)
        prev = None
        for hop, rtt in path:
            if prev and hop and rtt is not None:
                edge_data.setdefault((prev, hop), []).append(rtt)
            prev = hop
    return edge_data


def build_graph(edge_data):
    G = nx.DiGraph()
    for (u, v), rtts in edge_data.items():
        avg = sum(rtts) / len(rtts)
        G.add_edge(u, v, weight=round(avg, 2))
        logging.debug(f"Edge {u}->{v}: avg={avg:.2f}ms over {len(rtts)} runs")
    return G


def cost(path, G):
    return sum(G[u][v]['weight'] for u, v in zip(path, path[1:]))


def get_initial_path(G, src, dst, cutoff=30):
    try:
        path = nx.shortest_path(G, src, dst, weight='weight')
        logging.info(f"Initial Dijkstra path: {path}, cost={cost(path, G):.2f}ms")
        return path
    except nx.NetworkXNoPath:
        paths = list(nx.all_simple_paths(G, src, dst, cutoff=cutoff))
        if not paths:
            raise ValueError(f"No path from {src} to {dst}")
        path = random.choice(paths)
        logging.info(f"Initial random path: {path}, cost={cost(path, G):.2f}ms")
        return path


def neighbor(path, G):
    if len(path) <= 3:
        return path
    new = path.copy()
    i, j = sorted(random.sample(range(1, len(path)-1), 2))
    new[i], new[j] = new[j], new[i]
    if all(G.has_edge(u, v) for u, v in zip(new, new[1:])):
        return new
    return path


def simulated_annealing(G, src, dst, T0, T_min, alpha, N):
    current = get_initial_path(G, src, dst)
    best = current
    T = T0
    it = 0
    while T > T_min:
        for _ in range(N):
            it += 1
            cand = neighbor(current, G)
            delta = cost(cand, G) - cost(current, G)
            if delta < 0 or random.random() < math.exp(-delta / T):
                current = cand
                if cost(current, G) < cost(best, G):
                    best = current
                    logging.info(f"New best: {best}, cost={cost(best, G):.2f}ms, iter={it}, T={T:.4f}")
        T *= alpha
    logging.info(f"SA done: iterations={it}, best cost={cost(best, G):.2f}ms")
    return best


def export_to_csv(edge_data, best, outdir):
    od = Path(outdir)
    od.mkdir(exist_ok=True)
    edge_file = od / 'edge_data.csv'
    with open(edge_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['src', 'dst', 'sample_count', 'avg_latency_ms'])
        for (u, v), rtts in edge_data.items():
            avg = round(sum(rtts) / len(rtts), 2)
            writer.writerow([u, v, len(rtts), avg])
    logging.info(f"Exported edge_data.csv to {edge_file}")
    path_file = od / 'best_path.csv'
    with open(path_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['hop_index', 'node'])
        for idx, node in enumerate(best, start=1):
            writer.writerow([idx, node])
    logging.info(f"Exported best_path.csv to {path_file}")


def plot_graph(G, best):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrowsize=20)
    path_edges = list(zip(best, best[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    plt.title('Network Graph: Optimal Path')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('target', help='Destination hostname or IP')
    parser.add_argument('--runs', type=int, default=5, help='Number of traceroute runs')
    parser.add_argument('--max-ttl', type=int, default=30, help='Max TTL')
    parser.add_argument('--timeout', type=float, default=2.0, help='Probe timeout (s)')
    parser.add_argument('--sa-n', type=int, default=100, help='SA iterations per temp')
    parser.add_argument('--sa-t0', type=float, default=1000, help='SA initial temp')
    parser.add_argument('--sa-alpha', type=float, default=0.95, help='SA cooling rate')
    parser.add_argument('--sa-tmin', type=float, default=1e-3, help='SA min temp')
    parser.add_argument('--csv-dir', default='output_csv', help='Directory to save CSV outputs')
    parser.add_argument('--no-plot', action='store_true', help='Disable graph plotting')
    args = parser.parse_args()

    # Resolve target to IP
    try:
        dst_ip = socket.gethostbyname(args.target)
        logging.info(f"Resolved {args.target} to {dst_ip}")
    except socket.gaierror:
        logging.error(f"Unable to resolve {args.target}")
        return

    # Aggregate traceroute data
    edge_data = aggregate_paths(args.runs, dst_ip, args.max_ttl, args.timeout)
    G = build_graph(edge_data)
    logging.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Source and destination nodes
    src = next(iter(edge_data.keys()))[0]
    dst = dst_ip
    if dst not in G:
        logging.error(f"Destination {dst} not in graph nodes. Exiting.")
        return

    # Run Simulated Annealing
    best_path = simulated_annealing(G, src, dst, args.sa_t0, args.sa_tmin, args.sa_alpha, args.sa_n)

    # Export and plot
    export_to_csv(edge_data, best_path, args.csv_dir)
    if not args.no_plot:
        plot_graph(G, best_path)

if __name__ == '__main__':
    main()
