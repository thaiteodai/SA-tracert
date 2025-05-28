#!/usr/bin/env python3
"""
Traceroute + Simulated Annealing script nâng cao + xuất file CSV, vẽ đồ thị, ghi log

Các chức năng:
- Chay tracert/traceroute nhieu lan de tong hop du lieu latency
- Xây dựng đồ thị có hướng trọng số (độ trễ trung bình trên từng cạnh)
- Dùng giải thuật SA để tìm đường dẫn có độ trễ thấp nhất
- Lưu số liệu vào file CSV bao gồm thống kê các cạnh và đường dẫn tối ưu (sd module CSV)
- Vẽ đồ thị mạng với đường dẫn tối ưu
- Ghi log chi tiết

Cách sử dụng:
  python enhanced_traceroute_sa.py TARGET [--runs R] [--max-ttl T] [--timeout S] [--sa-* options] [--csv-dir DIR] [--no-plot]

Yêu cầu: scapy, networkx, matplotlib
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

# import scapy, bypass cảnh báo ARP (ko tìm được MAC của gateway nên sẽ broadcast tràn lan)
from scapy.all import IP, ICMP, sr1
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

import networkx as nx
import matplotlib.pyplot as plt

# Cấu hình log
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# chạy tracert bằng scapy
def traceroute_scapy(dst_ip, max_ttl=30, timeout=2.0): # mặc định max ttl = 30; timeout = 2s
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

# hàm tổng hợp dữ liệu bằng cách chạy tracert nhiều lần
def aggregate_paths(runs, dst_ip, max_ttl, timeout):
    edge_data = {}
    for i in range(runs):
        logging.info(f"Run {i+1}/{runs}")
        path = traceroute_scapy(dst_ip, max_ttl, timeout)
        logging.debug(f"Traceroute path: {[hop for hop, _ in path]}")
        prev = None
        last_valid_hop = None
        for hop, rtt in path:
            if hop and rtt is not None:
                if prev:
                    edge_data.setdefault((prev, hop), []).append(rtt)
                prev = hop
                last_valid_hop = hop
            elif hop is not None:
                prev = hop  # Hop có IP nhưng không đo RTT
        # Nếu không tới được đích, thêm thủ công cạnh từ hop cuối đến dst_ip
        if dst_ip not in [hop for hop, _ in path if hop]:
            if last_valid_hop and last_valid_hop != dst_ip:
                logging.warning(f"Destination {dst_ip} not reached. Adding synthetic edge from {last_valid_hop}.")
                edge_data.setdefault((last_valid_hop, dst_ip), []).append(timeout * 1000)
    return edge_data


# tạo graph
def build_graph(edge_data):
    G = nx.DiGraph()
    for (u, v), rtts in edge_data.items():
        avg = sum(rtts) / len(rtts)
        G.add_edge(u, v, weight=round(avg, 2))
        logging.debug(f"Edge {u}->{v}: avg={avg:.2f}ms over {len(rtts)} runs")
    return G

# tính tổng độ trễ của path
def cost(path, G):
    return sum(G[u][v]['weight'] for u, v in zip(path, path[1:]))

# tạo đường đi ban đầu giữa src và dst bằng Dijkstra
def get_initial_path(G, src, dst, cutoff=30):
    try:
        path = nx.shortest_path(G, src, dst, weight='weight') #dijkstra
        logging.info(f"Initial Dijkstra path: {path}, cost={cost(path, G):.2f}ms")
        return path
    except nx.NetworkXNoPath: # nếu dijkstra không tìm được
        paths = list(nx.all_simple_paths(G, src, dst, cutoff=cutoff)) #fallback, chọn ngâu nhiên trong các đường đi đơn giản (khi đồ thị không liên thông về trọng số)
        if not paths:
            raise ValueError(f"No path from {src} to {dst}")
        path = random.choice(paths)
        logging.info(f"Initial random path: {path}, cost={cost(path, G):.2f}ms") #chọn ngẫu nhiên và ghi log khi không còn đường đi nào
        return path

# tạo hàng xóm lân cận của path bằng cách hoán đổi 2 node trong đường đi, giữ nguyên src và dst
def neighbor(path, G):
    if len(path) <= 3:
        return path
    new = path.copy()
    i, j = sorted(random.sample(range(1, len(path)-1), 2))
    new[i], new[j] = new[j], new[i]
    if all(G.has_edge(u, v) for u, v in zip(new, new[1:])):
        return new
    return path

# thuật toán chính
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

# lưu vào csv
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

# vẽ đồ thị

# def plot_graph(G: nx.DiGraph, path: list[str], use_sa: bool = False):
#     pos = nx.spring_layout(G, seed=42)
#     plt.figure(figsize=(12, 8))

#     # Tô màu tất cả các cạnh mặc định (màu sáng/xám)
#     nx.draw_networkx_edges(G, pos, alpha=0.3)

#     # Tô màu các node
#     nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue", edgecolors="black")

#     # Label node
#     nx.draw_networkx_labels(G, pos, font_size=10)

#     # Vẽ đường đi tốt nhất (nếu có)
#     if path and len(path) >= 2:
#         path_edges = list(zip(path, path[1:]))
#         edge_color = 'red' if use_sa else 'gray'
#         nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2.5, edge_color=edge_color)

#     plt.title("Traceroute Network Graph" + (" (Simulated Annealing)" if use_sa else " (Initial Path)"))
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()

def plot_graph(G: nx.DiGraph, path: list[str], use_sa: bool = False):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))

    # Vẽ các cạnh bình thường
    nx.draw_networkx_edges(G, pos, alpha=0.3)

    # Hiển thị latency trên cạnh (trọng số)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5)

    # Vẽ các nút
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue", edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Vẽ đường đi tối ưu
    if path and len(path) >= 2:
        path_edges = list(zip(path, path[1:]))
        edge_color = 'green' if use_sa else 'red'
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2.5, edge_color=edge_color)

    plt.title("Traceroute Network Graph" + (" (Simulated Annealing)" if use_sa else " (Dijkstra)"))
    plt.axis('off')
    plt.tight_layout()
    plt.show()



# hàm thiết lập các flag thêm tham số cho script
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
    # thêm flag --SA và --plot để tách biệt lựa chọn sử dụng thuật toán
    parser.add_argument('--SA', action='store_true', help='Enable Simulated Annealing optimization')
    # parser.add_argument('--plot', action='store_true', help='Enable graphic plotting (disable by default)')
    args = parser.parse_args()

    # Phân giải mục tiêu (có thể là tên miền hoặc IP) thành IP
    try:
        dst_ip = socket.gethostbyname(args.target)
        logging.info(f"Resolved {args.target} to {dst_ip}")
    except socket.gaierror:
        logging.error(f"Unable to resolve {args.target}")
        return

    # tổng hợp dữ liệu từ tracert
    edge_data = aggregate_paths(args.runs, dst_ip, args.max_ttl, args.timeout)
    G = build_graph(edge_data)
    logging.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Source and destination nodes
    src = next(iter(edge_data.keys()))[0]
    dst = dst_ip
    if dst not in G:
        logging.error(f"Destination {dst} not in graph nodes. Exiting.")
        return

    # chạy thuật toán Simulated Annealing, hoặc không :Đ
    if args.SA:
        best_path = simulated_annealing(G, src, dst, args.sa_t0, args.sa_tmin, args.sa_alpha, args.sa_n)
    else:
        best_path = get_initial_path(G, src, dst)
        logging.info(f"Using initial path without SA: {best_path}, cost={cost(best_path, G):.2f}ms") 

    # Xuất file csv và vẽ đồ thị
    export_to_csv(edge_data, best_path, args.csv_dir)
    if not args.no_plot:
        plot_graph(G, best_path, use_sa=args.SA)

if __name__ == '__main__':
    main()
    

