import math
import time
import heapq
import random
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================================
# 1. KHỞI TẠO TỌA ĐỘ VÀ ĐỒ THỊ MẠNG LƯỚI ĐƯỜNG BỘ HÀ NỘI (dữ liệu đưohclấy từ Google maps và Google earth)
# ==========================================================
coords = {
    "A": (20.9895833, 105.7512678),    # AEON Mall Hà Đông (Điểm xuất phát)
    "B": (21.0099234, 105.7863024),    # Bảo tàng Hà Nội - Đường Phạm Hùng
    "C": (21.0060280, 105.8057246),    # Ngã tư Lê Văn Lương - Hoàng Đạo Thúy
    "D": (21.0087432, 105.8286961),    # Học viện Ngân hàng - Phố Chùa Bộc
    "E": (21.0010301, 105.8411324),    # Bệnh viện Bạch Mai - Đường Giải Phóng
    "M": (20.9806160, 105.7576356),    # Công viên Thiên văn học Dương Nội
    "N": (20.9734048, 105.7784225),    # Bưu điện Hà Đông - Đường Quang Trung
    "L": (20.9790030, 105.7856727),    # TTTM Hồ Gươm Plaza - Đường Trần Phú
    "P": (20.9918796, 105.8038468),    # Nút giao Khuất Duy Tiến - Nguyễn Trãi
    "F": (20.9995053, 105.8292533),    # Bảo tàng Phòng không - Không quân
    "NEU": (20.9993066, 105.8437235),  # ĐH Kinh tế Quốc dân (Điểm đích)
}

edges = [
    ("A", "B", 6.8),
    ("A", "M", 1.7),
    ("A", "N", 4.6),
    ("M", "L", 3.8),
    ("N", "L", 1.5),
    ("N", "P", 4.9),
    ("L", "C", 4.7),
    ("L", "P", 3.5),
    ("B", "C", 2.6),
    ("B", "D", 5.7),
    ("C", "D", 3.4),
    ("C", "F", 3.3),
    ("D", "E", 2.1),
    ("E", "NEU", 1.7),
    ("P", "F", 3.1),
    ("P", "NEU", 5.9),
    ("F", "NEU", 2.3),
]

G = nx.Graph()
for u, v, w in edges:
    G.add_edge(u, v, weight=w)


# Chuẩn hóa tọa độ để trực quan hóa đồ thị
def normalize_coords(coords, scale=10):
    lats = [lat for lat, lon in coords.values()]
    lons = [lon for lat, lon in coords.values()]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    pos = {}
    for node, (lat, lon) in coords.items():
        x = (lon - min_lon) / (max_lon - min_lon) * scale
        y = (lat - min_lat) / (max_lat - min_lat) * scale
        pos[node] = (x, y)
    return pos


pos = normalize_coords(coords, scale=10)


# ==========================================================
# 2. TÍNH HÀM HEURISTIC BẰNG CÔNG THỨC HAVERSINE
# ==========================================================
def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


start = "A"
goal = "NEU"
heuristics = {
    node: haversine(coord, coords[goal]) for node, coord in coords.items()
}

print("=== Heuristic (Haversine) ===")
for node, hval in heuristics.items():
    print(f"h({node}) -> NEU = {hval:.2f} km")


# ==========================================================
# 3. CÀI ĐẶT THUẬT TOÁN GREEDY BEST-FIRST SEARCH & A*
# ==========================================================
def greedy_search(G, start, goal, heuristics):
    """
    Greedy Best-First Search: f(n) = h(n)
    Trả về: (path, total_cost, expanded_nodes, runtime_ms)
    """
    t0 = time.perf_counter()
    if start not in G or goal not in G:
        return None, float("inf"), 0, 0.0

    visited = set()
    pq = [(heuristics[start], start, [start])]

    while pq:
        _, current, path = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            runtime_ms = (time.perf_counter() - t0) * 1000
            total_cost = sum(
                G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
            )
            return path, total_cost, len(visited), runtime_ms

        neighbors = list(G.neighbors(current))
        if not neighbors:
            continue

        for neighbor in neighbors:
            if neighbor not in visited:
                heapq.heappush(
                    pq, (heuristics[neighbor], neighbor, path + [neighbor])
                )

    runtime_ms = (time.perf_counter() - t0) * 1000
    return None, float("inf"), len(visited), runtime_ms


def a_star_search(G, start, goal, heuristics):
    """
    A* Search: f(n) = g(n) + h(n)
    Trả về: (path, total_cost, expanded_nodes, runtime_ms)
    """
    t0 = time.perf_counter()
    if start not in G or goal not in G:
        return None, float("inf"), 0, 0.0

    open_set = [(heuristics[start], 0, start, [start])]
    visited = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            runtime_ms = (time.perf_counter() - t0) * 1000
            return path, g, len(visited), runtime_ms

        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                cost = G[current][neighbor]["weight"]
                new_g = g + cost
                new_f = new_g + heuristics[neighbor]
                heapq.heappush(
                    open_set, (new_f, new_g, neighbor, path + [neighbor])
                )

    runtime_ms = (time.perf_counter() - t0) * 1000
    return None, float("inf"), len(visited), runtime_ms


# ================================
# 4. HÀM CHẶN NGẪU NHIÊN
# ================================
def random_block_edges(G, num_blocked=2, seed=None):
    """
    Chặn ngẫu nhiên num_blocked cạnh liền kề trong graph.
    - num_blocked = 0: Không chặn cạnh nào (đồ thị gốc)
    - num_blocked = 1: Chặn 1 cạnh đơn lẻ
    - num_blocked >= 2: Các cạnh bị chặn có chung ít nhất 1 node
    """
    if seed is not None:
        random.seed(seed)

    G_blocked = G.copy()
    all_edges = list(G.edges())
    blocked = []

    if not all_edges or num_blocked < 1:
        return G_blocked, blocked

    first_edge = random.choice(all_edges)
    blocked.append(first_edge)
    if G_blocked.has_edge(*first_edge):
        G_blocked.remove_edge(*first_edge)

    if num_blocked == 1:
        return G_blocked, blocked

    current_candidates = [first_edge]
    while len(blocked) < num_blocked and current_candidates:
        u, v = random.choice(current_candidates)
        adjacent_edges = [
            e for e in all_edges if e not in blocked and (u in e or v in e)
        ]
        if not adjacent_edges:
            break

        next_edge = random.choice(adjacent_edges)
        blocked.append(next_edge)
        if G_blocked.has_edge(*next_edge):
            G_blocked.remove_edge(*next_edge)
        current_candidates.append(next_edge)

    return G_blocked, blocked


# ==========================================================
# 5. HÀM XUẤT 2 ẢNH: SO SÁNH ĐƯỜNG ĐI & SO SÁNH HIỆU SUẤT
# ==========================================================
def run_random_scenario_and_generate_images(G, pos, start, goal, heuristics):
    # Ngẫu nhiên chọn 1 trong 4 tình huống: chặn 0, 1, 2 hoặc 3 cạnh
    scenarios = [0, 1, 2, 3]
    num_blocked = random.choice(scenarios)

    # Đảm bảo cạnh bị chặn ngẫu nhiên không làm cô lập hoàn toàn đường tới đích
    while True:
        G_sim, blocked_edges = random_block_edges(G, num_blocked=num_blocked)
        if nx.has_path(G_sim, start, goal):
            break

    # Chạy 2 thuật toán trên tình huống vừa random
    g_path, g_cost, g_nodes, g_time = greedy_search(
        G_sim, start, goal, heuristics
    )
    a_path, a_cost, a_nodes, a_time = a_star_search(
        G_sim, start, goal, heuristics
    )

    blocked_str = (
        ", ".join([f"{u}-{v}" for u, v in blocked_edges])
        if blocked_edges
        else "None"
    )
    print(f"\n=== KẾT QUẢ TÌNH HUỐNG NGẪU NHIÊN ({num_blocked} CẠNH BỊ CHẶN) ===")
    print(f"Cạnh bị chặn : {blocked_str}")
    print(
        f"Greedy Path  : {' -> '.join(g_path)} | Cost: {g_cost:.2f} km | Nodes: {g_nodes} | Time: {g_time:.4f} ms"
    )
    print(
        f"A* Path      : {' -> '.join(a_path)} | Cost: {a_cost:.2f} km | Nodes: {a_nodes} | Time: {a_time:.4f} ms"
    )

    # ----------------------------------------------------------
    # ẢNH 1: SO SÁNH ĐƯỜNG ĐI GIỮA 2 THUẬT TOÁN
    # ----------------------------------------------------------
    g_edges = (
        [(g_path[i], g_path[i + 1]) for i in range(len(g_path) - 1)]
        if g_path
        else []
    )
    a_edges = (
        [(a_path[i], a_path[i + 1]) for i in range(len(a_path) - 1)]
        if a_path
        else []
    )
    edge_labels = nx.get_edge_attributes(G, "weight")

    fig1, ax1 = plt.subplots(figsize=(8.5, 6.5))
    nx.draw_networkx_nodes(
        G, pos, ax=ax1, node_color="lightgray", node_size=650
    )
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color="darkgray", width=1.5)

    # Vẽ cạnh bị chặn (Đỏ), đường đi Greedy (Xanh dương), đường đi A* (Xanh lá)
    if blocked_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=blocked_edges,
            ax=ax1,
            edge_color="red",
            width=2.5,
            label=f"Blocked ({blocked_str})",
        )
    if g_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=g_edges,
            ax=ax1,
            edge_color="blue",
            width=3.0,
            label=f"Greedy ({g_cost:.1f} km)",
        )
    if a_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=a_edges,
            ax=ax1,
            edge_color="green",
            width=3.0,
            label=f"A* ({a_cost:.1f} km)",
        )

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax1, font_size=8
    )
    ax1.set_title(
        f"Difference in path between two algorithms ({num_blocked} blocked edge(s): {blocked_str})",
        fontsize=12,
        fontweight="bold",
        y=-0.08,
    )
    ax1.legend(loc="upper left", frameon=True)
    ax1.axis("off")

    plt.tight_layout()
    path_img_name = "path_comparison.png"
    fig1.savefig(path_img_name, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"-> Đã lưu ảnh so sánh đường đi: {path_img_name}")

    # ----------------------------------------------------------
    # ẢNH 2: SO SÁNH HIỆU SUẤT GIỮA 2 THUẬT TOÁN
    # ----------------------------------------------------------
    metrics = [
        ("Path Cost", [g_cost, a_cost], "km", "{:.2f}"),
        ("Expanded Nodes", [g_nodes, a_nodes], "nodes", "{:d}"),
        ("Runtime", [g_time, a_time], "ms", "{:.4f}"),
    ]

    fig2, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, (title, vals, ylabel, fmt) in zip(axes, metrics):
        bars = ax.bar(
            ["Greedy", "A*"], vals, color=["blue", "green"], width=0.5
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        max_val = max(vals) if max(vals) > 0 else 1
        ax.set_ylim(0, max_val * 1.2)

        for bar in bars:
            h = bar.get_height()
            val_str = fmt.format(int(h) if fmt == "{:d}" else h)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max_val * 0.03,
                val_str,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.suptitle(
        f"Performance Comparison ({num_blocked} blocked edge(s): {blocked_str})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    perf_img_name = "performance_comparison.png"
    fig2.savefig(perf_img_name, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"-> Đã lưu ảnh so sánh hiệu suất: {perf_img_name}")


# ==========================================================
# 6. CHẠY CHƯƠNG TRÌNH
# ==========================================================
if __name__ == "__main__":
    run_random_scenario_and_generate_images(G, pos, start, goal, heuristics)
