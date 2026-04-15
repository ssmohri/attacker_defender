
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd
import streamlit as st

# -----------------------------
# Config
# -----------------------------
GRID_SIZE = 5
TOTAL_TEU = 5
VEHICLES = {
    "Heavy Truck (5 TEU)": {"capacity": 5, "fixed_cost": 140},
    "Medium Truck (3 TEU)": {"capacity": 3, "fixed_cost": 95},
    "Light Truck (2 TEU)": {"capacity": 2, "fixed_cost": 70},
    "Van (1 TEU)": {"capacity": 1, "fixed_cost": 40},
}
STATE_FILE = "hazmat_game_state.json"
RNG_SEED = 42
RISK_DAMAGE_MULTIPLIER = 120

ROUTE_COLORS = [
    "#ff7f0e", "#1f77b4", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf"
]

st.set_page_config(page_title="Hazmat Attacker–Defender Game", layout="wide")


# -----------------------------
# Node mapping: single ID per node
# -----------------------------
def node_to_id(node: Tuple[int, int]) -> int:
    r, c = node
    return r * GRID_SIZE + c


def id_to_node(node_id: int) -> Tuple[int, int]:
    return (node_id // GRID_SIZE, node_id % GRID_SIZE)


# -----------------------------
# Utilities
# -----------------------------
def edge_key(a: Tuple[int, int], b: Tuple[int, int]) -> str:
    u, v = sorted([a, b])
    return f"{u[0]},{u[1]}|{v[0]},{v[1]}"


def edge_label_from_key(key: str) -> str:
    left, right = key.split("|")
    a = tuple(map(int, left.split(",")))
    b = tuple(map(int, right.split(",")))
    return f"{node_to_id(a)} ↔ {node_to_id(b)}"


def is_adjacent(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def validate_route(nodes: List[Tuple[int, int]], grid_size: int = GRID_SIZE) -> Tuple[bool, str]:
    if len(nodes) < 2:
        return False, "Route must contain at least two nodes."
    if nodes[0] != (0, 0):
        return False, "Route must start at node 0."
    if nodes[-1] != (grid_size - 1, grid_size - 1):
        return False, f"Route must end at node {grid_size * grid_size - 1}."

    for n in nodes:
        if not (0 <= n[0] < grid_size and 0 <= n[1] < grid_size):
            return False, f"Node {n} is outside the grid."

    for i in range(len(nodes) - 1):
        if not is_adjacent(nodes[i], nodes[i + 1]):
            return False, f"Nodes {node_to_id(nodes[i])} and {node_to_id(nodes[i+1])} are not adjacent."

    return True, "Valid route."


def route_to_edges(nodes: List[Tuple[int, int]]) -> List[str]:
    return [edge_key(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


def load_state() -> Dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "game_mode": "Nash",
        "defender_submissions": [],
        "attacks": [],
        "network_seed": RNG_SEED,
    }


def save_state(state: Dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# -----------------------------
# Network generation
# -----------------------------
@st.cache_data
def build_network(grid_size: int = GRID_SIZE, seed: int = RNG_SEED):
    random.seed(seed)
    G = nx.Graph()
    edge_data = {}

    for i in range(grid_size):
        for j in range(grid_size):
            G.add_node((i, j))

    for i in range(grid_size):
        for j in range(grid_size):
            if i < grid_size - 1:
                a, b = (i, j), (i + 1, j)
                transport_cost = random.randint(8, 25)
                risk = random.randint(1, 10)
                key = edge_key(a, b)
                G.add_edge(a, b, transport_cost=transport_cost, risk=risk, key=key)
                edge_data[key] = {
                    "from": a,
                    "to": b,
                    "transport_cost": transport_cost,
                    "risk": risk,
                    "damage": risk * RISK_DAMAGE_MULTIPLIER,
                }
            if j < grid_size - 1:
                a, b = (i, j), (i, j + 1)
                transport_cost = random.randint(8, 25)
                risk = random.randint(1, 10)
                key = edge_key(a, b)
                G.add_edge(a, b, transport_cost=transport_cost, risk=risk, key=key)
                edge_data[key] = {
                    "from": a,
                    "to": b,
                    "transport_cost": transport_cost,
                    "risk": risk,
                    "damage": risk * RISK_DAMAGE_MULTIPLIER,
                }

    return G, edge_data


def draw_network(
    G,
    edge_data,
    route_specs: List[Dict] = None,
    attack_edges: Set[str] = None,
    show_legend: bool = False,
):
    route_specs = route_specs or []
    attack_edges = attack_edges or set()

    pos = {(i, j): (j, -i) for i, j in G.nodes()}
    fig, ax = plt.subplots(figsize=(10, 8))

    node_colors = []
    for n in G.nodes():
        if n == (0, 0):
            node_colors.append("lightgreen")
        elif n == (GRID_SIZE - 1, GRID_SIZE - 1):
            node_colors.append("lightcoral")
        else:
            node_colors.append("lightblue")

    nx.draw_networkx_nodes(G, pos, node_size=650, node_color=node_colors, ax=ax)

    node_labels = {n: node_to_id(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold", ax=ax)

    nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), edge_color="lightgray", width=1.5, ax=ax)

    legend_handles = []
    edge_volume_usage = defaultdict(int)

    for idx, spec in enumerate(route_specs):
        color = spec["color"]
        label = spec["label"]
        edges_to_draw = []
        for ekey in spec["edges"]:
            d = edge_data[ekey]
            edges_to_draw.append((d["from"], d["to"]))
            edge_volume_usage[ekey] += spec.get("volume_teu", 0)

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges_to_draw,
            edge_color=color,
            width=4.0,
            ax=ax
        )

        if show_legend:
            legend_handles.append(Line2D([0], [0], color=color, lw=4, label=label))

    if attack_edges:
        attacked_edges_to_draw = []
        for ekey in attack_edges:
            d = edge_data[ekey]
            attacked_edges_to_draw.append((d["from"], d["to"]))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=attacked_edges_to_draw,
            edge_color="red",
            width=5.5,
            style="dashed",
            ax=ax
        )
        if show_legend:
            legend_handles.append(Line2D([0], [0], color="red", lw=4, linestyle="--", label="Attacked link"))

    edge_labels = {}
    for _, d in edge_data.items():
        edge_labels[(d["from"], d["to"])] = f"C={d['transport_cost']}\nD={d['damage']}"

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        rotate=False,
        ax=ax,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )

    if route_specs:
        volume_edge_labels = {}
        for ekey, vol in edge_volume_usage.items():
            if vol > 0:
                d = edge_data[ekey]
                volume_edge_labels[(d["from"], d["to"])] = f"V={vol}"
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=volume_edge_labels,
            font_size=8,
            font_color="darkgreen",
            rotate=False,
            ax=ax,
            label_pos=0.3,
            bbox={"facecolor": "white", "edgecolor": "darkgreen", "alpha": 0.75},
        )

    if show_legend and legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    ax.set_title("5×5 Hazmat Network")
    ax.axis("off")
    plt.tight_layout()
    return fig


def compute_trip_cost(route_edges: List[str], edge_data: Dict[str, Dict], vehicle_name: str) -> Dict:
    route_transport_cost = sum(edge_data[e]["transport_cost"] for e in route_edges)
    base = VEHICLES[vehicle_name]["fixed_cost"]
    total_transport = base + route_transport_cost
    route_risk_score = sum(edge_data[e]["risk"] for e in route_edges)

    return {
        "vehicle_fixed_cost": base,
        "route_transport_cost": route_transport_cost,
        "total_transport_cost": total_transport,
        "route_risk_score": route_risk_score,
    }


def evaluate_game(defender_submissions: List[Dict], attacks: List[str], edge_data: Dict[str, Dict]) -> Dict:
    attacked_set = set(attacks)

    trip_results = []
    total_transport_cost = 0
    total_attack_damage = 0

    for trip in defender_submissions:
        route_edges = trip["route_edges"]
        used_attacked_edges = [e for e in route_edges if e in attacked_set]
        base_calc = compute_trip_cost(route_edges, edge_data, trip["vehicle"])
        damage = sum(edge_data[e]["damage"] for e in used_attacked_edges)

        trip_result = {
            "shipment_id": trip["shipment_id"],
            "vehicle": trip["vehicle"],
            "volume_teu": trip["volume_teu"],
            "route_length_edges": len(route_edges),
            "transport_cost": base_calc["total_transport_cost"],
            "attacked_edges_hit": len(used_attacked_edges),
            "attack_damage": damage,
            "total_trip_cost": base_calc["total_transport_cost"] + damage,
        }
        trip_results.append(trip_result)
        total_transport_cost += base_calc["total_transport_cost"]
        total_attack_damage += damage

    return {
        "trip_results": trip_results,
        "total_transport_cost": total_transport_cost,
        "total_attack_damage": total_attack_damage,
        "grand_total_cost": total_transport_cost + total_attack_damage,
        "attacker_score": total_attack_damage,
        "defender_score": -(total_transport_cost + total_attack_damage),
    }


def make_route_specs_from_trips(trips: List[Dict]) -> List[Dict]:
    specs = []
    for i, trip in enumerate(trips):
        specs.append({
            "edges": trip["route_edges"],
            "color": ROUTE_COLORS[i % len(ROUTE_COLORS)],
            "label": f"Trip {trip['shipment_id']}: {trip['volume_teu']} TEU",
            "volume_teu": trip["volume_teu"],
        })
    return specs


# -----------------------------
# App
# -----------------------------
state = load_state()
G, edge_data = build_network(seed=state.get("network_seed", RNG_SEED))

st.title("Hazmat Transportation: Attacker–Defender Game")

with st.sidebar:
    st.header("Game setup")
    game_mode = st.radio(
        "Choose game mode",
        ["Nash", "Stackelberg"],
        index=0 if state.get("game_mode", "Nash") == "Nash" else 1,
    )
    if game_mode != state.get("game_mode"):
        state["game_mode"] = game_mode
        save_state(state)

    role = st.radio("Choose role", ["Home", "Defender", "Attacker", "Results / Referee"])

    st.markdown("---")
    if st.button("Reset game state"):
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        st.success("Game state reset. Refresh the page.")
        st.stop()

st.markdown(
    f"""
**Current mode:** `{state.get("game_mode", "Nash")}`

- Defender moves **{TOTAL_TEU} TEU** from node **0** to node **{GRID_SIZE * GRID_SIZE - 1}**
- Edge labels show:
  - **C = transport cost**
  - **D = attack damage**
- In **Nash**, attacker cannot see defender routes
- In **Stackelberg**, attacker can see defender routes
"""
)

if role == "Home":
    st.subheader("Network")
    fig = draw_network(G, edge_data)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Instructions")
    st.markdown(
        f"""
1. Choose **Nash** or **Stackelberg** in the sidebar.  
2. Defender selects vehicles and clicks node buttons to build routes.  
3. Each route must go from node **0** to node **{GRID_SIZE * GRID_SIZE - 1}**.  
4. Attacker chooses **1 or 2 links** to attack.  
5. A link is identified by its two node IDs.  
6. Referee opens the results page to compare transport cost and attack damage.  
"""
    )

    st.subheader("Vehicle options")
    vehicle_df = pd.DataFrame([
        {"vehicle": name, "capacity_teu": spec["capacity"], "fixed_cost": spec["fixed_cost"]}
        for name, spec in VEHICLES.items()
    ])
    st.dataframe(vehicle_df, use_container_width=True)

elif role == "Defender":
    st.subheader("Defender panel")
    st.write(f"Create shipment plans whose total volume equals **{TOTAL_TEU} TEU**.")

    if "defender_working_trips" not in st.session_state:
        st.session_state.defender_working_trips = []
    if "current_route_nodes" not in st.session_state:
        st.session_state.current_route_nodes = [(0, 0)]

    used_volume = sum(t["volume_teu"] for t in st.session_state.defender_working_trips)
    remaining = TOTAL_TEU - used_volume

    current_route_specs = make_route_specs_from_trips(st.session_state.defender_working_trips)
    fig = draw_network(G, edge_data, route_specs=current_route_specs, show_legend=True)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Build a route by clicking node IDs")
    st.markdown(
        f"""
- Start node is fixed at **0**
- End node is **{GRID_SIZE * GRID_SIZE - 1}**
- Each next node must be adjacent to the previous one
- Click nodes in order until you reach the final node
"""
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        vehicle = st.selectbox("Vehicle", list(VEHICLES.keys()))
        cap = VEHICLES[vehicle]["capacity"]
        volume = st.number_input(
            f"Volume for this trip (max {cap}, remaining {remaining})",
            min_value=1,
            max_value=max(1, min(cap, remaining)) if remaining > 0 else 1,
            value=1,
            step=1,
            disabled=remaining <= 0,
        )
    with c2:
        current_route_ids = [node_to_id(n) for n in st.session_state.current_route_nodes]
        st.markdown(f"**Current route:** {' → '.join(map(str, current_route_ids))}")

    st.markdown("**Click nodes to extend the path**")
    for r in range(GRID_SIZE):
        cols = st.columns(GRID_SIZE)
        for c in range(GRID_SIZE):
            nid = node_to_id((r, c))
            node = (r, c)
            last_node = st.session_state.current_route_nodes[-1]

            disabled = False
            if nid == 0:
                disabled = len(st.session_state.current_route_nodes) > 1
            elif not is_adjacent(last_node, node):
                disabled = True

            if cols[c].button(str(nid), key=f"nodebtn_{r}_{c}", disabled=disabled):
                st.session_state.current_route_nodes.append(node)
                st.rerun()

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Undo last node", disabled=len(st.session_state.current_route_nodes) <= 1):
            st.session_state.current_route_nodes = st.session_state.current_route_nodes[:-1]
            st.rerun()
    with b2:
        if st.button("Clear current route"):
            st.session_state.current_route_nodes = [(0, 0)]
            st.rerun()
    with b3:
        last_id = node_to_id(st.session_state.current_route_nodes[-1])
        can_add_trip = remaining > 0 and last_id == (GRID_SIZE * GRID_SIZE - 1)
        if st.button("Add trip", disabled=not can_add_trip):
            if volume > cap:
                st.error(f"{vehicle} cannot carry {volume} TEU.")
            elif volume > remaining:
                st.error("This trip exceeds the remaining volume.")
            else:
                nodes = st.session_state.current_route_nodes
                valid, msg = validate_route(nodes)
                if not valid:
                    st.error(msg)
                else:
                    route_edges = route_to_edges(nodes)
                    calc = compute_trip_cost(route_edges, edge_data, vehicle)
                    st.session_state.defender_working_trips.append({
                        "shipment_id": len(st.session_state.defender_working_trips) + 1,
                        "vehicle": vehicle,
                        "volume_teu": int(volume),
                        "route_nodes": nodes,
                        "route_node_ids": [node_to_id(n) for n in nodes],
                        "route_edges": route_edges,
                        "preview_transport_cost": calc["total_transport_cost"],
                    })
                    st.session_state.current_route_nodes = [(0, 0)]
                    st.success("Trip added.")
                    st.rerun()

    if st.session_state.defender_working_trips:
        st.subheader("Current trips")
        preview_df = pd.DataFrame([
            {
                "shipment_id": t["shipment_id"],
                "vehicle": t["vehicle"],
                "volume_teu": t["volume_teu"],
                "route_nodes": " → ".join(map(str, t["route_node_ids"])),
                "edges_in_route": len(t["route_edges"]),
                "transport_cost_preview": t["preview_transport_cost"],
            }
            for t in st.session_state.defender_working_trips
        ])
        st.dataframe(preview_df, use_container_width=True)

        st.info(
            "Cost check: each trip is charged separately. So if the same route is used multiple times, "
            "the route cost and vehicle fixed cost are counted each time."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear all trips"):
            st.session_state.defender_working_trips = []
            st.session_state.current_route_nodes = [(0, 0)]
            st.success("All trips cleared.")
            st.rerun()
    with col_b:
        if st.button("Save defender plan"):
            total_volume = sum(t["volume_teu"] for t in st.session_state.defender_working_trips)
            if total_volume != TOTAL_TEU:
                st.error(f"Total volume must equal {TOTAL_TEU} TEU. Current total is {total_volume}.")
            else:
                state["defender_submissions"] = st.session_state.defender_working_trips
                save_state(state)
                st.success("Defender plan saved.")

elif role == "Attacker":
    st.subheader("Attacker panel")
    defender_trips = state.get("defender_submissions", [])
    current_mode = state.get("game_mode", "Nash")

    if current_mode == "Stackelberg":
        if defender_trips:
            st.success("Stackelberg mode: defender routes are visible to the attacker.")
            route_specs = make_route_specs_from_trips(defender_trips)
            fig = draw_network(G, edge_data, route_specs=route_specs, show_legend=True)
            st.pyplot(fig, clear_figure=True)
        else:
            st.warning("No defender plan has been saved yet.")
            fig = draw_network(G, edge_data)
            st.pyplot(fig, clear_figure=True)
    else:
        st.info("Nash mode: defender routes are hidden from the attacker.")
        fig = draw_network(G, edge_data)
        st.pyplot(fig, clear_figure=True)

    attack_count = st.radio("Number of attacks", [1, 2], horizontal=True)

    attack_options = sorted(edge_data.keys())
    valid_default_attacks = [e for e in state.get("attacks", []) if e in attack_options][:attack_count]

    selected_attack_edges = st.multiselect(
        "Select links to attack",
        options=attack_options,
        format_func=lambda x: edge_label_from_key(x),
        max_selections=attack_count,
        default=valid_default_attacks,
    )

    if selected_attack_edges:
        route_specs = make_route_specs_from_trips(defender_trips) if (current_mode == "Stackelberg" and defender_trips) else []
        fig = draw_network(G, edge_data, route_specs=route_specs, attack_edges=set(selected_attack_edges), show_legend=True)
        st.pyplot(fig, clear_figure=True)

    if st.button("Save attack plan"):
        if len(selected_attack_edges) != attack_count:
            st.error(f"Please select exactly {attack_count} attacked link(s).")
        else:
            state["attacks"] = selected_attack_edges
            save_state(state)
            st.success("Attack plan saved.")

elif role == "Results / Referee":
    st.subheader("Results / Referee")

    defender_trips = state.get("defender_submissions", [])
    attacks = state.get("attacks", [])

    if not defender_trips:
        st.error("No defender plan found.")
    if not attacks:
        st.error("No attack plan found.")

    if defender_trips and attacks:
        results = evaluate_game(defender_trips, attacks, edge_data)
        route_specs = make_route_specs_from_trips(defender_trips)
        fig = draw_network(G, edge_data, route_specs=route_specs, attack_edges=set(attacks), show_legend=True)
        st.pyplot(fig, clear_figure=True)

        st.subheader("Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transport cost", f"{results['total_transport_cost']:.0f}")
        c2.metric("Attack damage", f"{results['total_attack_damage']:.0f}")
        c3.metric("Grand total cost", f"{results['grand_total_cost']:.0f}")
        c4.metric("Attacker score", f"{results['attacker_score']:.0f}")

        st.subheader("Trip-by-trip results")
        trips_df = pd.DataFrame(results["trip_results"])
        st.dataframe(trips_df, use_container_width=True)
