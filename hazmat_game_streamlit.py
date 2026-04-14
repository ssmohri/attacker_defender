import json
import os
import random
from typing import Dict, List, Tuple, Set

import matplotlib.pyplot as plt
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

st.set_page_config(page_title="Hazmat Attacker–Defender Game", layout="wide")


# -----------------------------
# Utilities
# -----------------------------
def edge_key(a: Tuple[int, int], b: Tuple[int, int]) -> str:
    u, v = sorted([a, b])
    return f"{u[0]},{u[1]}|{v[0]},{v[1]}"


def edge_label_from_key(key: str) -> str:
    left, right = key.split("|")
    return f"({left}) ↔ ({right})"


def parse_node_sequence(text: str) -> List[Tuple[int, int]]:
    text = text.replace("->", ";")
    parts = [p.strip() for p in text.split(";") if p.strip()]
    nodes = []
    for part in parts:
        x_str, y_str = [t.strip() for t in part.split(",")]
        x, y = int(x_str), int(y_str)
        nodes.append((x, y))
    return nodes


def is_adjacent(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def validate_route(nodes: List[Tuple[int, int]], grid_size: int = GRID_SIZE) -> Tuple[bool, str]:
    if len(nodes) < 2:
        return False, "Route must contain at least two nodes."
    if nodes[0] != (0, 0):
        return False, "Route must start at the origin (0,0)."
    if nodes[-1] != (grid_size - 1, grid_size - 1):
        return False, f"Route must end at the destination ({grid_size-1},{grid_size-1})."

    for n in nodes:
        if not (0 <= n[0] < grid_size and 0 <= n[1] < grid_size):
            return False, f"Node {n} is outside the grid."

    for i in range(len(nodes) - 1):
        if not is_adjacent(nodes[i], nodes[i + 1]):
            return False, f"Nodes {nodes[i]} and {nodes[i+1]} are not adjacent."

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


def draw_network(G, edge_data, highlight_edges: Set[str] = None, attack_edges: Set[str] = None):
    highlight_edges = highlight_edges or set()
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

    nx.draw_networkx_nodes(G, pos, node_size=550, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    base_edges = []
    base_colors = []
    base_widths = []
    for u, v, data in G.edges(data=True):
        key = data["key"]
        base_edges.append((u, v))
        if key in attack_edges and key in highlight_edges:
            base_colors.append("purple")
            base_widths.append(4.8)
        elif key in attack_edges:
            base_colors.append("red")
            base_widths.append(3.8)
        elif key in highlight_edges:
            base_colors.append("orange")
            base_widths.append(3.8)
        else:
            base_colors.append("gray")
            base_widths.append(1.5)

    nx.draw_networkx_edges(G, pos, edgelist=base_edges, edge_color=base_colors, width=base_widths, ax=ax)

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
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )

    ax.set_title("5×5 Hazmat Network")
    ax.axis("off")
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

- Defender moves **{TOTAL_TEU} TEU** from **(0,0)** to **({GRID_SIZE-1},{GRID_SIZE-1})**
- Edge labels show:
  - **C = transport cost**
  - **D = attack damage**
- In **Nash**, attacker cannot see defender routes
- In **Stackelberg**, attacker can see defender routes
"""
)

# -----------------------------
# Home
# -----------------------------
if role == "Home":
    st.subheader("Network")
    fig = draw_network(G, edge_data)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Instructions")
    st.markdown(
        f"""
1. Choose **Nash** or **Stackelberg** in the sidebar.  
2. Defender selects vehicles and writes routes as a sequence of nodes.  
3. Each route must go from **(0,0)** to **({GRID_SIZE-1},{GRID_SIZE-1})**.  
4. Attacker chooses **1 or 2 links** to attack.  
5. A link is identified by its two end nodes.  
6. Referee opens the results page to compare transport cost and attack damage.  
"""
    )

    st.subheader("Vehicle options")
    vehicle_df = pd.DataFrame([
        {"vehicle": name, "capacity_teu": spec["capacity"], "fixed_cost": spec["fixed_cost"]}
        for name, spec in VEHICLES.items()
    ])
    st.dataframe(vehicle_df, use_container_width=True)

# -----------------------------
# Defender
# -----------------------------
elif role == "Defender":
    st.subheader("Defender panel")
    st.write(f"Create shipment plans whose total volume equals **{TOTAL_TEU} TEU**.")

    fig = draw_network(G, edge_data)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Route instructions")
    st.markdown(
        f"""
Write a route as a sequence of nodes, for example:

`0,0 -> 0,1 -> 1,1 -> 2,1 -> 3,1 -> 4,1 -> 4,2 -> 4,3 -> 4,4`

Rules:
- must start at **(0,0)**
- must end at **({GRID_SIZE-1},{GRID_SIZE-1})**
- each step must move to an adjacent node
"""
    )

    if "defender_working_trips" not in st.session_state:
        st.session_state.defender_working_trips = []

    used_volume = sum(t["volume_teu"] for t in st.session_state.defender_working_trips)
    remaining = TOTAL_TEU - used_volume

    c1, c2 = st.columns([1, 2])
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
        route_text = st.text_area(
            "Route node sequence",
            value="0,0 -> 0,1 -> 1,1 -> 2,1 -> 3,1 -> 4,1 -> 4,2 -> 4,3 -> 4,4",
            height=120,
            disabled=remaining <= 0,
        )

    if st.button("Add trip", disabled=remaining <= 0):
        if volume > cap:
            st.error(f"{vehicle} cannot carry {volume} TEU.")
        elif volume > remaining:
            st.error("This trip exceeds the remaining volume.")
        else:
            try:
                nodes = parse_node_sequence(route_text)
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
                        "route_edges": route_edges,
                        "preview_transport_cost": calc["total_transport_cost"],
                    })
                    st.success("Trip added.")
            except Exception as e:
                st.error(f"Could not parse route. Error: {e}")

    if st.session_state.defender_working_trips:
        st.subheader("Current trips")
        preview_df = pd.DataFrame([
            {
                "shipment_id": t["shipment_id"],
                "vehicle": t["vehicle"],
                "volume_teu": t["volume_teu"],
                "edges_in_route": len(t["route_edges"]),
                "transport_cost_preview": t["preview_transport_cost"],
            }
            for t in st.session_state.defender_working_trips
        ])
        st.dataframe(preview_df, use_container_width=True)

        combined_edges = set()
        for t in st.session_state.defender_working_trips:
            combined_edges.update(t["route_edges"])
        fig = draw_network(G, edge_data, highlight_edges=combined_edges)
        st.pyplot(fig, clear_figure=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear current trips"):
            st.session_state.defender_working_trips = []
            st.success("Current trips cleared.")
    with col_b:
        if st.button("Save defender plan"):
            total_volume = sum(t["volume_teu"] for t in st.session_state.defender_working_trips)
            if total_volume != TOTAL_TEU:
                st.error(f"Total volume must equal {TOTAL_TEU} TEU. Current total is {total_volume}.")
            else:
                state["defender_submissions"] = st.session_state.defender_working_trips
                save_state(state)
                st.success("Defender plan saved.")

# -----------------------------
# Attacker
# -----------------------------
elif role == "Attacker":
    st.subheader("Attacker panel")
    defender_trips = state.get("defender_submissions", [])
    current_mode = state.get("game_mode", "Nash")

    if current_mode == "Stackelberg":
        if defender_trips:
            used_edges = set()
            for t in defender_trips:
                used_edges.update(t["route_edges"])
            st.success("Stackelberg mode: defender routes are visible to the attacker.")
            fig = draw_network(G, edge_data, highlight_edges=used_edges)
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
    
    valid_default_attacks = [
        e for e in state.get("attacks", [])
        if e in attack_options
    ][:attack_count]
    
    selected_attack_edges = st.multiselect(
        "Select links to attack",
        options=attack_options,
        format_func=lambda x: edge_label_from_key(x),
        max_selections=attack_count,
        default=valid_default_attacks,
    )

    if selected_attack_edges:
        fig = draw_network(G, edge_data, attack_edges=set(selected_attack_edges))
        st.pyplot(fig, clear_figure=True)

    if st.button("Save attack plan"):
        if len(selected_attack_edges) != attack_count:
            st.error(f"Please select exactly {attack_count} attacked link(s).")
        else:
            state["attacks"] = selected_attack_edges
            save_state(state)
            st.success("Attack plan saved.")

# -----------------------------
# Results
# -----------------------------
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

        used_edges = set()
        for t in defender_trips:
            used_edges.update(t["route_edges"])
        fig = draw_network(G, edge_data, highlight_edges=used_edges, attack_edges=set(attacks))
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