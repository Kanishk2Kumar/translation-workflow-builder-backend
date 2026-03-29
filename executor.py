from collections import defaultdict
from nodes.registry import NODE_REGISTRY

# Lower number = runs earlier
NODE_PRIORITY = {
    "document_upload": 0,
    "text_input":      0,
    "rag_tm":          1,   # must run BEFORE llm_agent
    "vector_db":       1,   # must run BEFORE llm_agent
    "llm_agent":       2,
    "translation":     2,
    "compliance":      3,
    "comet_qe":        3,
    "output":          4,
    "document_output": 4,
    "learning":        5,
}


def build_execution_order(nodes: list[dict], edges: list[dict]) -> list[str]:
    node_ids = {n["id"] for n in nodes}
    node_type_map = {
        n["id"]: n.get("data", {}).get("nodeType", "") for n in nodes
    }

    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src in node_ids and tgt in node_ids:
            adjacency[src].append(tgt)
            in_degree[tgt] += 1

    def priority(nid: str) -> int:
        return NODE_PRIORITY.get(node_type_map.get(nid, ""), 99)

    # Start with all zero-in-degree nodes, sorted by priority
    ready = sorted(
        [nid for nid, deg in in_degree.items() if deg == 0],
        key=priority,
    )

    order = []

    while ready:
        # Always pick the highest-priority (lowest number) ready node
        nid = ready.pop(0)
        order.append(nid)

        for neighbour in adjacency[nid]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                ready.append(neighbour)
                ready.sort(key=priority)

    if len(order) != len(node_ids):
        raise ValueError("Workflow graph has a cycle — cannot execute.")

    return order


async def execute_workflow(
    nodes: list[dict],
    edges: list[dict],
    initial_context: dict,
) -> dict:
    order = build_execution_order(nodes, edges)
    node_map = {n["id"]: n for n in nodes}
    context = {**initial_context, "_logs": []}

    for node_id in order:
        node_def = node_map[node_id]
        node_data = node_def.get("data", {})
        node_type = node_data.get("nodeType", "")
        node_config = node_data.get("config", {})

        NodeClass = NODE_REGISTRY.get(node_type)

        if NodeClass is None:
            context["_logs"].append({
                "node_id": node_id,
                "node_type": node_type,
                "status": "skipped",
                "reason": f"No handler for nodeType '{node_type}'",
            })
            continue

        instance = NodeClass(node_id=node_id, config=node_config)
        context = await instance.execute(context)

    return context