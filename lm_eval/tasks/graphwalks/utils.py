import re

import datasets
from huggingface_hub import hf_hub_download


RWKV_PARENT_QUERY_REPEATS = 20


def _rwkv_graph(doc):
    prompt = doc["prompt"]
    graph_section = prompt.split("Here is the graph to operate on:", 1)[1]
    edge_section, operation_section = graph_section.split("\n\n\nOperation:\n", 1)
    operation = operation_section.split("\n\nYou should", 1)[0].strip()

    edges = []
    nodes = []
    seen_edges = set()
    for line in edge_section.splitlines():
        match = re.fullmatch(r"(\S+)\s+->\s+(\S+)", line.strip())
        if not match or match.groups() in seen_edges:
            continue
        edge = match.groups()
        seen_edges.add(edge)
        edges.append(edge)
        for node in edge:
            if node not in nodes:
                nodes.append(node)
    return operation, edges, nodes


def _rwkv_parent_prompt(operation, edges, nodes):
    labels = {node: str(index) for index, node in enumerate(nodes)}
    target_match = re.search(r"parents of node\s+(\S+)", operation, re.IGNORECASE)
    if target_match is None:
        raise ValueError(f"Unsupported GraphWalks parents operation: {operation}")
    target = target_match.group(1).rstrip(".,")
    target_label = labels[target]

    incoming = {}
    for source, destination in edges:
        source_label = labels[source]
        destination_label = labels[destination]
        if source_label == destination_label:
            continue
        sources = incoming.setdefault(destination_label, [])
        if source_label not in sources:
            sources.append(source_label)

    query_sources = incoming.get(target_label, [])
    index_text = "\n".join(
        f"Node {destination} has parent labels [{', '.join(sources)}]."
        for destination, sources in incoming.items()
        if destination != target_label
    )
    query_record = (
        f"Query record: node {target_label} has parent labels "
        f"[{', '.join(query_sources)}]."
    )
    query_records = "\n".join([query_record] * RWKV_PARENT_QUERY_REPEATS)
    empty_instruction = (
        "The query record is empty, so output exactly Final Answer: []."
        if not query_sources
        else "Copy every label inside the query record brackets, once each."
    )
    return (
        "Complete incoming-edge index using temporary integer labels:\n"
        f"{index_text}\n"
        f"{query_records}\n"
        f"{empty_instruction}\n"
        f"Question: Find the parents of node {target_label}; never return node "
        f"{target_label} itself. Return only one line in this format: "
        "Final Answer: [comma-separated labels]"
    )


def _rwkv_bfs_depth(operation):
    depth_match = re.search(r"depth\s+(\d+)", operation, re.IGNORECASE)
    if depth_match is None:
        raise ValueError(f"Unsupported GraphWalks BFS operation: {operation}")
    return int(depth_match.group(1))


def _rwkv_depth_one_bfs_prompt(operation, edges, nodes):
    labels = {node: str(index) for index, node in enumerate(nodes)}
    start_match = re.search(r"BFS from node\s+(\S+)", operation, re.IGNORECASE)
    if start_match is None:
        raise ValueError(f"Unsupported GraphWalks BFS operation: {operation}")
    start = start_match.group(1).rstrip(".,")
    start_label = labels[start]

    outgoing = {}
    for source, destination in edges:
        source_label = labels[source]
        destination_label = labels[destination]
        if source_label == destination_label:
            continue
        destinations = outgoing.setdefault(source_label, [])
        if destination_label not in destinations:
            destinations.append(destination_label)

    query_destinations = outgoing.get(start_label, [])
    index_text = "\n".join(
        f"Node {source} has outgoing labels [{', '.join(destinations)}]."
        for source, destinations in outgoing.items()
        if source != start_label
    )
    query_record = (
        f"Query record: node {start_label} has outgoing labels "
        f"[{', '.join(query_destinations)}]."
    )
    query_records = "\n".join([query_record] * RWKV_PARENT_QUERY_REPEATS)
    return (
        "Complete outgoing-edge index using temporary integer labels:\n"
        f"{index_text}\n"
        f"{query_records}\n"
        "Copy every label inside the query record brackets, once each. "
        f"Question: Perform BFS from node {start_label} at depth 1; never return "
        f"node {start_label} itself. Return only one line in this format: "
        "Final Answer: [comma-separated labels]"
    )


def _rwkv_bfs_prompt(operation, edges, nodes):
    if _rwkv_bfs_depth(operation) == 1:
        return _rwkv_depth_one_bfs_prompt(operation, edges, nodes)

    adjacency = {}
    for source, target in edges:
        targets = adjacency.setdefault(source, [])
        if target not in targets:
            targets.append(target)

    adjacency_text = "\n".join(
        f"{source} -> [{', '.join(targets)}]"
        for source, targets in adjacency.items()
    )
    return (
        "Execute this graph operation exactly.\n"
        f"Operation: {operation}\n\n"
        "Outgoing adjacency list:\n"
        f"{adjacency_text}\n\n"
        f"Operation: {operation}\n"
        "For parents collect every unique source with an edge to the target. "
        "For BFS follow outgoing edges for exactly the requested depth and return "
        "only the final layer. Never return the starting node. Return only this "
        "format with no explanation:\n"
        "Final Answer: [comma-separated unique node IDs]"
    )


def doc_to_text_rwkv(doc):
    operation, edges, nodes = _rwkv_graph(doc)
    if operation.casefold().startswith("find the parents"):
        return _rwkv_parent_prompt(operation, edges, nodes)
    return _rwkv_bfs_prompt(operation, edges, nodes)


def load_dataset(**kwargs):
    """
    Load the graphwalks dataset with specific data file.

    Args:
        kwargs: Must contain 'data_file' key specifying which parquet file to load

    Returns:
        Dictionary with 'train' split containing the dataset
    """
    data_file = kwargs.get("data_file")
    if not data_file:
        raise ValueError("data_file must be specified in dataset_kwargs")

    parquet_path = hf_hub_download(
        repo_id="openai/graphwalks",
        filename=data_file,
        repo_type="dataset",
        revision=kwargs.get("revision"),
    )
    dataset = datasets.load_dataset(
        "parquet", data_files={"train": parquet_path}, split="train"
    )
    return {"train": dataset}


def extract_answer_list(response: str) -> tuple[list[str], bool]:
    """
    Extract the answer list from a model response.

    Args:
        response: The model's generated response

    Returns:
        Tuple of (list of nodes, is_error)
        - list of nodes: extracted node IDs
        - is_error: True if parsing failed, False otherwise
    """
    # Get the very last line of the response (strip trailing newlines first)
    line = response.rstrip("\n").split("\n")[-1]

    # Check if formatted correctly
    if "Final Answer:" not in line:
        return [], True

    # Extract the list part using regex with capturing group
    match = re.search(r"Final Answer:\s*\[(.*)\]", line)
    if match:
        # Extract content between brackets using group(1)
        bracket_content = match.group(1)
        # Handle empty list case
        if not bracket_content.strip():
            return [], False
        # Split by comma and clean up whitespace and quotes
        result_list = [
            item.strip().strip("'\"")
            for item in bracket_content.split(",")
            if item.strip()
        ]
        return result_list, False
    else:
        return [], True


def extract_answer_list_flexible(response: str) -> tuple[list[str], bool]:
    """
    Extract the answer list from a model response (flexible version).
    Searches backwards through all lines to find "Final Answer:" pattern.
    More lenient than extract_answer_list which only checks the last line.

    Args:
        response: The model's generated response

    Returns:
        Tuple of (list of nodes, is_error)
        - list of nodes: extracted node IDs
        - is_error: True if parsing failed, False otherwise
    """
    lines = response.rstrip("\n").split("\n")
    for line in reversed(lines):
        match = re.search(r"Final Answer:\s*\[(.*)\]", line)
        if match:
            # Extract content between brackets using group(1)
            bracket_content = match.group(1)
            # Handle empty list case
            if not bracket_content.strip():
                return [], False
            # Split by comma and clean up whitespace and quotes
            result_list = [
                item.strip().strip("'\"")
                for item in bracket_content.split(",")
                if item.strip()
            ]
            return result_list, False

    # No "Final Answer:" found anywhere
    return [], True


def process_results(doc, results):
    """
    Process results and compute set-based F1 scores.
    Returns both strict F1 (last line only) and flexible F1 (search all lines).

    Args:
        doc: Document containing ground truth answer_nodes
        results: List containing model generation

    Returns:
        Dictionary with f1 and flexible_f1 scores
    """
    # Extract model response (first element of results)
    response = results[0]

    # Get ground truth nodes
    gold_nodes = doc["answer_nodes"]

    # Parse the response using strict extraction
    predicted_nodes_strict, _ = extract_answer_list(response)
    sampled_set_strict = set(predicted_nodes_strict)
    truth_set = set(gold_nodes)

    # Calculate strict F1
    n_overlap_strict = len(sampled_set_strict & truth_set)
    n_sampled_strict = len(sampled_set_strict)
    n_golden = len(truth_set)

    recall_strict = n_overlap_strict / n_golden if n_golden > 0 else 0.0
    precision_strict = (
        n_overlap_strict / n_sampled_strict if n_sampled_strict > 0 else 0.0
    )
    f1_strict = (
        2 * (recall_strict * precision_strict) / (recall_strict + precision_strict)
        if (recall_strict + precision_strict) > 0
        else 0.0
    )

    # Parse the response using flexible extraction
    predicted_nodes_flexible, _ = extract_answer_list_flexible(response)
    sampled_set_flexible = set(predicted_nodes_flexible)

    # Calculate flexible F1
    n_overlap_flexible = len(sampled_set_flexible & truth_set)
    n_sampled_flexible = len(sampled_set_flexible)

    recall_flexible = n_overlap_flexible / n_golden if n_golden > 0 else 0.0
    precision_flexible = (
        n_overlap_flexible / n_sampled_flexible if n_sampled_flexible > 0 else 0.0
    )
    f1_flexible = (
        2
        * (recall_flexible * precision_flexible)
        / (recall_flexible + precision_flexible)
        if (recall_flexible + precision_flexible) > 0
        else 0.0
    )

    return {
        "f1": f1_strict,
        "flexible_f1": f1_flexible,
    }


def process_results_rwkv(doc, results):
    operation, _, nodes = _rwkv_graph(doc)
    uses_integer_labels = operation.casefold().startswith(
        "find the parents"
    ) or _rwkv_bfs_depth(operation) == 1
    if not uses_integer_labels:
        return process_results(doc, results)

    label_to_node = {str(index): node for index, node in enumerate(nodes)}
    response = results[0]
    strict_labels, _ = extract_answer_list(response)
    flexible_labels, _ = extract_answer_list_flexible(response)
    gold_nodes = set(doc["answer_nodes"])

    def score(labels):
        predicted_nodes = {
            label_to_node[label] for label in labels if label in label_to_node
        }
        overlap = len(predicted_nodes & gold_nodes)
        recall = overlap / len(gold_nodes) if gold_nodes else 0.0
        precision = overlap / len(predicted_nodes) if predicted_nodes else 0.0
        return (
            2 * recall * precision / (recall + precision)
            if recall + precision
            else 0.0
        )

    return {
        "f1": score(strict_labels),
        "flexible_f1": score(flexible_labels),
    }
