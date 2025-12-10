"""Small RKNN inspection helper.

This script reads an RKNN container (or plain RKNN JSON) and prints a concise
summary of its contents, including model metadata, graph inputs / outputs, and
per-node wiring.
"""

import argparse
import io
import json
import struct
from typing import Dict, Iterable, List, Tuple


def _read_uint64(stream: io.BytesIO) -> int:
    data = stream.read(8)
    if len(data) != 8:
        raise ValueError("Unexpected end of file while reading uint64.")
    return struct.unpack("<Q", data)[0]


def _detect_signature(buffer: bytes, length: int | None = None) -> str | None:
    length = len(buffer) if length is None else min(len(buffer), length)
    head = buffer[:length]
    if length >= 4 and head[:4] == b"RKNN":
        return "rknn"
    if length >= 8 and head[:8] == b"CYPTRKNN":
        return "cyptrknn"
    if length >= 8 and head[4:8] == b"RKNN":
        return "flatbuffers"
    if length >= 4 and head[:4] == b"VPMN":
        return "openvx"
    return None


def _normalize_node_connections(model: Dict) -> None:
    nodes: List[Dict] = model.get("nodes", [])
    for node in nodes:
        node.setdefault("input", [])
        node.setdefault("output", [])
    for connection in model.get("connection", []):
        left = connection.get("left")
        if left == "input":
            nodes[connection["node_id"]]["input"].append(connection)
            right_node = connection.get("right_node")
            if right_node:
                outputs = nodes[right_node["node_id"]].setdefault("output", [])
                tensor_id = right_node.get("tensor_id", 0)
                while len(outputs) <= tensor_id:
                    outputs.append(None)
                outputs[tensor_id] = connection
        elif left == "output":
            nodes[connection["node_id"]]["output"].append(connection)


def _connection_target(connection: Dict) -> str:
    tensor_ref = connection.get("right_tensor")
    if tensor_ref:
        tensor_type = tensor_ref.get("type", "tensor")
        tensor_id = tensor_ref.get("tensor_id", 0)
        return f"{tensor_type}:{tensor_id}"
    node_ref = connection.get("right_node")
    if node_ref:
        return f"node{node_ref.get('node_id', 0)}:{node_ref.get('tensor_id', 0)}"
    right = connection.get("right")
    if right is not None:
        return str(right)
    return "?"


def _describe_json_model(model: Dict) -> None:
    _normalize_node_connections(model)
    print("Model metadata")
    print("==============")
    print(f"Name:     {model.get('name', '')}")
    print(f"Version:  {model.get('version', '')}")
    platforms = model.get("target_platform") or model.get("network_platform")
    if platforms:
        if isinstance(platforms, list):
            platforms = ",".join(platforms)
        print(f"Platform: {platforms}")
    producer = model.get("ori_network_platform") or model.get("network_platform")
    if producer:
        print(f"Producer: {producer}")
    print(
        "Tensors:  const={const} virtual={virtual} norm={norm}".format(
            const=len(model.get("const_tensor", [])),
            virtual=len(model.get("virtual_tensor", [])),
            norm=len(model.get("norm_tensor", [])),
        )
    )
    print(f"Nodes:    {len(model.get('nodes', []))}")
    print()

    graph_ios: List[Tuple[str, str, str]] = []
    for entry in model.get("graph", []):
        right_name = f"{entry.get('right')}:{entry.get('right_tensor_id', 0)}"
        left_tensor_id = entry.get("left_tensor_id", 0)
        left_name = entry.get("left", "io") + (str(left_tensor_id) if left_tensor_id else "")
        graph_ios.append((entry.get("left", "io"), left_name, right_name))

    def _print_io(label: str, selector: str) -> None:
        filtered = [io_entry for io_entry in graph_ios if io_entry[0] == selector]
        if filtered:
            print(label)
            for _, name, target in filtered:
                print(f"  - {name:8s} -> {target}")
            print()

    _print_io("Graph inputs", "input")
    _print_io("Graph outputs", "output")

    print("Nodes")
    print("=====")
    for index, node in enumerate(model.get("nodes", [])):
        op_type = node.get("op", "?")
        name = node.get("name", f"node_{index}")
        print(f"[{index}] {name} ({op_type})")
        if node.get("input"):
            print("  inputs:")
            for connection in node.get("input", []):
                print(f"    - {_connection_target(connection)}")
        if node.get("output"):
            print("  outputs:")
            for connection in node.get("output", []):
                print(f"    - {_connection_target(connection)}")
        attributes = node.get("nn") or {}
        if attributes:
            print("  attributes:")
            for block_name, params in attributes.items():
                for attr_name, value in params.items():
                    print(f"    - {block_name}.{attr_name}: {value}")
        print()


def _parse_container(buffer: bytes) -> Tuple[str | None, Dict[str, bytes]]:
    signature = _detect_signature(buffer)
    if not signature:
        try:
            json.loads(buffer.decode("utf-8"))
            return "json", {"json": buffer}
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, {}
    if signature == "cyptrknn":
        raise ValueError("Encrypted RKNN files (CYPTRKNN) are not supported.")
    if signature in {"flatbuffers", "openvx"}:
        return signature, {signature: buffer}

    stream = io.BytesIO(buffer)
    stream.seek(8)
    version = _read_uint64(stream)
    data_size = _read_uint64(stream)
    if (version & 0xFF) > 1 and data_size > 0:
        stream.seek(stream.tell() + 40)
    data_start = stream.tell()
    inner_signature = _detect_signature(buffer[data_start : data_start + min(data_size, 16)], data_size)
    data_block = stream.read(data_size)
    json_size = _read_uint64(stream)
    json_block = stream.read(json_size)
    entries: Dict[str, bytes] = {"json": json_block}
    if inner_signature:
        entries[inner_signature] = data_block
    return "rknn", entries


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect RKNN containers and print graph details.")
    parser.add_argument("path", help="Path to an .rknn file or RKNN JSON dump.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    with open(args.path, "rb") as f:
        buffer = f.read()

    signature, entries = _parse_container(buffer)
    if not signature:
        raise SystemExit("Unrecognized RKNN container or JSON file.")

    print(f"Detected container type: {signature}")
    if "json" in entries:
        try:
            model = json.loads(entries["json"].decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise SystemExit(f"Failed to decode JSON block: {exc}")
        _describe_json_model(model)
    else:
        print("This container does not embed a JSON graph (flatbuffers/openvx only).")


if __name__ == "__main__":
    main()
