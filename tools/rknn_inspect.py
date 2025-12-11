"""Small RKNN inspection helper.

This script reads an RKNN container (or plain RKNN JSON) and prints a concise
summary of its contents, including model metadata, graph inputs / outputs,
per-node wiring, and nested graphs when present.
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


def _parse_openvx(buffer: bytes) -> Dict:
    reader = io.BytesIO(buffer)

    def _read_uint16() -> int:
        data = reader.read(2)
        if len(data) != 2:
            raise ValueError("Unexpected EOF in OpenVX header.")
        return struct.unpack("<H", data)[0]

    def _read_uint32() -> int:
        data = reader.read(4)
        if len(data) != 4:
            raise ValueError("Unexpected EOF in OpenVX header.")
        return struct.unpack("<I", data)[0]

    reader.seek(4)  # signature
    major = _read_uint16()
    _ = _read_uint16()  # minor
    reader.seek(reader.tell() + 4)
    name = reader.read(64).split(b"\x00", 1)[0].decode("ascii", errors="ignore")
    node_count = _read_uint32()
    if major > 3:
        reader.seek(reader.tell() + 296)
    elif major > 1:
        reader.seek(reader.tell() + 288)
    else:
        reader.seek(reader.tell() + 32)

    _ = _read_uint32()  # inputOffset
    _ = _read_uint32()  # inputSize
    _ = _read_uint32()  # outputOffset
    _ = _read_uint32()  # outputSize
    node_offset = _read_uint32()
    _ = _read_uint32()  # nodeSize

    reader.seek(node_offset)
    nodes: List[Dict[str, object]] = []
    for _ in range(node_count):
        type_bytes = reader.read(64)
        node_type = type_bytes.split(b"\x00", 1)[0].decode("ascii", errors="ignore")
        index = _read_uint32()
        c_val = _read_uint32()
        d_val = _read_uint32() if major > 3 else None
        node: Dict[str, object] = {"type": node_type, "index": index, "c": c_val}
        if d_val is not None:
            node["d"] = d_val
        nodes.append(node)

    return {"name": name, "version_major": major, "nodes": nodes}


def _parse_flatbuffers(buffer: bytes) -> Dict:
    data = memoryview(buffer)

    def read_u32(offset: int) -> int:
        return struct.unpack_from("<I", data, offset)[0]

    def read_u16(offset: int) -> int:
        return struct.unpack_from("<H", data, offset)[0]

    def field_offset(table: int, index: int) -> int:
        vtable = table - read_u32(table)
        vlen = read_u16(vtable)
        entry = vtable + 4 + index * 2
        if entry >= vtable + vlen:
            return 0
        return read_u16(entry)

    def field_uoffset(table: int, index: int) -> int | None:
        off = field_offset(table, index)
        if off == 0:
            return None
        return table + off

    def read_string(uoffset: int) -> str:
        base = uoffset + read_u32(uoffset)
        strlen = read_u32(base)
        start = base + 4
        return bytes(data[start : start + strlen]).decode("utf-8", errors="ignore")

    def read_vector(uoffset: int) -> Tuple[int, int]:
        base = uoffset + read_u32(uoffset)
        length = read_u32(base)
        return base + 4, length

    def read_vector_ints(uoffset: int) -> List[int]:
        start, length = read_vector(uoffset)
        return list(struct.unpack_from(f"<{length}i", data, start)) if length else []

    root = read_u32(0)
    if buffer[4:8] != b"RKNN":
        raise ValueError("FlatBuffers RKNN block missing identifier.")

    model_table = root
    graphs_vec = field_uoffset(model_table, 2)
    graphs: List[Dict] = []
    tensors: List[Dict] = []
    nodes: List[Dict] = []
    if graphs_vec is not None:
        start, length = read_vector(graphs_vec)
        for i in range(length):
            graph_offset = start + i * 4
            graph_table = graph_offset + read_u32(graph_offset)
            graph_tensors: List[Dict] = []
            graph_nodes: List[Dict] = []
            tensor_vec = field_uoffset(graph_table, 0)
            if tensor_vec is not None:
                t_start, t_len = read_vector(tensor_vec)
                for j in range(t_len):
                    tensor_off = t_start + j * 4
                    tensor_table = tensor_off + read_u32(tensor_off)
                    tensor_name = "tensor" + str(j)
                    name_offset = field_uoffset(tensor_table, 5)
                    if name_offset is not None:
                        tensor_name = read_string(name_offset)
                    shape_offset = field_uoffset(tensor_table, 4)
                    shape = read_vector_ints(shape_offset) if shape_offset is not None else []
                    index_offset = field_uoffset(tensor_table, 18)
                    tensor_index = j
                    if index_offset is not None:
                        tensor_index = struct.unpack_from("<i", data, index_offset + read_u32(index_offset))[0]
                    tensor_entry = {"index": tensor_index, "name": tensor_name, "shape": shape}
                    tensors.append(tensor_entry)
                    graph_tensors.append(tensor_entry)
            node_vec = field_uoffset(graph_table, 1)
            if node_vec is not None:
                n_start, n_len = read_vector(node_vec)
                for j in range(n_len):
                    node_off = n_start + j * 4
                    node_table = node_off + read_u32(node_off)
                    node_type = ""
                    type_offset = field_uoffset(node_table, 1)
                    if type_offset is not None:
                        node_type = read_string(type_offset)
                    node_name = f"node{j}"
                    name_offset = field_uoffset(node_table, 2)
                    if name_offset is not None:
                        node_name = read_string(name_offset)
                    inputs_offset = field_uoffset(node_table, 4)
                    outputs_offset = field_uoffset(node_table, 5)
                    inputs = read_vector_ints(inputs_offset) if inputs_offset is not None else []
                    outputs = read_vector_ints(outputs_offset) if outputs_offset is not None else []
                    node_entry = {
                        "type": node_type,
                        "name": node_name,
                        "inputs": inputs,
                        "outputs": outputs,
                    }
                    nodes.append(node_entry)
                    graph_nodes.append(node_entry)
            graphs.append({"tensors": graph_tensors, "nodes": graph_nodes})

    return {"graphs": graphs, "tensors": tensors, "nodes": nodes}


def _describe_json_model(model: Dict, container_entries: Dict[str, bytes]) -> None:
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
        if op_type in {"VSI_NN_OP_NBG", "RKNN_OP_NNBG"}:
            if "openvx" in container_entries:
                print("  expanded (openvx):")
                subgraph = _parse_openvx(container_entries["openvx"])
                for sub_index, sub_node in enumerate(subgraph.get("nodes", [])):
                    print(
                        "    - [{:02d}] {:20s} idx={} c={}{}".format(
                            sub_index,
                            sub_node.get("type", ""),
                            sub_node.get("index", "-"),
                            sub_node.get("c", "-"),
                            f" d={sub_node.get('d')}" if "d" in sub_node else "",
                        )
                    )
            elif "flatbuffers" in container_entries:
                print("  expanded (flatbuffers):")
                fb = _parse_flatbuffers(container_entries["flatbuffers"])
                tensors = {t.get("index", i): t for i, t in enumerate(fb.get("tensors", []))}
                for sub_index, sub_node in enumerate(fb.get("nodes", [])):
                    print(f"    - [{sub_index:02d}] {sub_node.get('name', '')} ({sub_node.get('type', '')})")
                    if sub_node.get("inputs"):
                        print("       inputs:")
                        for tidx in sub_node["inputs"]:
                            tensor = tensors.get(tidx)
                            if tensor:
                                shape = "x".join(str(x) for x in tensor.get("shape", []))
                                print(f"         - {tensor.get('name', 'tensor'+str(tidx))} [{shape}]")
                            else:
                                print(f"         - tensor{tidx}")
                    if sub_node.get("outputs"):
                        print("       outputs:")
                        for tidx in sub_node["outputs"]:
                            tensor = tensors.get(tidx)
                            if tensor:
                                shape = "x".join(str(x) for x in tensor.get("shape", []))
                                print(f"         - {tensor.get('name', 'tensor'+str(tidx))} [{shape}]")
                            else:
                                print(f"         - tensor{tidx}")
            else:
                print("  expanded: no nested graph data found in container.")
        print()


def _parse_container(buffer: bytes) -> Tuple[str | None, Dict[str, bytes], Dict[str, int]]:
    signature = _detect_signature(buffer)
    if not signature:
        try:
            json.loads(buffer.decode("utf-8"))
            return "json", {"json": buffer}, {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, {}, {}
    if signature == "cyptrknn":
        raise ValueError("Encrypted RKNN files (CYPTRKNN) are not supported.")
    if signature in {"flatbuffers", "openvx"}:
        return signature, {signature: buffer}, {}

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
    container_info = {
        "version": version,
        "data_size": data_size,
        "json_size": json_size,
    }
    return "rknn", entries, container_info


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect RKNN containers and print graph details.")
    parser.add_argument("path", help="Path to an .rknn file or RKNN JSON dump.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    with open(args.path, "rb") as f:
        buffer = f.read()

    signature, entries, container_info = _parse_container(buffer)
    if not signature:
        raise SystemExit("Unrecognized RKNN container or JSON file.")

    print(f"Detected container type: {signature}")
    if container_info:
        version = container_info.get("version")
        print(
            "Container blocks: data_size={data} bytes, json_size={json} bytes, version=0x{version:x}".format(
                data=container_info.get("data_size"),
                json=container_info.get("json_size"),
                version=version if version is not None else 0,
            )
        )
        embedded = [name for name in entries if name != "json"]
        if embedded:
            print(f"Embedded binary sections: {', '.join(embedded)}")
    if "json" in entries:
        try:
            model = json.loads(entries["json"].decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise SystemExit(f"Failed to decode JSON block: {exc}")
        _describe_json_model(model, entries)
    else:
        print("This container does not embed a JSON graph (flatbuffers/openvx only).")


if __name__ == "__main__":
    main()
