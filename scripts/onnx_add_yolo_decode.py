"""Append a decode head to a raw Ultralytics YOLO ONNX ([B, 4+C, N] -> [B, N, 6] = x1,y1,x2,y2,score,class).

This is the output layout expected by the DeepStream-Yolo `NvDsInferParseYolo` custom parser.
usage: python scripts/onnx_add_yolo_decode.py in.onnx out.onnx
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh


def main(src: str, dst: str) -> None:
    m = onnx.load(src)
    g = m.graph
    raw = g.output[0].name
    nodes, inits = [], []

    def const(name, arr):
        inits.append(nh.from_array(np.asarray(arr), name))
        return name

    nodes.append(h.make_node("Transpose", [raw], ["t"], perm=[0, 2, 1]))                       # [B,N,4+C]
    nodes.append(h.make_node("Slice", ["t", const("s0", [0]), const("e4", [4]), const("ax2", [2])], ["box"]))
    nodes.append(h.make_node("Slice", ["t", const("s4", [4]), const("eM", [2**31 - 1]), "ax2"], ["sc"]))
    nodes.append(h.make_node("Split", ["box"], ["cx", "cy", "w", "hh"], axis=2))
    two = const("two", np.float32(2.0))
    for n, c, d in (("hw", "w", None), ("hh2", "hh", None)):
        nodes.append(h.make_node("Div", [c, two], [n]))
    nodes += [h.make_node("Sub", ["cx", "hw"], ["x1"]), h.make_node("Sub", ["cy", "hh2"], ["y1"]),
              h.make_node("Add", ["cx", "hw"], ["x2"]), h.make_node("Add", ["cy", "hh2"], ["y2"]),
              h.make_node("ReduceMax", ["sc"], ["conf"], axes=[2], keepdims=1),
              h.make_node("ArgMax", ["sc"], ["cls_i"], axis=2, keepdims=1),
              h.make_node("Cast", ["cls_i"], ["cls"], to=T.FLOAT),
              h.make_node("Concat", ["x1", "y1", "x2", "y2", "conf", "cls"], ["decoded"], axis=2)]
    g.node.extend(nodes)
    g.initializer.extend(inits)
    del g.output[:]
    g.output.extend([h.make_tensor_value_info("decoded", T.FLOAT, ["batch", "boxes", 6])])
    m.opset_import[0].version = max(m.opset_import[0].version, 17)
    onnx.checker.check_model(m)
    onnx.save(m, dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
