import sys, numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference


def get_dims(vi):
    if not vi.type.tensor_type.shape.dim:
        return []
    return [d.dim_value if d.HasField("dim_value") else 0
            for d in vi.type.tensor_type.shape.dim]


def get_opset(model, domain=""):
    for imp in model.opset_import:
        if imp.domain == domain:
            return imp.version
    return onnx.defs.onnx_opset_version()


def fuse_yolo_mask_postprocessing_into_onnx(in_path):
    out_path = in_path.replace(".onnx", "-fused.onnx")
    
    model = onnx.load(in_path)
    g = model.graph
    opset = get_opset(model, "")

    # ---- Identify outputs by shape: det [1,K,L], proto [1,C,H,W]
    if len(g.output) < 2:
        raise RuntimeError("Expected 2 graph outputs: detections and prototypes.")
    outA, outB = g.output[:2]
    dA, dB = get_dims(outA), get_dims(outB)

    def is_det(d): return len(d) == 3 and d[-1] >= 8
    def is_proto(d): return len(d) == 4 and (d[1] in (0, 32))

    if is_det(dA) and is_proto(dB):
        det_name, proto_name = outA.name, outB.name
        det_dims, proto_dims = dA, dB
    elif is_det(dB) and is_proto(dA):
        det_name, proto_name = outB.name, outA.name
        det_dims, proto_dims = dB, dA
    else:
        raise RuntimeError(f"Could not identify outputs by shape. Got {dA} and {dB}.")

    B = det_dims[0] or 1
    K = det_dims[1] or 300
    L = det_dims[2] or 38
    C = proto_dims[1] or 32
    H = proto_dims[2] or 160
    W = proto_dims[3] or 160

    if L < 6 + C:
        raise RuntimeError(f"Det last dim {L} too small for 6 meta + {C} coeffs.")

    new_nodes = []

    # ---- Split [1,K,L] -> meta:[1,K,6], coeffs:[1,K,C]
    if opset >= 13:
        split_sizes = numpy_helper.from_array(np.array([6, C], np.int64), name="_split_sizes")
        g.initializer.append(split_sizes)
        new_nodes.append(helper.make_node("Split", [det_name, "_split_sizes"], ["_det_meta", "_det_coeff"], axis=2))
    else:
        new_nodes.append(helper.make_node("Split", [det_name], ["_det_meta", "_det_coeff"], axis=2, split=[6, C]))

    # ---- Squeeze proto batch: [1,C,H,W] -> [C,H,W]
    if opset >= 13:
        axes0 = numpy_helper.from_array(np.array([0], np.int64), name="_axes0")
        g.initializer.append(axes0)
        new_nodes.append(helper.make_node("Squeeze", [proto_name, "_axes0"], ["_proto_squeezed"]))
    else:
        new_nodes.append(helper.make_node("Squeeze", [proto_name], ["_proto_squeezed"], axes=[0]))

    # ---- Flatten proto spatially: [C,H,W] -> [C, H*W]
    new_nodes.append(helper.make_node("Flatten", ["_proto_squeezed"], ["_proto_flat"], axis=1))  # spec: axis is up to which dims to flatten.  [oai_citation:2‡ONNX](https://onnx.ai/onnx/operators/onnx__Flatten.html?utm_source=chatgpt.com)

    # ---- MatMul: [1,K,C] @ [C,H*W] -> [1,K,H*W]
    new_nodes.append(helper.make_node("MatMul", ["_det_coeff", "_proto_flat"], ["_masks_flat"]))

    # ---- Reshape to [1,K,H,W]
    mask_shape = numpy_helper.from_array(np.array([B, K, H, W], np.int64), name="_masks_shape")
    g.initializer.append(mask_shape)
    new_nodes.append(helper.make_node("Reshape", ["_masks_flat", "_masks_shape"], ["_masks_reshaped"]))

    # ---- Sigmoid to get probabilities
    new_nodes.append(helper.make_node("Sigmoid", ["_masks_reshaped"], ["_masks"]))

    # ==============================
    # ROI CROP (zero outside boxes)
    # ==============================

    # Find 4D image input to read (Hi, Wi)
    img_in = None
    img_in_dims = None
    for vi in g.input:
        dims = get_dims(vi)
        if len(dims) == 4:
            img_in, img_in_dims = vi.name, dims
            break
    if img_in is None:
        raise RuntimeError("Could not find 4D model input (image).")

    # 1) Slice xyxy from _det_meta: [1,K,6] -> [1,K,4] (last dim)
    sl_st = numpy_helper.from_array(np.array([0], np.int64), name="_sl_b_st")
    sl_en = numpy_helper.from_array(np.array([4], np.int64), name="_sl_b_en")
    sl_ax = numpy_helper.from_array(np.array([2], np.int64), name="_sl_b_ax")
    sl_sp = numpy_helper.from_array(np.array([1], np.int64), name="_sl_b_sp")
    g.initializer.extend([sl_st, sl_en, sl_ax, sl_sp])
    new_nodes.append(helper.make_node("Slice", ["_det_meta", "_sl_b_st", "_sl_b_en", "_sl_b_ax", "_sl_b_sp"],
                                      ["_boxes_raw"]))  # ONNX Slice inputs (starts/ends/axes/steps).  [oai_citation:3‡ONNX](https://onnx.ai/onnx/operators/onnx__Slice.html?utm_source=chatgpt.com)

    # 2) Scale to proto grid: wr=Wm/Wi, hr=Hm/Hi  (constant fast-path if shapes known)
    Hi_const, Wi_const = img_in_dims[2] or 0, img_in_dims[3] or 0
    dyn = []

    if Wi_const > 0 and Hi_const > 0 and W > 0 and H > 0:
        wr_vec = "_wr_c"; hr_vec = "_hr_c"
        g.initializer.extend([
            numpy_helper.from_array(np.array([float(W) / float(Wi_const)], np.float32), name=wr_vec),
            numpy_helper.from_array(np.array([float(H) / float(Hi_const)], np.float32), name=hr_vec),
        ])
    else:
        # dynamic path
        dyn += [helper.make_node("Shape", [img_in], ["_shape_in"])]
        g2_in = numpy_helper.from_array(np.array([2], np.int64), name="_g2_in")
        g3_in = numpy_helper.from_array(np.array([3], np.int64), name="_g3_in")
        g.initializer.extend([g2_in, g3_in])
        dyn += [
            helper.make_node("Gather", ["_shape_in", "_g2_in"], ["_Hi"]),
            helper.make_node("Gather", ["_shape_in", "_g3_in"], ["_Wi"]),
            helper.make_node("Shape", [proto_name], ["_shape_p"]),
        ]
        g2_p = numpy_helper.from_array(np.array([2], np.int64), name="_g2_p")
        g3_p = numpy_helper.from_array(np.array([3], np.int64), name="_g3_p")
        g.initializer.extend([g2_p, g3_p])
        dyn += [
            helper.make_node("Gather", ["_shape_p", "_g2_p"], ["_Hm"]),
            helper.make_node("Gather", ["_shape_p", "_g3_p"], ["_Wm"]),
            helper.make_node("Cast", ["_Wi"], ["_Wi_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["_Hi"], ["_Hi_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["_Wm"], ["_Wm_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["_Hm"], ["_Hm_f"], to=TensorProto.FLOAT),
            helper.make_node("Div", ["_Wm_f", "_Wi_f"], ["_wr_s"]),
            helper.make_node("Div", ["_Hm_f", "_Hi_f"], ["_hr_s"]),
        ]
        ax0 = numpy_helper.from_array(np.array([0], np.int64), name="_ax0")
        g.initializer.append(ax0)
        dyn += [
            helper.make_node("Unsqueeze", ["_wr_s", "_ax0"], ["_wr_c"]),
            helper.make_node("Unsqueeze", ["_hr_s", "_ax0"], ["_hr_c"]),
        ]
        wr_vec, hr_vec = "_wr_c", "_hr_c"
    new_nodes += dyn

    # Build [wr, hr, wr, hr] -> [1,1,4] and scale boxes
    new_nodes.append(helper.make_node("Concat", [wr_vec, hr_vec, wr_vec, hr_vec], ["_scale_4"], axis=0))
    ax01 = numpy_helper.from_array(np.array([0, 1], np.int64), name="_ax01")
    g.initializer.append(ax01)
    new_nodes.append(helper.make_node("Unsqueeze", ["_scale_4", "_ax01"], ["_scale_114"]))  # [1,1,4]
    new_nodes.append(helper.make_node("Mul", ["_boxes_raw", "_scale_114"], ["_boxes_p"]))   # [1,K,4]

    # 3) Rectangle crop on proto grid

    # Split xyxy into components (each [1,K,1])
    split4 = numpy_helper.from_array(np.array([1, 1, 1, 1], np.int64), name="_split4")
    g.initializer.append(split4)
    new_nodes.append(helper.make_node("Split", ["_boxes_p", "_split4"], ["_x1", "_y1", "_x2", "_y2"], axis=2))

    # Coordinate ramps: x:[W] → [1,1,1,W], y:[H] → [1,1,H,1]
    if W <= 0 or H <= 0:
        raise RuntimeError("Prototype W/H unknown; export with static mask size or set defaults.")
    g.initializer.extend([
        numpy_helper.from_array(np.arange(W, dtype=np.float32), name="_x_coords"),
        numpy_helper.from_array(np.arange(H, dtype=np.float32), name="_y_coords"),
    ])
    ax012 = numpy_helper.from_array(np.array([0, 1, 2], np.int64), name="_ax012")
    ax013 = numpy_helper.from_array(np.array([0, 1, 3], np.int64), name="_ax013")
    g.initializer.extend([ax012, ax013])
    new_nodes += [
        helper.make_node("Unsqueeze", ["_x_coords", "_ax012"], ["_xr"]),  # [1,1,1,W]
        helper.make_node("Unsqueeze", ["_y_coords", "_ax013"], ["_yr"]),  # [1,1,H,1]
    ]

    # *** KEY FIX: Unsqueeze boxes to [1,K,1,1] so broadcasting matches ***
    ax3 = numpy_helper.from_array(np.array([3], np.int64), name="_ax3")
    g.initializer.append(ax3)
    new_nodes += [
        helper.make_node("Unsqueeze", ["_x1", "_ax3"], ["_x1_4d"]),  # [1,K,1,1]
        helper.make_node("Unsqueeze", ["_x2", "_ax3"], ["_x2_4d"]),
        helper.make_node("Unsqueeze", ["_y1", "_ax3"], ["_y1_4d"]),
        helper.make_node("Unsqueeze", ["_y2", "_ax3"], ["_y2_4d"]),
    ]

    # Compare into rectangle masks (Numpy-style broadcasting per ONNX spec)  [oai_citation:4‡ONNX](https://onnx.ai/onnx/operators/onnx__Less.html?utm_source=chatgpt.com)
    new_nodes += [
        helper.make_node("GreaterOrEqual", ["_xr", "_x1_4d"], ["_cx1"]),   # [1,K,1,W]
        helper.make_node("Less",           ["_xr", "_x2_4d"], ["_cx2"]),
        helper.make_node("And",            ["_cx1", "_cx2"],  ["_condx"]),
        helper.make_node("GreaterOrEqual", ["_yr", "_y1_4d"], ["_cy1"]),   # [1,K,H,1]
        helper.make_node("Less",           ["_yr", "_y2_4d"], ["_cy2"]),
        helper.make_node("And",            ["_cy1", "_cy2"],  ["_condy"]),
        helper.make_node("And",            ["_condx", "_condy"], ["_roi_bool"]),  # [1,K,H,W]
        helper.make_node("Cast",           ["_roi_bool"], ["_roi_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul",            ["_masks", "_roi_f"], ["_masks_cropped"]),
    ]

    # Final outputs
    g.node.extend(new_nodes)
    del g.output[:]
    g.output.extend([
        helper.make_tensor_value_info("_det_meta", TensorProto.FLOAT, [B, K, 6]),
        helper.make_tensor_value_info("_masks_cropped", TensorProto.FLOAT, [B, K, H, W]),
    ])

    # Save as a single binary .onnx (no external data shards)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save_model(model, out_path, save_as_external_data=False)
    print(f"Saved {out_path}\nOutputs: _det_meta=({B},{K},6), _masks_cropped=({B},{K},{H},{W})")

    return out_path