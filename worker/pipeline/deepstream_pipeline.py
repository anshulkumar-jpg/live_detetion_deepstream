"""DeepStream pipeline group (one logical worker).

    nvurisrcbin(cam0) --\\                                                   /-- queue-nvvideoconvert-nvdsosd-nvvideoconvert-nvv4l2h264enc-h264parse-rtspclientsink (cam0 annotated)
    nvurisrcbin(cam1) ---> nvstreammux -> nvinfer(PGIE..) -> nvtracker -> nvinfer(SGIE..) -> queue -> nvstreamdemux
                                                                   (probe: metadata -> Kafka)        \\-- queue-fakesink (cam1, no stream output)

* NVDEC: nvurisrcbin decodes with nvv4l2decoder into NVMM (GPU) memory; frames stay on
  the GPU through mux, inference, tracking, OSD and NVENC.
* Sources are added/removed at runtime (request pads on nvstreammux/nvstreamdemux),
  so adding or removing a camera does not interrupt the other camera in the group.
* RTSP reconnection: nvurisrcbin ``rtsp-reconnect-interval`` handles transport
  reconnects; a stall watchdog marks the source RECONNECTING and, if the bin gives
  up (error), the source is torn down and re-added after a backoff.

Requires the DeepStream runtime (GStreamer NV plugins + pyds). Import fails
cleanly elsewhere; the logic that decides events lives in events.py.
"""
from __future__ import annotations

import ctypes
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

from app.core.security import inject_credentials  # noqa: E402
from app.core.trackers import is_python_tracker  # noqa: E402
from worker.tracking import make_tracker  # noqa: E402
from worker.pipeline.config_gen import tracker_properties, write_stage_configs  # noqa: E402
from worker.pipeline.events import EventEmitter, FrameResult, ObjectResult, SourceStats  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class SourceSlot:
    index: int
    camera_id: str
    spec: dict[str, Any]
    bin: Any = None
    branch: list[Any] = field(default_factory=list)
    demux_pad: Any = None
    status: str = "STARTING"  # STARTING | RUNNING | RECONNECTING | FAILED
    reason: str | None = None
    stats: SourceStats | None = None
    readd_at: float | None = None


class PipelineError(RuntimeError):
    def __init__(self, msg: str, model_id: str | None = None, model_version_id: int | None = None):
        super().__init__(msg)
        self.model_id = model_id
        self.model_version_id = model_version_id


class PipelineGroup:
    def __init__(self, spec: dict[str, Any], settings, *, decrypt: Callable[[str | None], str | None],
                 emitter: EventEmitter, on_event: Callable[[str, dict[str, Any]], None], config_dir: str):
        Gst.init(None)
        self.spec = spec
        self.s = settings
        self.decrypt = decrypt
        self.emitter = emitter
        self.on_event = on_event
        self.config_dir = config_dir
        self.gpu = int(spec["gpu_index"])
        self.capacity = int(spec["capacity"])
        self.plan = spec["plan"]
        self.slots: dict[int, SourceSlot] = {}
        self.by_camera: dict[str, SourceSlot] = {}
        self.lock = threading.RLock()
        self.pipeline = Gst.Pipeline.new(f"pipeline-{spec['worker_id']}")
        self.loop = GLib.MainLoop()
        self.mux_w, self.mux_h = settings.streammux_width, settings.streammux_height
        self.gie_to_stage = {st["gie_unique_id"]: st for st in self.plan["stages"]}
        self.reid_gies = {st["gie_unique_id"] for st in self.plan["stages"] if st["task"] == "reid"}
        # Secondary YOLO models run on person crops: stage hint "x-vms-decode: yolo_classes" means the model
        # outputs [N, 6] = x1,y1,x2,y2,score,class and the best class becomes a label on the parent object.
        self.posture_gies = {
            st["gie_unique_id"] for st in self.plan["stages"]
            if (st.get("nvinfer_properties") or {}).get("x-vms-decode") == "yolo_classes"
        }
        # ByteTrack / OC-SORT / BoT-SORT / StrongSORT: pure-python trackers, one instance per source (camera)
        self.py_tracking = bool(self.plan["tracking_enabled"]) and is_python_tracker(self.plan["tracker_type"])
        self.py_trackers: dict[int, Any] = {}
        self.primary_gies = {st["gie_unique_id"] for st in self.plan["stages"] if st["process_mode"] == 1}
        self.fatal: PipelineError | None = None
        self._build_core()

    # ------------------------------------------------------------- build
    def _make(self, factory: str, name: str | None = None, **props) -> Any:
        el = Gst.ElementFactory.make(factory, name)
        if el is None:
            raise PipelineError(f"GStreamer element '{factory}' is unavailable (is DeepStream installed?)")
        for k, v in props.items():
            el.set_property(k.rstrip("_").replace("_", "-"), v)
        self.pipeline.add(el)
        return el

    def _build_core(self) -> None:
        self.mux = self._make("nvstreammux", "mux", batch_size=self.capacity, width=self.mux_w, height=self.mux_h,
                              batched_push_timeout=self.s.streammux_batched_push_timeout_us, live_source=True,
                              gpu_id=self.gpu, attach_sys_ts=True)
        chain = [self.mux]
        cfgs = write_stage_configs(self.plan, self.gpu, self.s.model_repository_dir, self.config_dir)
        self.gie_elements: dict[str, int] = {}

        def infer(st):
            factory = "nvinferserver" if st["backend"] == "nvinferserver" else "nvinfer"
            name = f"gie{st['gie_unique_id']}_{st['model_id']}"
            el = self._make(factory, name, config_file_path=cfgs[st["gie_unique_id"]])
            if factory == "nvinfer":
                el.set_property("gpu-id", self.gpu)
            self.gie_elements[name] = st["gie_unique_id"]
            return el

        for st in self.plan["stages"]:
            if st["process_mode"] == 1:
                chain.append(infer(st))
        if self.plan["tracking_enabled"] and not self.py_tracking:  # python trackers run in the metadata probe
            trk = self._make("nvtracker", "tracker")
            for k, v in tracker_properties(self.plan["tracker_type"], self.s, self.gpu).items():
                trk.set_property(k, v)
            chain.append(trk)
        for st in self.plan["stages"]:
            if st["process_mode"] == 2:
                chain.append(infer(st))
        q = self._make("queue", "meta_queue", leaky=2, max_size_buffers=8)
        chain.append(q)
        self.demux = self._make("nvstreamdemux", "demux")
        chain.append(self.demux)
        for a, b in zip(chain, chain[1:]):
            if not a.link(b):
                raise PipelineError(f"failed to link {a.get_name()} -> {b.get_name()}")
        q.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, self._metadata_probe, None)
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus)

    # ------------------------------------------------------------- sources
    def _free_index(self) -> int:
        for i in range(self.capacity):
            if i not in self.slots:
                return i
        raise PipelineError("worker is at capacity")

    def add_source(self, src: dict[str, Any]) -> None:
        with self.lock:
            cam = src["camera_id"]
            if cam in self.by_camera:
                return  # idempotent
            idx = self._free_index()
            slot = SourceSlot(idx, cam, src, stats=SourceStats(src.get("fps") or 30))
            self.slots[idx] = slot
            self.by_camera[cam] = slot
            self._attach(slot)
            self.on_event("SOURCE_ADDED", {"camera_id": cam, "source_index": idx})

    def _attach(self, slot: SourceSlot) -> None:
        src = slot.spec
        uri = inject_credentials(src["uri"], self.decrypt(src.get("username_enc")), self.decrypt(src.get("password_enc")))
        b = Gst.ElementFactory.make("nvurisrcbin", f"src_{slot.index}")
        if b is None:
            raise PipelineError("nvurisrcbin unavailable")
        b.set_property("uri", uri)
        b.set_property("gpu-id", self.gpu)
        b.set_property("rtsp-reconnect-interval", self.s.rtsp_reconnect_interval_s)
        b.set_property("rtsp-reconnect-attempts", self.s.rtsp_reconnect_attempts)
        b.set_property("latency", self.s.rtsp_latency_ms)
        if uri.startswith("rtsp") and self.s.rtsp_tcp:
            b.set_property("select-rtp-protocol", 4)  # TCP only
        if uri.startswith("file://"):
            b.set_property("file-loop", True)  # benchmarks / validation with sample clips
        b.connect("pad-added", self._on_src_pad, slot)
        self.pipeline.add(b)
        slot.bin = b
        self._attach_output(slot)
        b.sync_state_with_parent()

    def _on_src_pad(self, _bin, pad, slot: SourceSlot) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps.get_structure(0).get_name().startswith("video"):
            return
        sinkpad = self.mux.request_pad_simple(f"sink_{slot.index}") if hasattr(self.mux, "request_pad_simple") \
            else self.mux.get_request_pad(f"sink_{slot.index}")
        if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
            log.error("failed to link source to mux", extra={"camera_id": slot.camera_id})

    def _attach_output(self, slot: SourceSlot) -> None:
        """Per-camera branch after demux: annotated NVENC stream or a fakesink."""
        i, cam = slot.index, slot.camera_id
        dpad = self.demux.request_pad_simple(f"src_{i}") if hasattr(self.demux, "request_pad_simple") \
            else self.demux.get_request_pad(f"src_{i}")
        q = self._make("queue", f"outq_{i}", leaky=2, max_size_buffers=4)
        elems = [q]
        if slot.spec.get("annotated_output"):
            conv1 = self._make("nvvideoconvert", f"conv1_{i}", gpu_id=self.gpu)
            osd = self._make("nvdsosd", f"osd_{i}", gpu_id=self.gpu, process_mode=1)
            conv2 = self._make("nvvideoconvert", f"conv2_{i}", gpu_id=self.gpu)
            caps = self._make("capsfilter", f"caps_{i}", caps=Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
            enc = self._make("nvv4l2h264enc", f"enc_{i}", bitrate=self.s.encoder_bitrate_bps,
                             iframeinterval=int(slot.spec.get("fps") or 30), gpu_id=self.gpu)
            try:
                enc.set_property("insert-sps-pps", True)
                enc.set_property("idrinterval", int(slot.spec.get("fps") or 30))
            except TypeError:
                pass  # property names differ on Jetson encoders
            parse = self._make("h264parse", f"parse_{i}", config_interval=-1)
            if self.s.stream_push_protocol == "rtmp":
                mux = self._make("flvmux", f"flv_{i}", streamable=True)
                q_auth = f"?user=worker&pass={self.s.internal_service_token}" if self.s.internal_service_token else ""
                sink = self._make("rtmpsink", f"sink_{i}",
                                  location=f"{self.s.mediamtx_rtmp_push_url}/{slot.spec['output_path']}{q_auth}",
                                  sync=False, async_=False)
                elems += [conv1, osd, conv2, caps, enc, parse, mux, sink]
            else:
                sink = self._make("rtspclientsink", f"sink_{i}",
                                  location=f"{self.s.mediamtx_rtsp_push_url}/{slot.spec['output_path']}",
                                  protocols="tcp")
                if self.s.internal_service_token:  # MediaMTX http-auth checks this on publish
                    sink.set_property("user-id", "worker")
                    sink.set_property("user-pw", self.s.internal_service_token)
                elems += [conv1, osd, conv2, caps, enc, parse, sink]
        else:
            elems.append(self._make("fakesink", f"sink_{i}", sync=False, async_=False))
        for a, b in zip(elems, elems[1:]):
            if not a.link(b):
                raise PipelineError(f"failed to link output branch for {cam}: {a.get_name()} -> {b.get_name()}")
        if dpad.link(q.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
            raise PipelineError(f"failed to link demux pad for {cam}")
        slot.branch = elems
        slot.demux_pad = dpad
        for e in elems:
            e.sync_state_with_parent()

    def remove_source(self, camera_id: str, reason: str = "removed", readd: bool = False) -> None:
        with self.lock:
            slot = self.by_camera.get(camera_id)
            if slot is None:
                if not readd:
                    self.on_event("SOURCE_REMOVED", {"camera_id": camera_id, "reason": "not present"})
                return
            self._detach(slot)
            if readd:
                slot.status, slot.reason = "RECONNECTING", reason
                slot.readd_at = time.monotonic() + self.s.source_readd_backoff_s
                return
            del self.slots[slot.index]
            del self.by_camera[camera_id]
            self.emitter.forget_camera(camera_id)
            self.on_event("SOURCE_REMOVED", {"camera_id": camera_id, "reason": reason})

    def _detach(self, slot: SourceSlot) -> None:
        self.py_trackers.pop(slot.index, None)  # forget python-tracker state (new camera must not inherit old ids)
        if slot.bin is not None:
            slot.bin.set_state(Gst.State.NULL)
            sinkpad = self.mux.get_static_pad(f"sink_{slot.index}")
            if sinkpad is not None:
                sinkpad.send_event(Gst.Event.new_flush_stop(False))
                self.mux.release_request_pad(sinkpad)
            self.pipeline.remove(slot.bin)
            slot.bin = None
        if slot.demux_pad is not None:
            if slot.branch:
                slot.demux_pad.unlink(slot.branch[0].get_static_pad("sink"))
            self.demux.release_request_pad(slot.demux_pad)
            slot.demux_pad = None
        for e in slot.branch:
            e.set_state(Gst.State.NULL)
            self.pipeline.remove(e)
        slot.branch = []

    # ------------------------------------------------------------- metadata
    def _metadata_probe(self, pad, info, _data):
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        try:
            batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            now = time.monotonic()
            now_ns = time.time_ns()
            l_frame = batch.frame_meta_list
            while l_frame is not None:
                fm = pyds.NvDsFrameMeta.cast(l_frame.data)
                slot = self.slots.get(fm.pad_index)
                if slot is not None:
                    lat = (now_ns - fm.ntp_timestamp) / 1e6 if fm.ntp_timestamp else None
                    slot.stats.on_frame(now, lat)
                    if slot.status != "RUNNING":
                        slot.status, slot.reason = "RUNNING", None
                        self.on_event("SOURCE_RUNNING", {"camera_id": slot.camera_id})
                    if self.py_tracking:
                        self._py_track(fm)
                    self.emitter.process(slot.camera_id, self._frame_result(fm), self.mux_w, self.mux_h)
                l_frame = l_frame.next
        except Exception:  # never let analytics break the video path
            log.exception("metadata probe error")
        return Gst.PadProbeReturn.OK

    def _embedding(self, om):
        """Re-ID embedding of one object (from a 'reid' secondary model), or None."""
        l_user = om.obj_user_meta_list
        while l_user is not None:
            um = pyds.NvDsUserMeta.cast(l_user.data)
            if um.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                tm = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
                if tm.unique_id in self.reid_gies:
                    layer = pyds.get_nvds_LayerInfo(tm, 0)
                    dim = self.gie_to_stage[tm.unique_id].get("embedding_dim") or self.s.embedding_dim
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    return [float(ptr[k]) for k in range(dim)]
            l_user = l_user.next
        return None

    def _py_track(self, fm) -> None:
        """Run the python tracker of this source on the detector's boxes and write the track ids back."""
        import numpy as np

        tracker = self.py_trackers.get(fm.pad_index)
        if tracker is None:
            tracker = self.py_trackers[fm.pad_index] = make_tracker(self.plan["tracker_type"])
        objs, boxes, scores, classes, feats = [], [], [], [], []
        l_obj = fm.obj_meta_list
        while l_obj is not None:
            om = pyds.NvDsObjectMeta.cast(l_obj.data)
            if om.unique_component_id in self.primary_gies:
                r = om.rect_params
                objs.append(om)
                boxes.append([r.left, r.top, r.left + r.width, r.top + r.height])
                scores.append(float(om.confidence))
                classes.append(int(om.class_id))
                if self.reid_gies:
                    feats.append(self._embedding(om))
            l_obj = l_obj.next
        f = None
        if feats and all(x is not None for x in feats):
            f = np.asarray(feats, dtype=float)
        ids = tracker.update(np.asarray(boxes, dtype=float).reshape(-1, 4), np.asarray(scores), np.asarray(classes), f)
        for om, tid in zip(objs, ids):
            if tid > 0:
                om.object_id = int(tid)
                om.text_params.display_text = f"{om.obj_label or 'obj'} {int(tid)}"

    POSTURE_COLORS = {"standing": (0.0, 0.85, 0.0), "sitting": (1.0, 0.6, 0.0), "fallen": (1.0, 0.0, 0.0)}

    def _apply_posture(self, om, o: ObjectResult) -> None:
        """Label one person with standing / sitting / fallen and colour the box.

        1. Use the secondary YOLO model (tensor on the person crop) when it is confident (score >= x-vms-min-score).
        2. Otherwise estimate from the box shape: tall+narrow = standing, wide+low = fallen, else sitting.
        """
        for gie in self.posture_gies:
            stage = self.gie_to_stage[gie]
            if om.unique_component_id != stage.get("operate_on_gie_id"):
                continue  # only the parent detector's person boxes
            props = stage.get("nvinfer_properties") or {}
            labels = stage.get("labels") or []
            min_score = float(props.get("x-vms-min-score", 0.35))
            if stage.get("conf_threshold"):  # the camera set its own confidence for this model
                min_score = float(stage["conf_threshold"])
            label, score, source = None, 0.0, "est"
            l_user = om.obj_user_meta_list
            while l_user is not None:
                um = pyds.NvDsUserMeta.cast(l_user.data)
                if um.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tm = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
                    if tm.unique_id == gie:
                        layer = pyds.get_nvds_LayerInfo(tm, 0)
                        n = int(layer.inferDims.d[0])
                        ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                        best, best_cls = 0.0, -1
                        for k in range(n):
                            sc = ptr[k * 6 + 4]
                            if sc > best:
                                best, best_cls = sc, int(ptr[k * 6 + 5])
                        if 0 <= best_cls < len(labels) and best >= min_score:
                            label, score, source = labels[best_cls], float(best), "model"
                l_user = l_user.next
            if label is None:
                w, h = om.rect_params.width, om.rect_params.height
                if w <= 0 or h <= 0:
                    return
                if h / w >= float(props.get("x-vms-standing-ratio", 2.0)):
                    label = "standing"
                elif w / h >= float(props.get("x-vms-fallen-ratio", 1.3)):
                    label = "fallen"
                else:
                    label = "sitting"
            picked = stage.get("detect_class_ids")  # postures the camera wants highlighted (None = all)
            if picked and label not in {labels[i] for i in picked if i < len(labels)}:
                return
            o.classifications.append({"gie_id": gie, "label": label, "class_id": -1, "prob": score, "source": source})
            base = f"person {int(om.object_id)}" if om.object_id < 2**63 else "person"
            om.text_params.display_text = f"{base} {label.upper()}"
            col = self.POSTURE_COLORS.get(label)
            if col:
                om.rect_params.border_color.red, om.rect_params.border_color.green = col[0], col[1]
                om.rect_params.border_color.blue, om.rect_params.border_color.alpha = col[2], 1.0
                om.text_params.font_params.font_size = 14
            return

    def _frame_result(self, fm) -> FrameResult:
        objs = []
        l_obj = fm.obj_meta_list
        while l_obj is not None:
            om = pyds.NvDsObjectMeta.cast(l_obj.data)
            r = om.rect_params
            o = ObjectResult(
                gie_id=om.unique_component_id, class_id=om.class_id, label=om.obj_label or None,
                confidence=float(om.confidence), tracker_id=int(om.object_id), left=r.left, top=r.top,
                width=r.width, height=r.height,
                parent_tracker_id=int(om.parent.object_id) if om.parent is not None else None,
            )
            l_cls = om.classifier_meta_list
            while l_cls is not None:
                cm = pyds.NvDsClassifierMeta.cast(l_cls.data)
                l_lbl = cm.label_info_list
                while l_lbl is not None:
                    li = pyds.NvDsLabelInfo.cast(l_lbl.data)
                    o.classifications.append({"gie_id": cm.unique_component_id, "label": li.result_label,
                                              "class_id": li.result_class_id, "prob": float(li.result_prob)})
                    l_lbl = l_lbl.next
                l_cls = l_cls.next
            if self.posture_gies:
                self._apply_posture(om, o)
            if self.reid_gies:
                l_user = om.obj_user_meta_list
                while l_user is not None:
                    um = pyds.NvDsUserMeta.cast(l_user.data)
                    if um.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                        tm = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
                        if tm.unique_id in self.reid_gies:
                            layer = pyds.get_nvds_LayerInfo(tm, 0)
                            dim = self.gie_to_stage[tm.unique_id].get("embedding_dim") or self.s.embedding_dim
                            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                            o.embeddings[tm.unique_id] = [float(ptr[k]) for k in range(dim)]
                    l_user = l_user.next
            objs.append(o)
            l_obj = l_obj.next
        return FrameResult(source_id=fm.pad_index, frame_num=fm.frame_num, ntp_ns=fm.ntp_timestamp,
                           src_width=fm.source_frame_width, src_height=fm.source_frame_height, objects=objs)

    # ------------------------------------------------------------- bus / watchdog
    def _slot_for_element(self, el) -> SourceSlot | None:
        while el is not None:
            name = el.get_name() or ""
            if name.startswith("src_"):
                try:
                    return self.slots.get(int(name.split("_")[1]))
                except ValueError:
                    return None
            el = el.get_parent()
        return None

    def _gie_for_element(self, el) -> dict | None:
        while el is not None:
            gid = self.gie_elements.get(el.get_name() or "")
            if gid is not None:
                return self.gie_to_stage[gid]
            el = el.get_parent()
        return None

    def _on_bus(self, _bus, msg) -> None:
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            slot = self._slot_for_element(msg.src)
            if slot is not None:
                # source-level failure: isolate it, keep other camera running, retry later
                log.warning("source error: %s", err.message, extra={"camera_id": slot.camera_id})
                self.on_event("SOURCE_RECONNECTING", {"camera_id": slot.camera_id, "reason": err.message[:300]})
                GLib.idle_add(lambda: (self.remove_source(slot.camera_id, err.message[:200], readd=True), False)[1])
                return
            stage = self._gie_for_element(msg.src)
            if stage is not None:
                self.fatal = PipelineError(f"model load/inference error: {err.message}", stage["model_id"],
                                           stage["model_version_id"])
            else:
                self.fatal = PipelineError(f"{msg.src.get_name()}: {err.message}")
            log.error("pipeline error: %s (%s)", err.message, dbg)
            self.loop.quit()
        elif t == Gst.MessageType.WARNING:
            w, _ = msg.parse_warning()
            log.warning("pipeline warning from %s: %s", msg.src.get_name(), w.message)

    def watchdog(self) -> bool:
        """Runs every second in the GLib loop: stall detection + delayed re-add."""
        now = time.monotonic()
        with self.lock:
            for slot in list(self.slots.values()):
                if slot.readd_at is not None:
                    if now >= slot.readd_at:
                        slot.readd_at = None
                        try:
                            self._attach(slot)
                            slot.status = "STARTING"
                        except PipelineError as e:
                            slot.readd_at = now + self.s.source_readd_backoff_s
                            slot.reason = str(e)
                    continue
                last = slot.stats.last_frame_at
                if slot.status == "RUNNING" and last is not None and now - last > self.s.source_stall_timeout_s:
                    slot.status, slot.reason = "RECONNECTING", f"no frames for {now - last:.0f}s"
                    self.on_event("SOURCE_RECONNECTING", {"camera_id": slot.camera_id, "reason": slot.reason})
        return True

    # ------------------------------------------------------------- lifecycle
    def start(self, sources: list[dict[str, Any]]) -> None:
        for src in sources:
            self.add_source(src)
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise PipelineError("pipeline failed to enter PLAYING (check model engines and GPU)")
        GLib.timeout_add_seconds(1, self.watchdog)

    def run(self) -> None:
        self.loop.run()

    def stop(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)
        if self.loop.is_running():
            self.loop.quit()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            sources = {}
            for s in self.slots.values():
                sources[s.camera_id] = {"status": s.status, "reason": s.reason, "fps": round(s.stats.fps(), 2),
                                        "input_fps": s.spec.get("fps"), "latency_ms": s.stats.latency_ms(),
                                        "frames": s.stats.frames, "deficit_frames": int(s.stats.deficit_frames)}
        return sources
