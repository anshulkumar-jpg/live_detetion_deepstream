# Architecture: how the whole project works, from the basics

This guide explains what every part of the project does, how a video frame travels through it, and what happens at each step. It starts with the idea, then the pieces, then the life of one camera and one frame.

Contents

1. [What the project does](#1-what-the-project-does)
2. [Basic words you need](#2-basic-words-you-need)
3. [The big picture](#3-the-big-picture)
4. [The services (Docker containers)](#4-the-services-docker-containers)
5. [What happens when you add a camera](#5-what-happens-when-you-add-a-camera)
6. [The life of one frame](#6-the-life-of-one-frame)
7. [How the detector decides what is a person, car or fall](#7-how-the-detector-decides)
8. [Tracking: giving objects an ID](#8-tracking-giving-objects-an-id)
9. [Per-camera settings and where they take effect](#9-per-camera-settings)
10. [After the frame: events, storage, analytics](#10-after-the-frame-events-storage-analytics)
11. [How the video reaches your browser](#11-how-the-video-reaches-your-browser)
12. [Processes and crash safety](#12-processes-and-crash-safety)
13. [Security](#13-security)
14. [Start-up and the helper scripts](#14-start-up-and-the-helper-scripts)
15. [Where things are in the folders](#15-where-things-are-in-the-folders)
16. [Limits and honest notes](#16-limits-and-honest-notes)

---

## 1. What the project does

You have IP cameras (here, the channels of a recorder called an NVR). This project:

1. connects to each camera and reads its live video,
2. runs AI models on the video on an NVIDIA GPU (find people, vehicles, standing / sitting / fallen),
3. follows each object from frame to frame and gives it an ID,
4. draws boxes and labels on the video and streams it to a browser, live,
5. sends a small message ("person 7 seen on camera 5") to a message system so other programs can store and analyse it.

The goal is to do this for many cameras at once, safely, without one camera's trouble stopping the others.

---

## 2. Basic words you need

| Word | Simple meaning |
|---|---|
| **Frame** | One picture. A camera sends about 25 frames every second. Video is a fast series of frames. |
| **RTSP** | The protocol a camera or NVR uses to send live video over the network. Address looks like `rtsp://ip:554/...`. |
| **H.264 / H.265** | How video is compressed so it fits on the network. A frame must be *decoded* (unpacked) before AI can use it. |
| **Decode / encode** | Decode = unpack compressed video into pictures. Encode = pack pictures back into compressed video. The GPU has special chips for both: **NVDEC** (decode) and **NVENC** (encode). |
| **GPU memory** | The graphics card's own memory. Moving pictures to normal RAM and back is slow, so the project keeps frames in GPU memory the whole time. |
| **Batch** | Several frames processed together (here up to 2 cameras at once), because the GPU is faster that way. |
| **Model** | The trained AI. Input: a picture. Output: boxes with a class and a score. |
| **ONNX / TensorRT engine** | ONNX is the model in a portable file. A **TensorRT engine** is that model rebuilt and optimised for your exact GPU. It takes about 5 minutes to build once and is then loaded in seconds. |
| **Inference** | Running the model on a picture. |
| **Class** | What an object is: person, car, bus... |
| **Confidence** | How sure the model is (0 to 1). A box is shown only if it is above your chosen limit. |
| **NMS and IoU** | The model often draws several boxes on the same object. NMS keeps the best one and removes the others that overlap too much. **IoU** is the overlap measure; the IoU setting is "how much overlap counts as the same object". |
| **Tracker** | Gives an object the same ID in the next frame, so "person 7" stays "person 7". |
| **Cascade** | Run a second model only on what the first model found (for example posture on each person box). |
| **Kafka** | A message system. Programs drop messages into named lists called *topics*; other programs read them. |
| **DeepStream** | NVIDIA's toolkit that does steps 1 to 4 above on the GPU as a *pipeline* of connected parts. |
| **Pipeline** | A chain of parts. A frame enters at one end and leaves at the other, each part doing one job. |

---

## 3. The big picture

```
  CAMERAS / NVR                                         YOU
  (RTSP video)                              browser / scripts / API calls
       |                                                   |
       |                                                   v
       |                                     +-------------------------------+
       |                                     |  ENVOY gateway  (https :8443) |
       |                                     +---------------+---------------+
       |                                        /api/        |      /hls/ /webrtc/
       |                                          v          |           v
       |                                 +--------------+    |    +--------------+
       |                                 |  API         |    |    |  MediaMTX    |  video server
       |                                 |  (FastAPI)   |<---+----|  WebRTC/HLS  |
       |                                 +------+-------+  auth   +------^-------+
       |                                        | reads/writes           | annotated video
       |                                        v                        | (RTSP push)
       |                                 +--------------+                |
       |                                 | PostgreSQL   |  cameras, models, settings, state
       |                                 +------+-------+                |
       |                                        ^                        |
       |                          desired state | every 2 s              |
       |                                 +------+-------+                |
       |                                 |  SCHEDULER   |  decides which camera runs where
       |                                 +------+-------+                |
       |                                        | commands (Kafka: pipeline.commands)
       |                                        v                        |
       |   RTSP    +------------------------------------------------------+--------+
       +---------->|  GPU AGENT (one per GPU server)                               |
                   |   +---------------------------------------------------------+ |
                   |   | WORKER process  (one DeepStream pipeline, 1-2 cameras)  | |
                   |   |  decode -> batch -> detect -> track -> posture -> draw  | |
                   |   |  -> encode ------------------------------------------>-+-+
                   |   +----------------------+----------------------------------+ |
                   +---------------------------|------------------------------------+
                                               | events (Kafka: detection.events ...)
                                               v
                          +----------------------------------------+
                          |  KAFKA                                 |
                          +---+-------------+----------------+-----+
                              v             v                v
                     analytics-stream     MTMC (optional)   Logstash (optional)
                     -> PostgreSQL        -> Milvus         -> Elasticsearch -> Kibana
```

The system has four jobs, kept apart on purpose:

| Part | Job | Lives in |
|---|---|---|
| **Control** | Remember what you want, decide where it runs | API, PostgreSQL, Scheduler |
| **Video processing** | Read video, run the AI, draw, re-encode | GPU agent and its worker processes |
| **Streaming** | Hand the finished video to browsers | MediaMTX, Envoy |
| **Analytics** | Store and analyse what was detected | Kafka, analytics-stream, PostgreSQL (and optional MTMC / Elasticsearch) |

---

## 4. The services (Docker containers)

Everything runs in Docker containers, started by `docker-compose.yml`.

| Container | What it does | Port |
|---|---|---|
| **postgres** | The database. Cameras, models, settings, workers, saved events. | internal |
| **kafka** | Message system between the parts. | internal |
| **migrate** | Runs once at start: creates or updates the database tables (Alembic migrations), then exits. | none |
| **api** | The control API (FastAPI). Login, cameras, models, workers, GPU status, analytics queries. Also tells the video server who may watch or publish. | internal 8000 |
| **scheduler** | The "manager". Every 2 seconds compares *what you want* (database) with *what is running* and sends commands to fix any difference. | metrics 9101 |
| **gpu-agent** | Runs on the GPU machine. Starts and stops worker processes, reports GPU health. Contains DeepStream. | metrics 9102 |
| **analytics-stream** | Reads detection messages and writes track-level summaries to the database. | metrics 9103 |
| **mediamtx** | The video server. Receives finished video from workers, serves it to browsers. | RTSP 8554, WebRTC media 8189 |
| **envoy** | The single front door with HTTPS. Sends `/api/` to the API and `/hls/`, `/webrtc/` to the video server. Blocks `/internal/`. | **8443** |

Optional, started only on request:

| Profile | Containers | Purpose |
|---|---|---|
| analytics | mtmc, milvus, etcd, elasticsearch, logstash, kibana | Follow the same person between cameras (MTMC), search events, dashboards |
| monitoring | prometheus, grafana, kafka-exporter | Graphs of GPU, fps, latency |

Outside Docker, two small helpers run on your PC: `scripts/live_viewer.py` (the live page on port 8080) and the batch files.

---

## 5. What happens when you add a camera

Example: you run `add_camera.bat` and add channel 5 with the person model.

```
 1. add_camera.py   asks IP, user, password, channels, model, classes, tracker,
                    detection frequency, confidence / IoU
 2.                 tests the RTSP address with ffprobe (one login attempt, stops on a wrong password)
 3. -> API          POST /cameras           camera saved in PostgreSQL (password encrypted)
                    POST /cameras/{id}/models   which model(s) + your settings saved
                    POST /cameras/{id}/start    desired_state = RUNNING
 4. scheduler       (within 2 s) sees: camera wants to run, nothing is running it
 5.   - builds the "plan": which models, which tracker, which thresholds
 6.   - computes the pipeline signature  (a fingerprint of the plan)
 7.   - looks for an existing worker with the same signature and a free slot
        yes -> ADD_SOURCE (adds the camera to the running pipeline; other cameras are not touched)
        no  -> START_WORKER (a new worker)
 8.   - asks the GPU manager: is there room on the GPU? (admission control)
        no  -> camera waits in PENDING_CAPACITY with a reason
 9.   - sends the command through Kafka (topic pipeline.commands) to the right GPU server
10. gpu-agent       receives it, starts a worker.runner process (or forwards ADD_SOURCE to it)
11. worker          writes the DeepStream config files, loads the model engines, builds the pipeline,
                    connects to the camera. Status goes back: STARTING -> RUNNING
12. API / scripts   show the camera as RUNNING with its fps and lag
```

Why the "signature" matters: two cameras can share one worker only if they use exactly the same models, tracker, classes and thresholds. Cameras with different settings get separate workers. A worker holds up to `CAMERAS_PER_WORKER` (default 2) cameras.

Why the database is the source of truth: if a worker crashes, a message is lost, or you restart everything, the scheduler simply compares the database with reality again and repairs the difference.

---

## 6. The life of one frame

This is the heart of the project. Every frame goes through the same pipeline inside a worker. The video itself stays in **GPU memory** from decode to encode. The Python code never touches the pixels; it only reads small **metadata** (boxes, classes, IDs) attached to each frame.

```
 camera
   |  RTSP over TCP, compressed H.264/H.265
   v
 [1] nvurisrcbin ......... connect to the camera, receive packets, DECODE on the GPU (NVDEC)
   |                       -> one raw frame in GPU memory
   v
 [2] nvstreammux ......... collect frames from the cameras of this worker into one BATCH,
   |                       scale to 1920x1080, stamp time. Waits at most 40 ms for the batch.
   v
 [3] nvinfer (primary) ... DETECTOR: find objects (person / vehicle ...)       <- AI model #1
   |                       attaches a box + class + confidence to the frame as metadata
   v
 [4] nvtracker ........... give every box an ID that stays the same across frames
   |                       (or a Python tracker later in the probe, see section 8)
   v
 [5] nvinfer (secondary) . CASCADE (optional): crop each person, run posture model   <- AI model #2
   |
   v
 [6] probe (Python) ...... read the metadata, count fps and lag, posture labels,
   |                       build events and send them to Kafka (never slows the video)
   v
 [7] nvstreamdemux ....... split the batch back into one stream per camera
   |
   +--> camera A branch -> [8] nvvideoconvert -> nvdsosd (DRAW boxes + text on the GPU)
   |                          -> nvvideoconvert -> NVENC H.264 encode -> RTSP push to MediaMTX
   +--> camera B branch -> same
```

### Step by step

**[1] Reading and decoding (`nvurisrcbin`)**
The camera sends compressed video over RTSP using TCP. A small buffer of 200 ms (`RTSP_LATENCY_MS`) smooths out network jitter. The GPU's NVDEC chip unpacks each frame into raw pixels in GPU memory. If the camera stops sending for 10 seconds, the source reconnects by itself.

**[2] Batching (`nvstreammux`)**
Frames from the (up to two) cameras of a worker are put together as one batch and resized to 1920 by 1080, so the rest of the pipeline sees a fixed size. It waits up to 40 ms (`STREAMMUX_BATCHED_PUSH_TIMEOUT_US`) for all cameras, then moves on. Each frame gets a timestamp, which is how the "LAG" number in the status table is measured: the time from entering this batcher until the AI finishes.

**[3] Detection (`nvinfer`, primary)**
Details in section 7. In short: resize the frame to the model's input size, run the TensorRT engine, turn the raw numbers into boxes, drop weak and unwanted ones, merge duplicates, and attach the survivors to the frame as metadata. If you chose "skip frames", this step runs only every Nth batch.

**[4] Tracking (`nvtracker` or Python)**
Matches this frame's boxes with the previous frames' objects and gives each one an ID. Details in section 8. The tracker runs on every frame even when detection is skipped, predicting where objects moved.

**[5] Cascade (secondary `nvinfer`, optional)**
For the person + posture setup, each person box is cropped out of the frame, resized, and sent to the posture model. To save work it re-checks each tracked person only every 60 frames.

**[6] The probe (Python, in the worker)**
A small function that looks at the metadata of every frame as it passes:
- measures fps and lag per camera,
- runs a Python tracker if one was chosen,
- decides each person's posture label and colours the box,
- converts the boxes into plain events and hands them to the event sender (section 10).

It must be fast, because it sits in the video path. This is why the code that reads the posture model output is vectorised, and why Kafka sending can only drop messages, never wait.

**[7] Splitting (`nvstreamdemux`)**
The batch is split back into separate per-camera streams.

**[8] Drawing and encoding (one branch per camera)**
- `nvdsosd` draws the boxes and labels onto the frame **on the GPU**.
- `nvv4l2h264enc` compresses the picture with the GPU's NVENC chip (about 2.5 Mbps, a keyframe every second so a viewer can join quickly).
- `rtspclientsink` pushes the finished stream to the video server (MediaMTX) at `cam/<camera id>/processed`.

Between steps there are small **queues**. If a queue fills because something downstream is slow, it drops the oldest frame instead of growing. That keeps latency low: you see a slightly lower frame rate rather than a growing delay.

### Rough time budget of one frame

| Stage | About |
|---|---|
| Network, camera encoding, 200 ms RTSP buffer | 200 to 400 ms (outside our control) |
| Decode | a few ms |
| Waiting in the batcher | up to 40 ms |
| Detection, tracking, posture (measured "LAG") | about 55 to 65 ms |
| Encode and send to video server | tens of ms |
| WebRTC to the browser | about 100 ms |

So what you see is about half a second behind real life. Measured on this machine: 6 of 7 cameras at 24.6 fps through WebRTC, longest freeze 0.08 s.

---

## 7. How the detector decides

Detection is one `nvinfer` element driven by a generated text config. In order:

1. **Preprocess.** The 1920x1080 frame is scaled to the model's input size (800x800 for the person model, 640x640 for the others), keeping its shape with grey padding, converted to RGB, and every pixel divided by 255.
2. **Inference.** The TensorRT engine (FP16) turns the picture into a table of thousands of candidate boxes. The ONNX files in this project have an extra *decode step built in*, so the table already has one row per candidate: `x1, y1, x2, y2, score, class`.
3. **Parse.** A small C++ library (`libnvdsinfer_custom_impl_Yolo.so`, function `NvDsInferParseYolo`) reads that table and turns the rows into DeepStream boxes.
4. **Confidence filter.** Boxes below your **confidence** (`pre-cluster-threshold`) are dropped.
5. **Class filter.** Boxes of classes you did not choose (`filter-out-class-ids`) are dropped. The vehicle model is a general 80-class model; the config keeps only car, motorcycle, bus and truck, and your per-camera choice narrows that further.
6. **NMS.** Overlapping boxes of the same object are merged using your **IoU** (`nms-iou-threshold`), keeping the best one.
7. **Metadata.** The survivors are attached to the frame as object metadata.

The three models in use:

| Model | Source | Input | Classes |
|---|---|---|---|
| person_detection | YOLO26s | 800x800 | person |
| vehicle_detection | YOLO26n | 640x640 | car, motorcycle, bus, truck |
| fall_detection | YOLO11n fine-tuned | 640x640 | fallen, sitting, standing |

**Posture label (person + posture mode).** For each person box:
1. If the fall model, run on that person's crop, is confident (score at least your limit, default 0.35), use its label.
2. Otherwise estimate from the box shape: tall and narrow means *standing*, wide and low means *fallen*, anything else means *sitting*.

In practice the fall model is rarely confident on this camera view, so the shape estimate does most of the work.

---

## 8. Tracking: giving objects an ID

Detection finds objects in each frame separately. Tracking links them across frames so a person keeps the same number.

Eight trackers are available, chosen per camera:

| Tracker | Where it runs | Idea |
|---|---|---|
| **NvDCF** (default) | GPU, in the pipeline | Learns what each object looks like and follows it, even on frames where detection is skipped |
| **NvSORT** (SORT) | GPU | Motion only: predicts where each box moves, matches by overlap |
| **IOU** | GPU | Matches boxes only by overlap, the lightest |
| **NvDeepSORT** | GPU | Adds appearance, needs a Re-ID model (not installed) |
| **ByteTrack** | Python (worker) | Two-pass matching that also uses low-confidence boxes, good in crowds |
| **OC-SORT** | Python | Handles people hidden behind others and crossing paths |
| **BoT-SORT** | Python | ByteTrack style plus appearance when a Re-ID model exists |
| **StrongSORT** | Python | DeepSORT style plus appearance when a Re-ID model exists |

How matching works in simple words (every tracker follows this idea):
1. Predict where each known object should be now, from its past movement (a Kalman filter).
2. Compare predictions with this frame's boxes (overlap, and for some trackers appearance).
3. Pair them up so the total match is best. A paired object keeps its ID.
4. A box with no match becomes a **new object** with a new ID.
5. An object with no box for a while is kept "lost" for up to about 30 frames, then removed.

The Python trackers (`worker/tracking/trackers.py`) need a detection on every frame, so the "skip frames" setting is ignored for them.

---

## 9. Per-camera settings

When you add a camera you can choose these. They are stored with the camera in the database, become part of the plan and the signature, and are turned into DeepStream settings when the worker starts.

| You choose | Stored as | Becomes |
|---|---|---|
| Model(s) | camera model assignments | which engines load; `person + posture` makes the posture model a cascade |
| Classes | `detect_class_ids` | `filter-out-class-ids` in the detector config (for posture: which labels are shown) |
| Confidence | `conf_threshold` | `pre-cluster-threshold` (for posture: minimum model score) |
| IoU | `iou_threshold` | `nms-iou-threshold` |
| Tracker | `tracker_type` | which tracker element or Python tracker |
| Detection frequency | `detection_interval` (0 to 4) | `interval` of the detector: 0 = every frame, 1 = skip one frame, and so on |

Changing a setting on a running camera rebuilds only that camera's part of the pipeline; cameras in other workers are not touched.

The generated config files are written inside the worker container under `/opt/vms/generated/<worker>/`.

---

## 10. After the frame: events, storage, analytics

**What is sent.** The probe turns each frame's objects into events, but not blindly:

- a detection event is sent when a track first appears, then again every 15 frames (`DETECTION_EVENT_INTERVAL_FRAMES`) for that track, not every frame,
- each camera has a limit of 60 events per second (a *token bucket*); extra events are counted and thrown away,
- if the Kafka queue is full, events are dropped rather than waited for.

So Kafka can never slow the video. The cost is that under extreme load some events are lost; the dropped count is reported.

**Kafka topics**

| Topic | Carries | Written by | Read by |
|---|---|---|---|
| `pipeline.commands` | start / stop / add camera commands | scheduler | gpu-agent |
| `pipeline.events` | camera or worker started, failed, reconnecting | worker, agent | scheduler, Logstash |
| `worker.heartbeat` | alive signal and status every 5 s | worker, agent | scheduler |
| `pipeline.metrics` | fps, latency numbers of each worker | worker | no program in this project; available to outside tools |
| `detection.events` | objects seen (per track) | worker | analytics-stream, Logstash |
| `embedding.events` | appearance vectors of tracks | worker | MTMC |
| `analytics.events` | track started / ended, occupancy | analytics-stream | Logstash |
| `reid.events` | same person seen on two cameras | MTMC | Logstash |
| `camera.status` | camera state changes | scheduler | no program in this project; available to outside tools |
| `*.dlq` | messages that failed processing | consumers | for inspection |

**What the analytics stream does.** It reads `detection.events`, keeps one summary per camera and track, and every few seconds saves it to PostgreSQL (`detection_events`, one row per track, not per frame). It also creates `track_started`, `track_ended` (with how long the object stayed) and `occupancy` events.

**Optional analytics profile.** MTMC compares appearance vectors of people across cameras and stores them in Milvus. Logstash copies the topics into Elasticsearch so Kibana can search and chart them.

**What PostgreSQL stores**

| Group | Tables |
|---|---|
| Users and audit | users, audit_logs |
| Cameras and models | cameras, models, model_versions, camera_model_assignments |
| GPU and workers | gpu_devices, gpu_resource_metrics, capacity_profiles, workers, worker_camera_assignments, pipeline_sessions |
| Streaming | stream_sessions |
| Analytics | detection_events, analytics_events, reid_associations |
| System | system_configuration, processed_commands (makes commands safe to repeat) |

Model files live in the `models/` folder (`.onnx` plus the built `.engine`). That folder is not in git.

---

## 11. How the video reaches your browser

```
 worker --(RTSP push, tcp)--> MediaMTX  cam/<id>/processed
                                  |
                                  +--> WebRTC  (low delay, UDP/TCP port 8189)   <- used first
                                  +--> HLS     (chunks, 2 to 6 s delay)         <- fallback
                                  +--> RTSP    (port 8554)
 browser <-- live_viewer.py (http://localhost:8080) <-- Envoy (https://localhost:8443)
```

- **Who may watch.** For every read or publish, MediaMTX asks the API (`/internal/mediamtx/auth`). A viewer needs a valid login token (JWT); a worker needs the internal service token.
- **WebRTC** gives the lowest delay. The browser sends an offer to `/webrtc/cam/<id>/processed/whep`, gets an answer, and the video then flows over UDP port 8189.
- **The live viewer** (`scripts/live_viewer.py`) logs in for you, keeps the token fresh, forwards the requests, and shows every running camera in a grid. Each tile shows its fps and delay and rebuilds itself if the picture freezes.
- **HLS** is used only if WebRTC fails.

---

## 12. Processes and crash safety

```
 gpu-agent container
   agent process            talks to Kafka, starts and stops workers, reads GPU health (NVML)
     |-- worker.runner #1   one DeepStream pipeline, camera A + B      (own process)
     |-- worker.runner #2   one DeepStream pipeline, camera C          (own process)
     `-- worker.runner #3   ...
```

- **One worker = one pipeline = one operating-system process.** If a pipeline crashes, only its cameras stop, and the agent reports it and the scheduler restarts them.
- The agent talks to each worker through its standard input (add or remove a camera) and reads its status from standard output.
- **Scheduler safety.** A PostgreSQL advisory lock lets only one scheduler act at a time; a second one waits.
- **Admission control.** Before starting a camera, the GPU manager checks measured GPU use, memory, decoder, encoder, temperature and latency against limits, and counts cameras admitted but not yet running. If there is no room, the camera waits instead of overloading the GPU.
- **Reconnects.** A lost camera connection is retried automatically every 10 seconds.

---

## 13. Security

| Item | How |
|---|---|
| Login | Username and password give a signed token (JWT). Roles: admin, operator, viewer. |
| Camera passwords | Stored encrypted (Fernet key in `.env`), decrypted only inside the worker. |
| Front door | Envoy with HTTPS on port 8443. `/internal/` is never exposed. |
| Video access | MediaMTX checks every read and publish with the API. |
| Service-to-service | Services talk through Kafka and the database. The only shared video secret is `MEDIAMTX_PUBLISH_TOKEN`, which lets a worker publish video and nothing else. The gateway removes `x-internal-token` from outside requests. |
| Kafka | Encrypted (TLS) and every service has its own login (SCRAM-SHA-512) with only the permissions it needs. Only the scheduler can write `pipeline.commands`. Plaintext access exists only on a loopback port inside the Kafka container. |
| Commands to GPU nodes | Signed with `COMMAND_SIGNING_KEY` (HMAC-SHA256), tied to one node, valid for about a minute, and never run twice. The agent refuses anything unsigned, altered, expired or repeated. |
| What a model may load | A model can only load a library from `/opt/vms/lib` (built into the image) and only read files inside the model folder. Camera sources must be `rtsp://` or `rtsps://`. Checked when a model is registered, when the agent receives a command and again when the worker starts. |
| Secrets | In `.env`, which git ignores. |

---

## 14. Start-up and the helper scripts

Order when everything starts: PostgreSQL and Kafka, then `migrate` (tables), then API, scheduler, analytics-stream, MediaMTX, Envoy, and the GPU agent. The scheduler then starts any cameras whose desired state is RUNNING.

| File | Use |
|---|---|
| `start_services.bat` | Start all Docker services (optionally analytics and monitoring) |
| `stop_services.bat` | Stop them, keeping the containers |
| `run_all.bat` | Start services, register models, start all cameras, open the live viewer, show a status table |
| `add_camera.bat` | Add cameras interactively (IP, login, channels, model, classes, tracker, detection frequency, confidence and IoU) |
| `delete_camera.bat` | Delete cameras by number or ID |
| `scripts/run_all.py --status` | Print the status table only |

The status table shows each camera, its model(s), detection frequency, tracker, frames per second and lag.

---

## 15. Where things are in the folders

```
app/                      the control side
  main.py                   starts the API
  api/v1/                   endpoints: auth, cameras, models, workers_gpu, analytics, internal
  services/scheduler.py     the manager loop (desired vs actual)
  services/model_router.py  builds the "plan" and signature for a camera
  services/gpu.py           GPU telemetry and admission control
  services/mediamtx.py      talks to the video server
  bus/                      Kafka helpers and message formats
  db/                       database tables and sessions
  core/                     settings, security, tracker names, class selection
worker/                   the video side
  agent.py                  the GPU agent (starts workers)
  runner.py                 one worker process
  pipeline/deepstream_pipeline.py   the pipeline and the probe
  pipeline/config_gen.py    writes the DeepStream config files
  pipeline/events.py        decides which events to send
  tracking/trackers.py      the Python trackers
analytics/                the analytics side (stream, MTMC, Milvus)
migrations/               database upgrades (0001 to 0004)
config/models/models.yaml the model registry (what each model is and how to run it)
deploy/                   Envoy, MediaMTX, Prometheus, Grafana, Helm charts
docker/                   the two images (control plane, DeepStream worker)
scripts/                  helper tools (add/delete camera, run_all, live viewer, benchmarks)
models/                   model files and built engines (not in git)
docs/                     this guide, API reference, deployment, benchmark
```

---

## 16. Limits and honest notes

- **One GPU is shared by all cameras.** With 7 cameras running, the RTX 5050 was around 90% busy. Adding cameras lowers the frame rates first. The heaviest setup is person + posture.
- **Different settings mean separate workers.** Each worker is a separate process and the GPU time-slices between them. Cameras with identical settings share a worker and are cheaper.
- **The fall model is weak on this view.** The posture label usually comes from the box shape, so it is an estimate, not true fall recognition. For real fall detection the model should be retrained on footage from your own cameras.
- **Python trackers need every frame.** The detection-frequency setting is ignored for them. They have no appearance matching without a Re-ID model, and none is installed.
- **Events can be dropped under extreme load** by design, so video never stalls. The count of dropped events is reported.
- **Windows and Docker Desktop.** The system runs there, but a Linux server with an NVIDIA GPU is the intended home; WebRTC and Docker networking have extra layers on Windows.
- **NVR logins.** Many recorders lock an account after a few wrong passwords. The add-camera tool stops at the first rejected login for this reason.
- **Unit tests** were in the `tests/` folder; check the git history if you need them.
