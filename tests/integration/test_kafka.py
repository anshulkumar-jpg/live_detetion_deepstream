"""Real-broker tests (TEST_KAFKA_BOOTSTRAP, default localhost:9092)."""
import asyncio
import json
import os
import uuid

import pytest

from app.bus.kafka import KafkaBus, KafkaConsumerLoop, ensure_topics
from app.bus.schemas import Topics
from app.core.config import Settings

pytestmark = pytest.mark.kafka
BOOT = os.environ.get("TEST_KAFKA_BOOTSTRAP", "localhost:9092")


def kafka_up() -> bool:
    try:
        from confluent_kafka.admin import AdminClient
        AdminClient({"bootstrap.servers": BOOT, "socket.timeout.ms": 3000}).list_topics(timeout=3)
        return True
    except Exception:
        return False


@pytest.fixture
def ksettings(settings) -> Settings:
    if not kafka_up():
        pytest.skip("Kafka not available")
    return settings.model_copy(update={"kafka_enabled": True, "kafka_bootstrap_servers": BOOT,
                                       "kafka_topic_prefix": f"t{uuid.uuid4().hex[:6]}.", "kafka_default_partitions": 3,
                                       "kafka_max_retries": 2})


async def consume_until(loop_obj: KafkaConsumerLoop, cond, timeout=30):
    task = asyncio.create_task(loop_obj.run())
    try:
        for _ in range(int(timeout * 10)):
            if cond():
                return True
            await asyncio.sleep(0.1)
        return False
    finally:
        loop_obj.stop()
        await asyncio.sleep(1.2)
        task.cancel()
        loop_obj.close()


async def test_topics_publish_consume_ordering_and_dlq(ksettings):
    created = ensure_topics(ksettings)
    assert len(created) == 2 * len(Topics.ALL)
    assert ensure_topics(ksettings) == []  # idempotent
    bus = KafkaBus(ksettings, client_id="test")
    for i in range(20):
        assert bus.publish(Topics.DETECTION_EVENTS, {"camera_id": "CAM_1", "seq": i}, key="CAM_1")
    bus.publish(Topics.DETECTION_EVENTS, {"camera_id": "CAM_2", "seq": 0, "poison": True}, key="CAM_2")
    bus.publish_reliable(Topics.PIPELINE_COMMANDS, {"event_type": "PING"}, key="node")
    bus.flush()
    seen, attempts = [], {"poison": 0}

    async def handler(topic, key, msg):
        if msg.get("poison"):
            attempts["poison"] += 1
            raise RuntimeError("cannot process")
        seen.append((key, msg["seq"]))

    consumer = KafkaConsumerLoop([Topics.DETECTION_EVENTS], f"g-{uuid.uuid4().hex[:6]}", handler, bus, ksettings,
                                 auto_offset_reset="earliest")
    ok = await consume_until(consumer, lambda: len(seen) == 20 and bus.stats.dlq.get(Topics.DETECTION_EVENTS) == 1)
    assert ok, (seen, dict(bus.stats.dlq))
    assert [s for k, s in seen if k == "CAM_1"] == list(range(20))  # per-camera ordering via key
    assert attempts["poison"] == ksettings.kafka_max_retries + 1
    bus.flush()
    # poison message landed in the DLQ with error headers
    from confluent_kafka import Consumer
    c = Consumer({"bootstrap.servers": BOOT, "group.id": f"dlq-{uuid.uuid4().hex[:6]}", "auto.offset.reset": "earliest"})
    c.subscribe([ksettings.topic(Topics.dlq(Topics.DETECTION_EVENTS))])
    msg = None
    for _ in range(50):
        msg = c.poll(0.5)
        if msg is not None and not msg.error():
            break
    c.close()
    assert msg is not None and json.loads(msg.value())["poison"] is True
    assert dict(msg.headers())["x-error"].startswith(b"RuntimeError")
    bus.close()


async def test_scheduler_to_agent_over_kafka(ksettings, sm):
    """Control plane end-to-end over a real broker: scheduler -> pipeline.commands -> node agent ->
    worker process (protocol stand-in) and agent -> pipeline.events -> scheduler."""
    from app.services.scheduler import Scheduler
    from tests.conftest import gpu_heartbeat
    from tests.integration.helpers import add_cameras, seed_models
    from tests.unit.test_agent_mtmc import spawn
    from worker.agent import NodeAgent

    ensure_topics(ksettings)
    bus = KafkaBus(ksettings, client_id="e2e")
    async with sm() as db:
        await seed_models(db)
        await add_cameras(db, ["CAM_1", "CAM_2", "CAM_3"])
    sched = Scheduler(sm, bus, ksettings)
    agent = NodeAgent(ksettings, bus, spawn=spawn)
    agent_loop = KafkaConsumerLoop([Topics.PIPELINE_COMMANDS], f"agent-{uuid.uuid4().hex[:6]}", agent.handle_command,
                                   bus, ksettings, auto_offset_reset="earliest")
    await sched.handle_message(Topics.WORKER_HEARTBEAT, None, gpu_heartbeat())
    stats = await sched.reconcile_once()
    assert stats["workers_created"] == 2
    # CAM_2 reaches its worker via ADD_SOURCE -> agent -> worker stdin -> worker status line
    ok = await consume_until(agent_loop, lambda: len(agent.workers) == 2 and any(
        "CAM_2" in (w.last_metrics.get("sources") or {}) for w in agent.workers.values()))
    assert ok, {k: w.last_metrics for k, w in agent.workers.items()}
    agent.shutdown()
    bus.close()
