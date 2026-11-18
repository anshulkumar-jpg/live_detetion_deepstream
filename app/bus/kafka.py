"""Kafka producer/consumer with retries, dead-letter queue, idempotence and
non-blocking backpressure. Also an in-memory bus with the same interface for tests.

Design rules:
* Producer never blocks the caller (video threads). When the local queue is full
  the message is dropped and counted (``dropped`` metric) instead of stalling.
* Control messages (commands) use ``publish_reliable`` which waits for the ack.
* Consumers commit offsets only after the handler succeeds; after
  ``max_retries`` failures the message is copied to ``<topic>.dlq`` with error
  headers and then committed so one poison message can't stall a partition.
* Keys: ``camera_id`` for per-camera ordering, ``worker_id``/``node_id`` for commands.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from app.bus.schemas import Topics
from app.core.config import Settings, get_settings

log = logging.getLogger(__name__)

Handler = Callable[[str, str | None, dict[str, Any]], Awaitable[None]]


def _common_conf(s: Settings) -> dict[str, Any]:
    conf: dict[str, Any] = {"bootstrap.servers": s.kafka_bootstrap_servers, "security.protocol": s.kafka_security_protocol}
    if s.kafka_sasl_mechanism:
        conf.update({
            "sasl.mechanism": s.kafka_sasl_mechanism,
            "sasl.username": s.kafka_sasl_username,
            "sasl.password": s.kafka_sasl_password,
        })
    if s.kafka_ssl_ca_location:
        conf["ssl.ca.location"] = s.kafka_ssl_ca_location
    return conf


def _encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), default=str).encode()


class BusStats:
    def __init__(self) -> None:
        self.produced: dict[str, int] = defaultdict(int)
        self.dropped: dict[str, int] = defaultdict(int)
        self.failed: dict[str, int] = defaultdict(int)
        self.dlq: dict[str, int] = defaultdict(int)
        self.consumed: dict[str, int] = defaultdict(int)


class KafkaBus:
    def __init__(self, settings: Settings | None = None, client_id: str = "vms"):
        from confluent_kafka import Producer

        self.s = settings or get_settings()
        self.stats = BusStats()
        conf = _common_conf(self.s)
        conf.update({
            "client.id": client_id,
            "enable.idempotence": True,  # implies acks=all, no duplicates on producer retry
            "linger.ms": 5,
            "compression.type": "lz4",
            "queue.buffering.max.messages": self.s.kafka_producer_queue_max,
            "message.timeout.ms": 30000,
        })
        self._producer = Producer(conf)
        self._stop = threading.Event()
        self._poller = threading.Thread(target=self._poll_loop, name="kafka-poll", daemon=True)
        self._poller.start()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            self._producer.poll(0.1)

    def _on_delivery(self, topic: str):
        def cb(err, _msg):
            if err is not None:
                self.stats.failed[topic] += 1
                log.warning("kafka delivery failed topic=%s err=%s", topic, err)
            else:
                self.stats.produced[topic] += 1
        return cb

    def publish(self, topic: str, payload: dict[str, Any], key: str | None = None) -> bool:
        """Fire-and-forget, never blocks. Returns False if dropped due to backpressure."""
        from confluent_kafka import KafkaException

        full = self.s.topic(topic)
        try:
            self._producer.produce(full, _encode(payload), key=key.encode() if key else None,
                                   on_delivery=self._on_delivery(topic))
            return True
        except BufferError:
            self.stats.dropped[topic] += 1
            return False
        except KafkaException as e:
            self.stats.failed[topic] += 1
            log.error("kafka produce error topic=%s err=%s", topic, e)
            return False

    def publish_reliable(self, topic: str, payload: dict[str, Any], key: str | None = None, timeout: float = 10.0) -> None:
        """Wait for broker ack (used for commands). Raises on failure."""
        full = self.s.topic(topic)
        result: dict[str, Any] = {}
        done = threading.Event()

        def cb(err, msg):
            result["err"] = err
            done.set()

        for attempt in range(self.s.kafka_max_retries + 1):
            done.clear()
            self._producer.produce(full, _encode(payload), key=key.encode() if key else None, on_delivery=cb)
            self._producer.flush(timeout)
            if done.wait(timeout) and result.get("err") is None:
                self.stats.produced[topic] += 1
                return
            time.sleep(min(0.2 * 2**attempt, 2.0))
        self.stats.failed[topic] += 1
        raise RuntimeError(f"failed to publish to {topic}: {result.get('err')}")

    def flush(self, timeout: float = 5.0) -> int:
        return self._producer.flush(timeout)

    def close(self) -> None:
        self.flush()
        self._stop.set()


class KafkaConsumerLoop:
    """Async wrapper around a confluent Consumer with retry + DLQ semantics."""

    def __init__(self, topics: list[str], group_id: str, handler: Handler, bus: KafkaBus,
                 settings: Settings | None = None, auto_offset_reset: str = "latest"):
        from confluent_kafka import Consumer

        self.s = settings or get_settings()
        self.topics = topics
        self.handler = handler
        self.bus = bus
        conf = _common_conf(self.s)
        conf.update({
            "group.id": group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": auto_offset_reset,
            "max.poll.interval.ms": 300000,
            "session.timeout.ms": 30000,
        })
        self._consumer = Consumer(conf)
        self._consumer.subscribe([self.s.topic(t) for t in topics])
        self._running = True
        self.lag: dict[str, int] = {}

    def _strip(self, topic: str) -> str:
        p = self.s.kafka_topic_prefix
        return topic[len(p):] if p and topic.startswith(p) else topic

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            msg = await loop.run_in_executor(None, self._consumer.poll, 1.0)
            if msg is None:
                continue
            if msg.error():
                log.warning("kafka consumer error: %s", msg.error())
                continue
            topic = self._strip(msg.topic())
            key = msg.key().decode() if msg.key() else None
            try:
                payload = json.loads(msg.value())
            except Exception as e:  # unparseable -> DLQ immediately
                self._to_dlq(topic, msg, f"decode error: {e}")
                self._commit(msg)
                continue
            for attempt in range(self.s.kafka_max_retries + 1):
                try:
                    await self.handler(topic, key, payload)
                    self.bus.stats.consumed[topic] += 1
                    break
                except Exception as e:
                    if attempt >= self.s.kafka_max_retries:
                        log.exception("handler failed permanently topic=%s -> DLQ", topic)
                        self._to_dlq(topic, msg, repr(e))
                        break
                    await asyncio.sleep(min(0.25 * 2**attempt, 5.0))
            self._commit(msg)

    def _commit(self, msg) -> None:
        try:
            self._consumer.commit(message=msg, asynchronous=True)
        except Exception as e:
            log.warning("offset commit failed: %s", e)

    def _to_dlq(self, topic: str, msg, error: str) -> None:
        self.bus.stats.dlq[topic] += 1
        self.bus._producer.produce(
            self.s.topic(Topics.dlq(topic)), msg.value(), key=msg.key(),
            headers=[("x-error", error[:1000].encode()), ("x-source-topic", topic.encode()),
                     ("x-source-offset", str(msg.offset()).encode())],
        )

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:
        self._running = False
        self._consumer.close()


def ensure_topics(settings: Settings | None = None, topics: list[str] | None = None) -> list[str]:
    """Create topics (and their DLQs) if missing. Returns topics created."""
    from confluent_kafka.admin import AdminClient, NewTopic

    s = settings or get_settings()
    admin = AdminClient(_common_conf(s))
    wanted = topics or Topics.ALL
    names = [s.topic(t) for t in wanted] + [s.topic(Topics.dlq(t)) for t in wanted]
    existing = set(admin.list_topics(timeout=10).topics.keys())
    new = [NewTopic(n, num_partitions=s.kafka_default_partitions, replication_factor=s.kafka_replication_factor)
           for n in names if n not in existing]
    created = []
    if new:
        for name, fut in admin.create_topics(new).items():
            try:
                fut.result()
                created.append(name)
            except Exception as e:
                if "TOPIC_ALREADY_EXISTS" not in str(e):
                    raise
    return created


# ---------------------------------------------------------------- in-memory
class InMemoryBus:
    """Same publish interface as KafkaBus; used in unit tests and when KAFKA_ENABLED=false."""

    def __init__(self, retain: bool = True) -> None:
        self.retain = retain  # False: count only (long-running benchmark/validation runs)
        self.messages: list[tuple[str, str | None, dict[str, Any]]] = []
        self.stats = BusStats()
        self._subs: dict[str, list[Callable[[str, str | None, dict], Any]]] = defaultdict(list)

    def subscribe(self, topic: str, fn: Callable[[str, str | None, dict], Any]) -> None:
        self._subs[topic].append(fn)

    def publish(self, topic: str, payload: dict[str, Any], key: str | None = None) -> bool:
        if self.retain:
            self.messages.append((topic, key, payload))
        self.stats.produced[topic] += 1
        for fn in self._subs.get(topic, []):
            fn(topic, key, payload)
        return True

    def publish_reliable(self, topic: str, payload: dict[str, Any], key: str | None = None, timeout: float = 10.0) -> None:
        self.publish(topic, payload, key)

    def flush(self, timeout: float = 5.0) -> int:
        return 0

    def close(self) -> None:
        pass

    def on(self, topic: str) -> list[dict[str, Any]]:
        return [p for t, _, p in self.messages if t == topic]
