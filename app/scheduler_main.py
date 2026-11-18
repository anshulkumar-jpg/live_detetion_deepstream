"""Scheduler service entrypoint (Worker Scheduler + GPU admission + recovery).

Run: python -m app.scheduler_main
Only one instance is active at a time (PostgreSQL advisory lock); standbys wait.
"""
from __future__ import annotations

import asyncio
import os
import logging
import signal

from prometheus_client import start_http_server

from app.bus.kafka import KafkaBus, KafkaConsumerLoop, ensure_topics
from app.bus.schemas import Topics
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.metrics import SCHED
from app.db.session import get_engine, get_sessionmaker, init_engine, wait_for_db
from app.services.mediamtx import StreamManager
from app.services.scheduler import Scheduler

log = logging.getLogger("scheduler")


async def main() -> None:
    s = get_settings()
    configure_logging("scheduler", s.log_level, s.log_json)
    if s.environment == "production":
        s.validate_for_production()
    start_http_server(int(os.environ.get("METRICS_PORT", "9101")))
    init_engine(s)
    await wait_for_db()
    ensure_topics(s)
    bus = KafkaBus(s, client_id="scheduler")
    sched = Scheduler(get_sessionmaker(), bus, s, source="scheduler", stream_manager=StreamManager(s))

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    # leader election: hold a dedicated connection with the advisory lock
    lock_conn = await get_engine().connect()
    while not stop.is_set():
        if await sched.try_acquire_leadership(lock_conn):
            break
        SCHED.is_leader.set(0)
        log.info("standby: another scheduler holds the leader lock")
        try:
            await asyncio.wait_for(stop.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
    if stop.is_set():
        return
    SCHED.is_leader.set(1)
    log.info("acquired scheduler leadership")

    consumer = KafkaConsumerLoop([Topics.WORKER_HEARTBEAT, Topics.PIPELINE_EVENTS], "scheduler",
                                 sched.handle_message, bus, s, auto_offset_reset="latest")
    consumer_task = asyncio.create_task(consumer.run())

    async def reconcile_loop():
        last_stats: dict = {}
        while not stop.is_set():
            try:
                with SCHED.reconcile_seconds.time():
                    stats = await sched.reconcile_once()
                if stats and stats != last_stats:
                    log.info("reconcile: %s", stats)
                last_stats = stats
            except Exception:
                log.exception("reconcile pass failed (will retry)")
            try:
                await asyncio.wait_for(stop.wait(), timeout=s.scheduler_interval_s)
            except asyncio.TimeoutError:
                pass

    await reconcile_loop()
    consumer.stop()
    await asyncio.sleep(1.5)
    consumer_task.cancel()
    consumer.close()
    bus.close()
    await lock_conn.close()


if __name__ == "__main__":
    asyncio.run(main())
