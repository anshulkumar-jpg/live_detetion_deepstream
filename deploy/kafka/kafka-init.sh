#!/bin/bash
# Runs inside the Kafka container, in the background, every time it starts.
# It waits until the broker answers on the private loopback listener (127.0.0.1:9094, not reachable from other
# containers), then creates the logins and the permissions, and finally writes /tmp/kafka-ready, which the
# container's health check waits for. Other services therefore never start before the security is in place.
#
# Principle: every service has its own login and may only do what it needs.
#   scheduler  : the ONLY one that can write pipeline.commands (what the GPU nodes will run)
#   agent      : reads pipeline.commands; writes events, heartbeats, metrics, detections  (also used by the workers)
#   analytics  : reads detections, writes analytics events
#   mtmc       : reads embeddings, writes re-identification events
#   api        : writes camera status only
#   logstash   : reads four topics;   exporter : read-only metadata for Prometheus
set -u
rm -f /tmp/kafka-ready
K=/opt/kafka/bin
BS=127.0.0.1:9094

for i in $(seq 1 180); do
  $K/kafka-broker-api-versions.sh --bootstrap-server $BS >/dev/null 2>&1 && break
  sleep 2
done

user() {  # name password
  $K/kafka-configs.sh --bootstrap-server $BS --alter --entity-type users --entity-name "$1" \
    --add-config "SCRAM-SHA-512=[iterations=8192,password=$2]" >/dev/null 2>&1 || echo "kafka-init: could not create user $1"
}
acl() {   # user resource-type resource operations... (resource-type: topic|group|cluster)
  local u=$1 t=$2 r=$3 pat=literal; shift 3
  # 'name-*' means "everything starting with name-"; a lone '*' means "all" (a literal wildcard)
  if [ "${#r}" -gt 1 ]; then case "$r" in *\*) pat=prefixed; r=${r%\*};; esac; fi
  local ops=(); for o in "$@"; do ops+=(--operation "$o"); done
  if [ "$t" = cluster ]; then
    $K/kafka-acls.sh --bootstrap-server $BS --add --allow-principal "User:$u" --cluster "${ops[@]}" >/dev/null 2>&1
  else
    $K/kafka-acls.sh --bootstrap-server $BS --add --allow-principal "User:$u" --$t "$r" --resource-pattern-type $pat "${ops[@]}" >/dev/null 2>&1
  fi || echo "kafka-init: ACL failed for $u on $t $r"
}

user vms-scheduler "$KAFKA_PASS_SCHEDULER"
user vms-agent     "$KAFKA_PASS_AGENT"
user vms-analytics "$KAFKA_PASS_ANALYTICS"
user vms-mtmc      "$KAFKA_PASS_MTMC"
user vms-api       "$KAFKA_PASS_API"
user vms-logstash  "$KAFKA_PASS_LOGSTASH"
user vms-exporter  "$KAFKA_PASS_EXPORTER"

# topics (and their dead-letter topics) so that no service needs permission to create topics
TOPICS="camera.commands camera.status pipeline.commands pipeline.events detection.events worker.heartbeat pipeline.metrics analytics.events reid.events embedding.events"
for t in $TOPICS; do
  for n in "$t" "$t.dlq"; do
    $K/kafka-topics.sh --bootstrap-server $BS --create --if-not-exists --topic "$n" \
      --partitions "${KAFKA_NUM_PARTITIONS:-12}" --replication-factor 1 >/dev/null 2>&1
  done
done

# every service may see which topics exist (services check this at start); nothing more
for u in vms-scheduler vms-agent vms-analytics vms-mtmc vms-api vms-logstash vms-exporter; do
  acl $u topic '*' Describe
done

# scheduler
acl vms-scheduler topic pipeline.commands Write
acl vms-scheduler topic camera.status Write
acl vms-scheduler topic worker.heartbeat Read
acl vms-scheduler topic pipeline.events Read
acl vms-scheduler topic worker.heartbeat.dlq Write
acl vms-scheduler topic pipeline.events.dlq Write
acl vms-scheduler group scheduler Read

# GPU agent and its worker processes
acl vms-agent topic pipeline.commands Read
acl vms-agent topic pipeline.commands.dlq Write
acl vms-agent topic pipeline.events Write
acl vms-agent topic worker.heartbeat Write
acl vms-agent topic pipeline.metrics Write
acl vms-agent topic detection.events Write
acl vms-agent topic embedding.events Write
acl vms-agent group 'agent-*' Read

# analytics
acl vms-analytics topic detection.events Read
acl vms-analytics topic detection.events.dlq Write
acl vms-analytics topic analytics.events Write
acl vms-analytics group analytics-stream Read
acl vms-mtmc topic embedding.events Read
acl vms-mtmc topic embedding.events.dlq Write
acl vms-mtmc topic reid.events Write
acl vms-mtmc group mtmc Read

# API
acl vms-api topic camera.status Write

# optional services
for t in detection.events analytics.events reid.events pipeline.events; do acl vms-logstash topic $t Read; done
acl vms-logstash group logstash-es Read
acl vms-exporter group '*' Describe
acl vms-exporter cluster kafka-cluster Describe

touch /tmp/kafka-ready
echo "kafka-init: logins and permissions are in place"
