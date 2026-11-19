#!/bin/sh
# One-shot Elasticsearch bootstrap (run by the es-setup service). Safe to run again.
set -eu
ES=http://elasticsearch:9200
AUTH="elastic:${ELASTIC_PASSWORD}"
call() { curl -fsS -u "$AUTH" -H 'Content-Type: application/json' -X "$1" "$ES$2" -d "$3" >/dev/null; }

# Kibana talks to Elasticsearch as the built-in kibana_system user
call POST /_security/user/kibana_system/_password "{\"password\":\"${KIBANA_SYSTEM_PASSWORD}\"}"

# Logstash may only create and write the vms-* indices (it cannot read, delete or manage anything else)
call PUT /_security/role/vms_logstash_writer '{"cluster":["monitor"],"indices":[{"names":["vms-*"],"privileges":["create_index","create","index","write"]}]}'
call PUT /_security/user/logstash_writer "{\"password\":\"${LOGSTASH_ES_PASSWORD}\",\"roles\":[\"vms_logstash_writer\"]}"
echo "es-setup: done"
