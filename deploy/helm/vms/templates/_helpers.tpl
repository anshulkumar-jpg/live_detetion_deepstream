{{- define "vms.labels" -}}
app.kubernetes.io/part-of: vms
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{- define "vms.env" -}}
envFrom:
  - configMapRef: {name: {{ .Release.Name }}-config}
  - secretRef: {name: {{ .Values.existingSecret }}}
{{- end }}

{{- define "vms.securityContext" -}}
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities: {drop: [ALL]}
{{- end }}
