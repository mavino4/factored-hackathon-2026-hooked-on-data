# Ollama → Docker proxy (local development)

Ollama listens only on `127.0.0.1:11434`, which containers can't reach. These user-level
systemd units listen on the Docker bridge address `172.17.0.1:11434` (what
`host.docker.internal` resolves to in `docker-compose.yml`) and forward to Ollama.
There's no sudo, Ollama's own config is unchanged, and nothing is exposed on your LAN.

Install:

```bash
cp deploy/ollama-docker-proxy/ollama-docker-proxy.{socket,service} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ollama-docker-proxy.socket
```

Remove:

```bash
systemctl --user disable --now ollama-docker-proxy.socket ollama-docker-proxy.service
rm ~/.config/systemd/user/ollama-docker-proxy.{socket,service}
systemctl --user daemon-reload
```
