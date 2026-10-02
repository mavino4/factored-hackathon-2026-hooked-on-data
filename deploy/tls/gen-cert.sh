#!/usr/bin/env sh
# TLS for the LAN demo (docker-compose.lan.yml): a private certificate authority for this
# machine, and a server certificate signed by it for the names and addresses the machine
# answers to. Run again after the machine's IP changes; the CA is kept, so devices that
# already trust it need nothing new.
#
#   ./deploy/tls/gen-cert.sh [extra-name-or-ip ...]        # make tls
#
# Browsers warn until the device trusts ca.crt (install it once per device; the LAN demo
# serves it at http://<this machine>:8000/ca.crt). ca.key can sign a certificate for any
# site, for whoever trusts ca.crt: it stays on this machine (mode 600, not in git).
set -eu
dir=$(dirname "$0")
cd "$dir"
umask 077

if [ ! -f ca.key ] || [ ! -f ca.crt ]; then
    openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes \
        -keyout ca.key -out ca.crt -days 3650 -subj "/CN=BankBot LAN demo CA ($(hostname))" \
        -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
        -addext "keyUsage=critical,keyCertSign,cRLSign" 2>/dev/null
    echo "created the certificate authority: $dir/ca.crt"
fi

# Names and addresses for the certificate: this machine's, without Docker's own networks.
sans="DNS:localhost,DNS:$(hostname),DNS:$(hostname).local,IP:127.0.0.1"
for ip in $(hostname -I); do
    case "$ip" in 172.1[6-9].*|172.2[0-9].*|172.3[01].*) continue ;; esac
    sans="$sans,IP:$ip"
done
for extra in "$@"; do
    case "$extra" in
        *[!0-9.]*) case "$extra" in *:*) sans="$sans,IP:$extra" ;; *) sans="$sans,DNS:$extra" ;; esac ;;
        *) sans="$sans,IP:$extra" ;;
    esac
done

# 397 days: browsers and iOS reject server certificates valid for longer.
openssl req -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes \
    -keyout server.key -out server.csr -subj "/CN=$(hostname)" 2>/dev/null
printf 'subjectAltName=%s\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=serverAuth\n' \
    "$sans" > server.ext
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out server.crt -days 397 -sha256 -extfile server.ext 2>/dev/null
rm -f server.csr server.ext
chmod 600 ca.key server.key
chmod 644 ca.crt server.crt
echo "server certificate for: $sans"
echo "valid until: $(openssl x509 -in server.crt -noout -enddate | cut -d= -f2)"
