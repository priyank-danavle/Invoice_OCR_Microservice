#!/usr/bin/env bash
# =============================================================================
# OCR Microservice — Server Setup Script
# =============================================================================
# Usage:
#   chmod +x deploy/setup_server.sh
#   sudo ./deploy/setup_server.sh
#
# What this script does:
#   1. Detects OS and package manager
#   2. Checks what is already installed (never re-installs blindly)
#   3. Installs only what is missing: git, docker, docker compose, nginx, certbot
#   4. Checks if port 9001 is free
#   5. Configures MySQL to allow connections from Docker bridge (172.17.0.1)
#   6. Pulls the repo and starts the service
#   7. Sets up Nginx + Let's Encrypt HTTPS
# =============================================================================

set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${BOLD}${CYAN}══ $* ══${NC}"; }

# ── Config (edit these before running) ────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/OCR_microservice.git}"
DEPLOY_DIR="${DEPLOY_DIR:-/opt/ocr-microservice}"
BRANCH="${BRANCH:-master}"
APP_PORT=9001           # external port on this server
CONTAINER_PORT=8001     # internal FastAPI port inside Docker
DOMAIN="${DOMAIN:-}"    # e.g. ocr.yourcompany.com — leave empty to skip HTTPS

# ── Must run as root ───────────────────────────────────────────────────────────
[[ $EUID -ne 0 ]] && error "Run this script as root: sudo $0"

# =============================================================================
section "Step 1: Detect Operating System"
# =============================================================================
if [[ -f /etc/os-release ]]; then
    source /etc/os-release
    OS_ID="${ID}"
    OS_NAME="${PRETTY_NAME}"
else
    error "Cannot detect OS. /etc/os-release not found."
fi

info "Detected: $OS_NAME"

case "$OS_ID" in
    ubuntu|debian|linuxmint)
        PKG_MGR="apt-get"
        PKG_UPDATE="apt-get update -qq"
        PKG_INSTALL="apt-get install -y -qq"
        ;;
    centos|rhel|rocky|almalinux|fedora)
        PKG_MGR="dnf"
        PKG_UPDATE="dnf check-update -q || true"
        PKG_INSTALL="dnf install -y -q"
        ;;
    *)
        error "Unsupported OS: $OS_ID. Supported: Ubuntu, Debian, CentOS, RHEL, Rocky, AlmaLinux, Fedora"
        ;;
esac

ok "Package manager: $PKG_MGR"

# =============================================================================
section "Step 2: Check Existing Programs"
# =============================================================================

check_cmd() {
    local cmd="$1"
    local version_flag="${2:---version}"
    if command -v "$cmd" &>/dev/null; then
        local ver
        ver=$("$cmd" $version_flag 2>&1 | head -1) || true
        ok "$cmd is installed — $ver"
        return 0
    else
        warn "$cmd is NOT installed"
        return 1
    fi
}

NEED_GIT=false
NEED_DOCKER=false
NEED_COMPOSE=false
NEED_NGINX=false
NEED_CERTBOT=false
NEED_CURL=false

check_cmd git       || NEED_GIT=true
check_cmd curl      || NEED_CURL=true

# Docker check
if command -v docker &>/dev/null; then
    DOCKER_VER=$(docker --version 2>&1)
    ok "docker is installed — $DOCKER_VER"
    # Check docker daemon is running
    if docker info &>/dev/null; then
        ok "Docker daemon is running"
    else
        warn "Docker daemon is NOT running — will enable it"
    fi
else
    warn "docker is NOT installed"
    NEED_DOCKER=true
fi

# Docker Compose v2 check (plugin: docker compose)
if docker compose version &>/dev/null 2>&1; then
    COMPOSE_VER=$(docker compose version 2>&1)
    ok "docker compose (v2 plugin) — $COMPOSE_VER"
elif command -v docker-compose &>/dev/null; then
    COMPOSE_VER=$(docker-compose --version 2>&1)
    warn "Found legacy docker-compose v1 — $COMPOSE_VER (will install v2 plugin)"
    NEED_COMPOSE=true
else
    warn "docker compose is NOT installed"
    NEED_COMPOSE=true
fi

# Nginx check
if command -v nginx &>/dev/null; then
    NGINX_VER=$(nginx -v 2>&1)
    ok "nginx is installed — $NGINX_VER"
else
    warn "nginx is NOT installed"
    NEED_NGINX=true
fi

# Certbot check
if command -v certbot &>/dev/null; then
    CERT_VER=$(certbot --version 2>&1)
    ok "certbot is installed — $CERT_VER"
else
    warn "certbot is NOT installed"
    NEED_CERTBOT=true
fi

echo ""
info "Summary — programs to install:"
$NEED_GIT     && echo "  • git" || true
$NEED_CURL    && echo "  • curl" || true
$NEED_DOCKER  && echo "  • docker" || true
$NEED_COMPOSE && echo "  • docker compose plugin" || true
$NEED_NGINX   && echo "  • nginx" || true
$NEED_CERTBOT && echo "  • certbot" || true

if ! $NEED_GIT && ! $NEED_CURL && ! $NEED_DOCKER && ! $NEED_COMPOSE && ! $NEED_NGINX && ! $NEED_CERTBOT; then
    ok "All required programs are already installed — no installations needed!"
fi

# =============================================================================
section "Step 3: Install Missing Programs"
# =============================================================================

if $NEED_GIT || $NEED_CURL; then
    info "Updating package index..."
    $PKG_UPDATE
    PKGS_TO_INSTALL=""
    $NEED_GIT  && PKGS_TO_INSTALL="$PKGS_TO_INSTALL git"
    $NEED_CURL && PKGS_TO_INSTALL="$PKGS_TO_INSTALL curl"
    info "Installing: $PKGS_TO_INSTALL"
    $PKG_INSTALL $PKGS_TO_INSTALL
    ok "Installed: $PKGS_TO_INSTALL"
fi

# ── Install Docker ─────────────────────────────────────────────────────────────
if $NEED_DOCKER; then
    info "Installing Docker via official convenience script..."
    $PKG_UPDATE
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    ok "Docker installed and started"
elif ! docker info &>/dev/null; then
    info "Starting Docker daemon..."
    systemctl enable docker
    systemctl start docker
    ok "Docker daemon started"
fi

# ── Install Docker Compose v2 plugin ──────────────────────────────────────────
if $NEED_COMPOSE; then
    info "Installing Docker Compose v2 plugin..."
    COMPOSE_LATEST=$(curl -fsSL https://api.github.com/repos/docker/compose/releases/latest \
        | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
    ARCH=$(uname -m)
    [[ "$ARCH" == "x86_64" ]] && ARCH="x86_64"
    [[ "$ARCH" == "aarch64" ]] && ARCH="aarch64"
    mkdir -p /usr/local/lib/docker/cli-plugins
    curl -fsSL "https://github.com/docker/compose/releases/download/${COMPOSE_LATEST}/docker-compose-linux-${ARCH}" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
    ok "Docker Compose plugin installed: $(docker compose version)"
fi

# ── Install Nginx ──────────────────────────────────────────────────────────────
if $NEED_NGINX; then
    info "Installing Nginx..."
    $PKG_UPDATE
    $PKG_INSTALL nginx
    systemctl enable nginx
    systemctl start nginx
    ok "Nginx installed and started"
fi

# ── Install Certbot ────────────────────────────────────────────────────────────
if $NEED_CERTBOT; then
    info "Installing Certbot..."
    if [[ "$OS_ID" == "ubuntu" || "$OS_ID" == "debian" ]]; then
        $PKG_INSTALL snapd || true
        snap install --classic certbot 2>/dev/null || $PKG_INSTALL certbot python3-certbot-nginx
    else
        $PKG_INSTALL epel-release || true
        $PKG_INSTALL certbot python3-certbot-nginx
    fi
    ok "Certbot installed"
fi

# =============================================================================
section "Step 4: Check Port 9001 Availability"
# =============================================================================

if ss -tlnp | grep -q ":${APP_PORT}"; then
    USING=$(ss -tlnp | grep ":${APP_PORT}" | awk '{print $NF}')
    error "Port ${APP_PORT} is already in use by: ${USING}\nChoose a different port or stop the conflicting service."
else
    ok "Port ${APP_PORT} is free and available"
fi

# =============================================================================
section "Step 5: MySQL — Allow Docker Bridge Connections"
# =============================================================================

DOCKER_BRIDGE_IP="172.17.0.1"
MYSQL_CONF_DIR=""

# Find MySQL config directory
for d in /etc/mysql /etc/my.cnf.d /etc/mysql/mysql.conf.d; do
    [[ -d "$d" ]] && MYSQL_CONF_DIR="$d" && break
done

if [[ -n "$MYSQL_CONF_DIR" ]]; then
    MY_CNF=""
    for f in "$MYSQL_CONF_DIR/mysqld.cnf" "$MYSQL_CONF_DIR/my.cnf" "/etc/my.cnf"; do
        [[ -f "$f" ]] && MY_CNF="$f" && break
    done

    if [[ -n "$MY_CNF" ]]; then
        BIND_ADDR=$(grep -E "^bind-address" "$MY_CNF" 2>/dev/null | awk '{print $3}' || true)
        if [[ "$BIND_ADDR" == "127.0.0.1" ]]; then
            warn "MySQL bind-address is 127.0.0.1 — Docker containers cannot connect!"
            warn "Updating to 0.0.0.0 (all interfaces) — restrict with firewall rules."
            sed -i "s/^bind-address.*=.*/bind-address = 0.0.0.0/" "$MY_CNF"
            systemctl restart mysql || systemctl restart mysqld || true
            ok "MySQL bind-address updated and service restarted"
        else
            ok "MySQL bind-address: '${BIND_ADDR:-not set (defaults to all)}' — Docker can connect"
        fi
    else
        warn "MySQL config file not found — skipping bind-address check"
        warn "If connection fails, add: bind-address = 0.0.0.0 to your MySQL config"
    fi
else
    warn "MySQL config directory not found — skipping bind-address check"
fi

info "Reminder: Grant MySQL access to Docker bridge IP in your MySQL shell:"
echo "  GRANT SELECT ON prod_dump.* TO 'ocr_readonly'@'172.17.0.%' IDENTIFIED BY 'your-password';"
echo "  FLUSH PRIVILEGES;"

# =============================================================================
section "Step 6: Deploy Application"
# =============================================================================

# Create deploy directory
mkdir -p "$DEPLOY_DIR"
cd "$DEPLOY_DIR"

if [[ -d ".git" ]]; then
    info "Git repo already exists — pulling latest changes..."
    git fetch origin
    git reset --hard "origin/${BRANCH}"
    ok "Repo updated to latest on branch: $BRANCH"
else
    info "Cloning repo from: $REPO_URL"
    git clone --branch "$BRANCH" "$REPO_URL" .
    ok "Repo cloned"
fi

# Copy .env.example → .env only if .env doesn't exist
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    warn "Created .env from .env.example"
    warn "⚠️  IMPORTANT: Edit .env now and fill in real credentials before starting the service!"
    warn "   nano $DEPLOY_DIR/.env"
    echo ""
    read -p "Press ENTER after editing .env to continue, or Ctrl+C to stop..." _dummy
else
    ok ".env already exists — keeping existing credentials"
fi

# =============================================================================
section "Step 7: Build and Start Docker Container"
# =============================================================================

info "Building Docker image..."
docker compose build --no-cache

info "Starting service..."
docker compose up -d

# Wait for health check
info "Waiting for service to be healthy..."
sleep 10

MAX_WAIT=60
WAITED=0
until curl -sf http://localhost:${CONTAINER_PORT}/health &>/dev/null; do
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        error "Service did not become healthy within ${MAX_WAIT}s\nCheck logs: docker logs ocr-microservice"
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    info "Still waiting... (${WAITED}s)"
done

ok "Service is healthy on localhost:${CONTAINER_PORT}"

# =============================================================================
section "Step 8: Configure Nginx Reverse Proxy"
# =============================================================================

NGINX_CONF="/etc/nginx/sites-available/ocr-microservice"
NGINX_ENABLED="/etc/nginx/sites-enabled/ocr-microservice"

# Remove default site if present
[[ -f /etc/nginx/sites-enabled/default ]] && rm -f /etc/nginx/sites-enabled/default

cat > "$NGINX_CONF" <<NGINX_EOF
# OCR Microservice — Nginx Reverse Proxy
# HTTP → HTTPS redirect (will be upgraded by Certbot)
server {
    listen 80;
    server_name ${DOMAIN:-_};

    location / {
        proxy_pass         http://127.0.0.1:${APP_PORT};
        proxy_http_version 1.1;
        proxy_set_header   Upgrade \$http_upgrade;
        proxy_set_header   Connection keep-alive;
        proxy_set_header   Host \$host;
        proxy_set_header   X-Real-IP \$remote_addr;
        proxy_set_header   X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;
        proxy_read_timeout 120s;
        client_max_body_size 15M;
    }

    # Health check bypass (no auth required)
    location /health {
        proxy_pass http://127.0.0.1:${APP_PORT}/health;
        access_log off;
    }
}
NGINX_EOF

# Enable site
ln -sf "$NGINX_CONF" "$NGINX_ENABLED"
nginx -t && systemctl reload nginx
ok "Nginx configured and reloaded"

# =============================================================================
section "Step 9: Obtain Let's Encrypt SSL Certificate"
# =============================================================================

if [[ -n "$DOMAIN" ]]; then
    info "Obtaining SSL certificate for: $DOMAIN"
    certbot --nginx \
        --non-interactive \
        --agree-tos \
        --redirect \
        --email "admin@${DOMAIN}" \
        -d "$DOMAIN"
    ok "HTTPS certificate obtained and configured"
    ok "Service available at: https://${DOMAIN}"
else
    warn "DOMAIN variable not set — skipping Let's Encrypt SSL setup"
    warn "To enable HTTPS later, re-run with:"
    warn "  DOMAIN=ocr.yourcompany.com sudo ./deploy/setup_server.sh"
    ok "Service available at: http://<SERVER_IP>:${APP_PORT}"
fi

# =============================================================================
section "Step 10: Final Status"
# =============================================================================

echo ""
echo -e "${BOLD}Container status:${NC}"
docker ps --filter name=ocr-microservice --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo -e "${BOLD}Recent logs:${NC}"
docker logs ocr-microservice --tail 20

echo ""
ok "══════════════════════════════════════════════════════"
ok "  OCR Microservice deployment complete!"
ok "══════════════════════════════════════════════════════"
echo ""
info "Useful commands:"
echo "  View logs:      docker logs ocr-microservice -f"
echo "  Restart:        docker compose -f $DEPLOY_DIR/docker-compose.yml restart"
echo "  Update & redeploy: cd $DEPLOY_DIR && git pull && docker compose up -d --build"
echo "  Health check:   curl http://localhost:${APP_PORT}/health"
echo ""
