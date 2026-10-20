# OCR Microservice — Deployment Guide

> Mistral-powered invoice OCR for textile ERP · FastAPI · Docker · Nginx · Let's Encrypt

---

## Architecture

```
Internet (HTTPS:443)
    │
    ▼
[ Nginx ]  ←── SSL terminated here, proxy_pass → 127.0.0.1:9001
    │
    ▼
[ Docker: ocr-microservice ] — host:9001 → container:8001 (Uvicorn/FastAPI)
    │
    ▼
[ MySQL ]  ←── same server, reached via Docker bridge 172.17.0.1:3306
```

---

## Pre-Deployment Checklist

- [ ] You have a domain/subdomain pointed at the server's public IP (for HTTPS)
- [ ] GitHub repo is public or you have deploy keys set up
- [ ] You have your Mistral API key ready
- [ ] You know your MySQL DB credentials
- [ ] Port 9001 is allowed in your server's firewall

---

## Step 1 — Push Local Code to GitHub

```bash
# On your local machine
cd /path/to/OCR_microservice

# If you haven't added a remote yet:
git remote add origin https://github.com/YOUR_USERNAME/OCR_microservice.git

# Commit latest changes (if any)
git add -A
git commit -m "chore: deploy config — fix port mapping, add setup scripts"

git push -u origin master
```

---

## Step 2 — SSH into Your Remote Server

```bash
ssh user@YOUR_SERVER_IP
```

---

## Step 3 — Run the Setup Script

```bash
# Download just the setup script first
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/OCR_microservice/master/deploy/setup_server.sh \
    -o setup_server.sh

# Set your repo URL and domain, then run
export REPO_URL="https://github.com/YOUR_USERNAME/OCR_microservice.git"
export DOMAIN="ocr.yourcompany.com"   # must point to this server's IP
export DEPLOY_DIR="/opt/ocr-microservice"
export BRANCH="master"

chmod +x setup_server.sh
sudo -E ./setup_server.sh
```

The script will:
1. **Check** git, curl, docker, docker compose, nginx, certbot — install only what's missing
2. **Check** port 9001 is free
3. **Fix** MySQL bind-address so Docker can reach it via `172.17.0.1`
4. **Clone** the repo to `/opt/ocr-microservice`
5. **Pause** and ask you to fill in `.env`
6. **Build** the Docker image and start the container
7. **Configure** Nginx and obtain a Let's Encrypt certificate

---

## Step 4 — Fill In `.env`

When the script pauses, edit the env file:

```bash
nano /opt/ocr-microservice/.env
```

```env
MISTRAL_API_KEY=your-real-mistral-key
MICROSERVICE_API_KEY=$(openssl rand -hex 32)   # generate a strong key
DB_HOST=172.17.0.1     # Docker bridge — reaches MySQL on the server host
DB_PORT=3306
DB_NAME=prod_dump
DB_USER=ocr_readonly
DB_PASSWORD=your-mysql-password
ALLOWED_ORIGINS=https://ocr.yourcompany.com,https://your-erp-domain.com
LOG_LEVEL=INFO
PORT=8001
```

Then press **Enter** to continue the setup.

---

## Step 5 — MySQL Grant (Run in MySQL Shell)

The Docker container connects to MySQL via the bridge IP `172.17.0.1`. You need to grant access:

```sql
-- Run as MySQL root
CREATE USER IF NOT EXISTS 'ocr_readonly'@'172.17.0.%' IDENTIFIED BY 'your-password';
GRANT SELECT ON prod_dump.* TO 'ocr_readonly'@'172.17.0.%';
FLUSH PRIVILEGES;
```

---

## Verification

```bash
# Container running?
docker ps --filter name=ocr-microservice

# Health endpoint
curl http://localhost:9001/health
curl https://ocr.yourcompany.com/health

# API docs
open https://ocr.yourcompany.com/docs

# Logs
docker logs ocr-microservice -f --tail 50
```

---

## Firewall Rules (UFW)

```bash
sudo ufw allow 22/tcp     # SSH
sudo ufw allow 80/tcp     # HTTP (for Let's Encrypt & redirect)
sudo ufw allow 443/tcp    # HTTPS
# Port 9001 does NOT need to be public — Nginx proxies to it internally
sudo ufw enable
sudo ufw status
```

---

## Day-2 Operations

| Task | Command |
|---|---|
| View live logs | `docker logs ocr-microservice -f` |
| Restart service | `docker compose -f /opt/ocr-microservice/docker-compose.yml restart` |
| Pull update & redeploy | `cd /opt/ocr-microservice && git pull && docker compose up -d --build` |
| Stop service | `docker compose -f /opt/ocr-microservice/docker-compose.yml down` |
| Renew SSL (auto via cron) | `sudo certbot renew --dry-run` |
| Shell into container | `docker exec -it ocr-microservice bash` |

---

## Troubleshooting

### Container exits immediately
```bash
docker logs ocr-microservice
# Usually a missing .env value — check MISTRAL_API_KEY and DB_* vars
```

### MySQL connection refused from container
```bash
# Verify MySQL is listening on 0.0.0.0 (not just 127.0.0.1)
sudo ss -tlnp | grep 3306
# If only 127.0.0.1:3306 — update bind-address in /etc/mysql/mysqld.cnf
```

### Port 9001 already in use
```bash
sudo ss -tlnp | grep 9001
# Kill or move the conflicting process, then retry
```

### SSL certificate fails
```bash
# Ensure DNS A record for ocr.yourcompany.com points to this server's IP
# Ensure port 80 is open in firewall (Let's Encrypt needs it)
dig +short ocr.yourcompany.com
```
