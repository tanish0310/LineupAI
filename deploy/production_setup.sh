#!/bin/bash

# FPL Optimizer Production Deployment Script
set -e

echo "🚀 Starting FPL Optimizer Production Deployment..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating environment configuration..."
    cat > .env << EOF
# Database Configuration
DB_PASSWORD=$(openssl rand -base64 32)

# API Configuration
API_KEYS=$(openssl rand -base64 32)

# Monitoring Configuration
GRAFANA_PASSWORD=$(openssl rand -base64 16)

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=info

# External APIs
FPL_API_BASE=https://fantasy.premierleague.com/api

# Security
JWT_SECRET=$(openssl rand -base64 64)
EOF
    print_success "Environment file created with secure random passwords"
else
    print_warning "Environment file already exists, skipping creation"
fi

# Create required directories
print_status "Creating required directories..."
mkdir -p {logs,models/saved,monitoring/{prometheus,grafana/{dashboards,datasources}},nginx/ssl,database}

# Generate SSL certificates for development (replace with real certificates for production)
if [ ! -f nginx/ssl/cert.pem ]; then
    print_status "Generating self-signed SSL certificates..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/key.pem \
        -out nginx/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    print_success "SSL certificates generated"
fi

# Create Nginx configuration
print_status "Creating Nginx configuration..."
cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }
    
    upstream frontend {
        server frontend:8501;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=frontend_limit:10m rate=30r/s;
    
    server {
        listen 80;
        server_name localhost;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl;
        server_name localhost;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
        
        # API routes
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            proxy_pass http://api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Frontend routes
        location / {
            limit_req zone=frontend_limit burst=50 nodelay;
            proxy_pass http://frontend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
        
        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
EOF

# Create Prometheus configuration
print_status "Creating monitoring configuration..."
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'fpl-optimizer-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
EOF

# Create Grafana datasource
mkdir -p monitoring/grafana/datasources
cat > monitoring/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# Create requirements files
print_status "Creating requirements files..."
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
asyncpg==0.29.0
pandas==2.1.4
numpy==1.24.4
scikit-learn==1.3.2
xgboost==2.0.3
pulp==2.7.0
redis==5.0.1
requests==2.31.0
python-multipart==0.0.6
python-dotenv==1.0.0
prometheus-client==0.19.0
pydantic==2.5.2
alembic==1.13.1
psycopg2-binary==2.9.9
aiofiles==23.2.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
EOF

cat > requirements-frontend.txt << 'EOF'
streamlit==1.28.2
plotly==5.17.0
requests==2.31.0
pandas==2.1.4
numpy==1.24.4
python-dotenv==1.0.0
streamlit-autorefresh==0.0.1
streamlit-aggrid==0.3.4
streamlit-option-menu==0.3.6
EOF

# Pull and build Docker images
print_status "Building Docker images..."
docker-compose build --no-cache

# Start services
print_status "Starting services..."
docker-compose up -d

# Wait for services to be healthy
print_status "Waiting for services to be healthy..."
sleep 30

# Check service health
print_status "Checking service health..."
services=("db" "redis" "api" "frontend")
all_healthy=true

for service in "${services[@]}"; do
    if docker-compose ps --services --filter health=healthy | grep -q "^${service}$"; then
        print_success "${service} is healthy"
    else
        print_error "${service} is not healthy"
        all_healthy=false
    fi
done

if [ "$all_healthy" = true ]; then
    print_success "All services are running and healthy!"
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo ""
    echo "📊 Access your FPL Optimizer at:"
    echo "   Frontend: https://localhost (Streamlit UI)"
    echo "   API Docs: https://localhost/api/docs (FastAPI Documentation)"
    echo "   Monitoring: http://localhost:3000 (Grafana Dashboard)"
    echo "   Metrics: http://localhost:9090 (Prometheus)"
    echo ""
    echo "🔐 Important Security Notes:"
    echo "   - Change default passwords in .env file"
    echo "   - Replace self-signed certificates with real SSL certificates"
    echo "   - Configure firewall rules for production"
    echo "   - Set up proper backup procedures"
    echo ""
    echo "📋 Next Steps:"
    echo "   1. Configure your FPL API credentials"
    echo "   2. Run initial data import: docker-compose exec api python -m scripts.initial_setup"
    echo "   3. Train ML models: docker-compose exec api python -m scripts.train_models"
    echo "   4. Set up monitoring alerts in Grafana"
    echo ""
else
    print_error "Some services are not healthy. Check logs with: docker-compose logs"
    exit 1
fi

print_success "FPL Optimizer deployment completed!"
