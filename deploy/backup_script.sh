#!/bin/bash

# FPL Optimizer Backup Script
set -e

BACKUP_DIR="/backups/fpl_optimizer"
DATE=$(date +%Y%m%d_%H%M%S)

echo "🔄 Starting backup process..."

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup database
echo "📊 Backing up database..."
docker-compose exec -T db pg_dump -U fpl_user fpl_optimizer | gzip > "$BACKUP_DIR/database_backup_$DATE.sql.gz"

# Backup ML models
echo "🤖 Backing up ML models..."
tar -czf "$BACKUP_DIR/models_backup_$DATE.tar.gz" models/saved/

# Backup configuration
echo "⚙️ Backing up configuration..."
tar -czf "$BACKUP_DIR/config_backup_$DATE.tar.gz" .env docker-compose.yml nginx/

# Clean old backups (keep last 7 days)
echo "🧹 Cleaning old backups..."
find "$BACKUP_DIR" -type f -mtime +7 -delete

echo "✅ Backup completed: $BACKUP_DIR"
