#!/usr/bin/env bash
set -e

# Kết nối vào 'postgres' DB (luôn tồn tại)
DB_CONN="--username $POSTGRES_USER --dbname postgres"

# Tạo rag_db nếu chưa có
if ! psql $DB_CONN -t -c "SELECT 1 FROM pg_database WHERE datname = 'rag_db'" | grep -q 1; then
    echo "Creating rag_db..."
    psql $DB_CONN -c "CREATE DATABASE rag_db;"
else
    echo "rag_db already exists."
fi

# Tạo fastapi_db nếu chưa có
if ! psql $DB_CONN -t -c "SELECT 1 FROM pg_database WHERE datname = 'fastapi_db'" | grep -q 1; then
    echo "Creating fastapi_db..."
    psql $DB_CONN -c "CREATE DATABASE fastapi_db;"
else
    echo "fastapi_db already exists."
fi

# Cấp quyền
psql $DB_CONN -c "GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;"
psql $DB_CONN -c "GRANT ALL PRIVILEGES ON DATABASE fastapi_db TO rag_user;"

echo "Initialization of rag_db and fastapi_db completed."
