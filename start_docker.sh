cd encodecmae
docker build -t encodecmae:latest .
docker compose up -d
docker attach encodecmae-train