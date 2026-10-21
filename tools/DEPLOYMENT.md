# TG-ReDial Chatbot Deployment Guide

## Model Selection

No recommender model is bundled. `TG-ReDial-main/chat_app.py` exposes
`make_handler(recommender)`; bring your own recommender object implementing
`recommend(context, top_k)` (and optionally `recommend_with_context`).

## Local Deployment

### Prerequisites

- Python 3.11+
- Virtual environment

### Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Bring your own recommender: create a server module that imports
# make_handler from TG-ReDial-main/chat_app.py and passes your model in.
# No model checkpoint is bundled or required at build time.

```

Access the chat at: <http://localhost:8000>

## Azure Deployment

### Option 1: Azure App Service (Recommended)

#### Azure App Service Prerequisites

- Azure subscription
- Azure CLI installed
- Docker installed

#### Azure App Service Deployment Steps

##### Build and Push Docker Image

```bash
# Login to Azure Container Registry
az acr login --name <your-registry-name>

# Build the image
docker build -t <your-registry-name>.azurecr.io/tg-redial-chatbot:latest .

# Push the image
docker push <your-registry-name>.azurecr.io/tg-redial-chatbot:latest

```

##### Create Azure App Service

```bash
# Create resource group
az group create --name tg-redial-rg --location eastus

# Create App Service plan
az appservice plan create \
    --name tg-redial-plan \
    --resource-group tg-redial-rg \
    --sku B1 \
    --is-linux

# Create web app
az webapp create \
    --name tg-redial-chatbot \
    --resource-group tg-redial-rg \
    --plan tg-redial-plan \
    --deployment-container-image-name <your-registry-name>.azurecr.io/tg-redial-chatbot:latest

# Configure settings
az webapp config appsettings set \
    --resource-group tg-redial-rg \
    --name tg-redial-chatbot \
    --settings WEBSITES_PORT=8000

```

##### Verify Deployment

```bash
# Get the URL
az webapp show \
    --resource-group tg-redial-rg \
    --name tg-redial-chatbot \
    --query defaultHostName \
    --output tsv

```

### Option 2: Azure Container Apps

#### Azure Container Apps Prerequisites

- Azure subscription
- Azure CLI installed

#### Azure Container Apps Deployment Steps

```bash
# Create resource group
az group create --name tg-redial-rg --location eastus

# Create Container Apps environment
az containerapp env create \
    --name tg-redial-env \
    --resource-group tg-redial-rg \
    --location eastus

# Build and push image (same as Option 1)

# Create container app
az containerapp create \
    --name tg-redial-chatbot \
    --resource-group tg-redial-rg \
    --environment tg-redial-env \
    --image <your-registry-name>.azurecr.io/tg-redial-chatbot:latest \
    --target-port 8000 \
    --ingress external \
    --cpu 0.5 \
    --memory 1Gi

```

## Configuration

### Environment Variables

The chat shell supports the following environment variables (used by your own
server module and the optional reranker):

- `HOST`: Server host (default: 127.0.0.1)
- `PORT`: Server port (default: 8000)
- `OPENAI_API_KEY`: Optional OpenAI API key for reranking
- `OPENAI_MODEL`: OpenAI model name for reranking
- `OPENAI_BASE_URL`: OpenAI API base URL

### Scaling Considerations

- **CPU**: Start with 0.5-1 vCPU
- **Memory**: 1-2 GB RAM
- **Storage**: ~200 MB for model files
- **Inference**: ~50-100ms per request on CPU

## Monitoring

### Health Check

```bash
curl http://<your-app-url>/api/health

```

Expected response:

```json
{"status": "ok"}

```

### Logs (Azure)

```bash
# View logs
az webapp log tail --resource-group tg-redial-rg --name tg-redial-chatbot

```

## Troubleshooting

### Common Issues

##### Port binding errors

- Ensure the port is not already in use
- Check firewall settings

##### Memory errors

- Increase memory allocation in Azure App Service

## Security Considerations

- Use HTTPS in production
- Implement authentication if needed
- Add rate limiting to prevent abuse
- Keep dependencies updated
- Use secrets management for API keys

## Cost Estimate (Azure)

**App Service (B1 tier)**:

- ~$13/month for basic tier
- Includes 1 vCPU, 1.75 GB RAM
- Good for development/testing

**Container Apps**:

- Pay-per-use pricing
- ~$0.0000065 per vCPU-second
- ~$0.000005 per GB-second
- Cost-effective for variable traffic
