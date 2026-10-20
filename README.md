# AI Invoice OCR Microservice

An AI-powered document processing microservice built with **FastAPI, Mistral OCR, Google Gemini, and MySQL** to extract structured information from invoices and delivery challans for textile ERP workflows.

The service converts scanned documents and images into structured JSON, validates document contents, and matches extracted entities against ERP master data.

## Features

- **AI-powered OCR:** Extract text from PDF documents and invoice images using Mistral OCR.
- **Image preprocessing:** Correct image orientation, deskew scanned documents, and enhance contrast and sharpness using OpenCV and Pillow.
- **Structured data extraction:** Use Google Gemini to convert OCR text into structured JSON based on a configurable output schema.
- **ERP master-data matching:** Resolve extracted entities against MySQL master data using fuzzy matching and exact-match workflows.
- **Confidence scoring:** Return match confidence scores and alternative candidates to support review and correction.
- **Textile ERP classification:** Identify inward transaction types using buyer, consignee, GST, and PAN details.
- **Line-item extraction:** Extract tabular item details and group the response into basic details, line items, and summary sections.
- **Document validation:** Detect multiple distinct delivery challans in uploaded documents.
- **REST API:** Expose document processing workflows through FastAPI endpoints with API-key authentication.
- **Interactive UI:** Use the included web interface to upload documents and review extracted information.
- **Production deployment:** Docker, Docker Compose, Nginx, HTTPS, health checks, and structured logging support.

## Tech Stack

| Category | Technologies |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| OCR | Mistral OCR |
| AI parsing | Google Gemini |
| Database | MySQL, aiomysql, PyMySQL |
| Entity matching | RapidFuzz |
| Image processing | OpenCV, Pillow, NumPy |
| Validation | Pydantic |
| Deployment | Docker, Docker Compose, Nginx |
| Testing | Pytest, pytest-asyncio |

## Architecture

```text
Invoice / Delivery Challan
            |
            v
     FastAPI REST API
            |
            v
 File Validation & Preprocessing
            |
            v
       Mistral OCR
            |
            v
      Extracted Text
            |
            v
    Google Gemini Parser
            |
            v
   Schema-Based JSON Data
            |
            v
  ERP Master-Data Matching
            |
            v
  Validation & Classification
            |
            v
 Structured JSON API Response
```

MySQL provides the ERP master data used for entity resolution. The service maintains an in-memory entity index and refreshes it periodically.

## Project Structure

```text
Invoice_OCR_Microservice/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── db/
│   │   └── indexer.py
│   ├── models/
│   │   ├── enums.py
│   │   ├── request.py
│   │   └── response.py
│   ├── ocr/
│   │   ├── extractor.py
│   │   ├── parser.py
│   │   ├── matcher.py
│   │   ├── inward_classifier.py
│   │   └── challan_validator.py
│   ├── schemas/
│   │   └── fabric_delivery_challan.json
│   └── utils/
│       ├── file_handler.py
│       └── image_processor.py
├── tests/
├── ui/
│   └── index.html
├── deploy/
│   ├── README.md
│   ├── nginx.conf
│   └── setup_server.sh
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── deploy.sh
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.10 or newer
- A Mistral API key
- A Google AI API key
- MySQL credentials with appropriate read permissions for the configured ERP master tables
- Docker and Docker Compose for containerized deployment

### 1. Clone the repository

```bash
git clone https://github.com/priyank-danavle/Invoice_OCR_Microservice.git
cd Invoice_OCR_Microservice
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Update `.env` with your credentials and database configuration.

```env
MISTRAL_API_KEY=your-mistral-api-key
GOOGLE_API_KEY=your-google-api-key
GEMINI_PARSE_MODEL=gemini-3.5-flash-lite

MICROSERVICE_API_KEY=your-strong-random-api-key

DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=your_database
DB_USER=your_readonly_user
DB_PASSWORD=your_database_password
DB_READ_TIMEOUT=10

ENTITY_INDEX_TTL_MINUTES=60
MAX_FILE_SIZE_MB=10

ALLOWED_ORIGINS=http://localhost:4200
LOG_LEVEL=INFO
PORT=8001
```

Use the actual model identifier supported by your Google AI account. Ensure the database tables and columns match the master-table mappings configured in `app/config.py`.

**Security:** Never commit `.env`, real API keys, database passwords, or production credentials to GitHub.

### 5. Start the application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

The API will be available at:

- API documentation: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`
- Health check: `http://localhost:8001/health`
- Web interface: `http://localhost:8001/ui/`

The service attempts to load ERP master data during startup. Entity matching requires a working database connection and a successfully loaded index.

## API Endpoints

All processing endpoints require the configured API key in the `X-API-Key` request header.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/invoice/parse` | Complete OCR and structured parsing workflow |
| POST | `/api/v1/invoice/extract` | Extract raw OCR text; optionally perform initial schema-based extraction |
| POST | `/api/v1/invoice/process` | Process OCR text or edited JSON into structured fields and resolve entities |
| GET | `/health` | Check service health and entity-index status |
| POST | `/api/v1/index/refresh` | Manually refresh the ERP master-data index |

### Example: Health Check

```bash
curl http://localhost:8001/health
```

### Example: Parse an Invoice

The parsing endpoint accepts multipart form data containing a document, a positive `client_id`, and an `output_schema` JSON string.

```bash
curl -X POST \
  http://localhost:8001/api/v1/invoice/parse \
  -H "X-API-Key: YOUR_MICROSERVICE_API_KEY" \
  -F "file=@invoice.pdf" \
  -F "client_id=123" \
  -F 'output_schema={"field_name":{"type":"text"}}'
```

Replace the sample schema with a valid schema matching the structure defined by the application. The example above illustrates the request format, not a complete textile ERP schema.

Supported document types are JPEG, PNG, and PDF, with a default maximum file size of 10 MB per file.

## Two-Step Processing Workflow

The service also supports a workflow that separates OCR extraction from structured processing.

**Step 1 — Extract**

`POST /api/v1/invoice/extract`

Upload documents and obtain OCR text. An optional output schema can be supplied for an initial structured extraction.

**Step 2 — Process**

`POST /api/v1/invoice/process`

Submit the OCR text or user-corrected extracted JSON together with the client identifier and output schema. The service parses the data, performs entity resolution, and returns the structured result.

This workflow allows extracted content to be reviewed or corrected before the final processing stage.

## Entity Matching and Validation

The service supports two matching approaches:

- **Fuzzy matching:** Uses RapidFuzz similarity scores to identify likely ERP master-data matches, return confidence levels, and provide alternatives.
- **Exact matching:** Supports strict entity resolution in the processing workflow. Unresolved required `auto_search` fields can cause the request to fail with HTTP 422.

The configurable schema supports field types such as `direct`, `auto_search`, `auto_populated`, and `line_items`. Fields can be organized into `basic_details`, `line_items`, and `summary` sections.

Additional validation detects documents containing multiple distinct delivery challans, while inward classification uses buyer and consignee details and branch GST information.

## Running Tests

Run the test suite from the project root:

```bash
pytest -v
```

The repository includes tests for extraction, parsing, matching, challan validation, inward classification, isolation, and API routes.

Some tests or workflows may require suitable environment configuration or mocked external services.

## Docker Deployment

Create your `.env` file first, then build and start the service:

```bash
docker compose up -d --build
```

Check container status and logs:

```bash
docker compose ps
docker compose logs -f ocr-microservice
```

Check service health:

```bash
curl http://localhost:9001/health
```

The supplied Docker Compose configuration uses host networking and runs Uvicorn on port `9001`, with the application configured to use port `8001` internally in its default application settings. Review `docker-compose.yml` and the deployment documentation before changing the network or port configuration.

For production deployment, refer to [`deploy/README.md`](deploy/README.md) for the server setup, Nginx reverse proxy, HTTPS configuration, and database connectivity requirements.

## Security Considerations

- Keep API keys and database credentials in environment variables.
- Use a dedicated, least-privilege MySQL account for master-data reads.
- Restrict API access to trusted clients and configure allowed CORS origins.
- Use HTTPS for production deployments.
- Avoid exposing internal debug data or sensitive business information in public repositories and logs.

## License

No license is currently specified. Add a `LICENSE` file if you intend to distribute or reuse this project under an open-source license.

