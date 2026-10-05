# 📈 Financial Intelligence Platform

An enterprise-grade Financial Analytics Platform built in Python, demonstrating end-to-end data engineering, analytics, machine learning, and interactive dashboarding.

🚀 **Live App:** [https://financial-intelligence-platform-e9xq.onrender.com](https://financial-intelligence-platform-e9xq.onrender.com)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Financial Intelligence Platform               │
├───────────────┬──────────────┬──────────────┬───────────────────┤
│   API Layer   │  ETL Layer   │ Analytics /  │   Dashboard       │
│               │              │     ML       │                   │
│  FRED Client  │  Yahoo ETL   │  returns.py  │  Sidebar          │
│  Yahoo Client │  FRED  ETL   │  volatility  │  KPI Cards        │
│               │              │  rolling.py  │  Stock Page       │
│               │              │  loader.py   │  Commodity Page   │
│               │              │  Prophet     │  Economic Page    │
│               │              │  Forecaster  │  Correlation Page │
└───────┬───────┴──────┬───────┴──────┬───────┴───────────────────┘
        │              │              │
        └──────────────▼──────────────┘
                 PostgreSQL (Neon)
```

---

## ✅ Features

| Feature | Details |
|---|---|
| **Stock Analysis** | Interactive price chart, 30-day SMA, daily returns, volatility, 30-day Prophet forecast |
| **Commodity Analysis** | Same analytics engine applied to oil, gold, silver, natural gas |
| **Economic Indicators** | Live FRED API data — GDP, Unemployment Rate, CPI, Federal Funds Rate |
| **Correlation Matrix** | Pearson correlation heatmap across any combination of assets |
| **Production Ready** | Dockerfile, docker-compose, gunicorn, GitHub Actions CI/CD |

---

## 🗂️ Project Structure

```
Financial Intelligence Platform/
├── src/
│   ├── api/
│   │   ├── fred/          # St. Louis Fed (FRED) API client
│   │   └── yahoo/         # Yahoo Finance API client
│   ├── analytics/
│   │   ├── loader.py      # DB → DataFrame interface
│   │   ├── returns.py     # Daily / monthly return calculations
│   │   ├── volatility.py  # Historical & Parkinson volatility
│   │   └── rolling.py     # Simple moving averages
│   ├── dashboard/
│   │   ├── app.py         # Dash application entry point
│   │   ├── components/    # Reusable UI components (sidebar, KPI cards)
│   │   ├── layouts/       # Page layouts (stock, commodity, economic, correlation)
│   │   └── callbacks/     # Dash callbacks (data fetching + chart rendering)
│   ├── database/          # SQLAlchemy engine, session factory, models
│   ├── etl/               # ETL pipelines (Yahoo Finance, FRED)
│   ├── ml/                # Prophet forecasting model
│   └── statistics/        # Statistical analysis modules
├── Dockerfile
├── docker-compose.yml
├── gunicorn.conf.py
├── .github/workflows/ci.yml
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start (Local Development)

### Prerequisites

- Python 3.11+
- A [Neon PostgreSQL](https://neon.tech) account (free tier works)
- A [FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html) (free)

### 1. Clone and set up the environment

```bash
git clone <your-repo-url>
cd "Financial Intelligence Platform"

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```env
POSTGRES_HOST=ep-your-endpoint.neon.tech
POSTGRES_PORT=5432
POSTGRES_DB=neondb
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
FRED_API_KEY=your_fred_api_key
```

### 3. Run database migrations

```bash
alembic upgrade head
```

### 4. Run the ETL pipeline (load data)

```bash
python -m src.etl.yahoo_etl   # Load stock prices
python -m src.etl.fred_etl    # Load economic data
```

### 5. Start the dashboard

```bash
python src/dashboard/app.py
```

Then open **http://127.0.0.1:8050** in your browser.

---

## 🐳 Run with Docker

```bash
# Build and start
docker-compose up --build

# Stop
docker-compose down
```

The dashboard will be available at **http://localhost:8050**.

---

## 🔄 CI/CD Pipeline

GitHub Actions automatically runs on every push to `main` or `develop`:

| Step | Tool | What it checks |
|---|---|---|
| Formatting | `black` | Code style compliance |
| Linting | `flake8` | Syntax errors, undefined names |
| Type checking | `mypy` | Static type correctness |
| Tests | `pytest` | Unit tests in `tests/` |
| Docker build | Docker Buildx | Image compiles cleanly |

---

## 📊 Dashboard Pages

| Page | URL | Description |
|---|---|---|
| Home | `/` | Landing page |
| Stock Analysis | `/stocks` | Price chart, SMA, volatility, Prophet forecast |
| Commodity Analysis | `/commodities` | Oil, gold, silver, natural gas analysis |
| Economic Indicators | `/economic` | GDP, CPI, Unemployment, Fed Funds Rate via FRED |
| Correlation Matrix | `/correlation` | Pearson correlation heatmap across assets |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Database | PostgreSQL (Neon Serverless) |
| ORM | SQLAlchemy 2.0 |
| Migrations | Alembic |
| API Clients | httpx, yfinance |
| Analytics | pandas, numpy, scipy |
| ML / Forecasting | Prophet (Meta) |
| Dashboard | Dash, Plotly, dash-bootstrap-components |
| Production Server | Gunicorn |
| Containerisation | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Logging | structlog (JSON format) |

---

## 📋 Development Principles

- One module per prompt — every module is independently testable
- No placeholder code, no TODO comments
- Production-quality code following PEP 8
- Modular architecture — each layer is decoupled
- Every module runs before the next is built

---

## 📄 License

MIT License. See `LICENSE` for details.
