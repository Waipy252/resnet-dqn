# Transformer DQN

This repository contains a Transformer-based Deep Q-Network (DQN) implementation for trading strategies, specifically applied to the Nikkei 225 index.

## Project Structure

- `try18res_py12/`: Main project directory
  - `visualize.py`: Gradio-based visualization tool for DQN model performance
  - `main.py`: Environment and training logic
  - `data.py`: Data generation and processing
  - `calc_performance.py`: Performance metrics calculation

## Quick Start

### Using Docker Compose (Recommended)

1. Navigate to the project directory:
   ```bash
   cd try18res_py12
   ```

2. Build and run the visualization tool:
   ```bash
   docker compose up -d --build
   ```

3. Open your browser and go to `http://localhost:13000` (or the port mapped in docker-compose.yml)

### Using uv (Local Development)

1. Install dependencies:
   ```bash
   cd try18res_py12
   uv sync
   ```

2. Run the visualization tool:
   ```bash
   uv run visualize.py
   ```

3. Open your browser and go to `http://localhost:7860`

## Features

- **Model Performance Comparison**: Compare DQN models trained at different steps
- **Performance Metrics**: View annual return, Sharpe ratio, max drawdown, win rate, etc.
- **Action Distribution**: Visualize buy/hold/sell action distributions
- **Equity Curves**: Compare individual model equity curves with ensemble results
- **Test Data**: View test data used for evaluation

## Docker Configuration

- `Dockerfile`: Python 3.12 slim image with uv package manager
- `docker-compose.yml`: Service configuration for the visualization tool
- `.dockerignore`: Files to exclude from the Docker build context

## Dependencies

- Python 3.12+
- Gradio (>=6.9.0)
- Stable Baselines3 (>=2.7.1)
- PyTorch (2.8.0)
- Plotly (>=6.6.0)
- Pandas (<3.0)
- yfinance
- gym/gymnasium

## Notes

- The project uses `uv` for dependency management
- Gradio interface runs on port 7860 (configurable via GRADIO_SERVER_PORT environment variable)
- Docker Compose maps port 7860 to host port 13000
