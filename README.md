# MLOps-AI-Application

This repository contains the AI application project for the course **DA5402**.  
GitHub Repository: [MLOps-AI-Application](https://github.com/mohammednawfal/MLOps-AI-Application)

## Overview

This project demonstrates a full MLOps pipeline for image denoising using the CIFAR-10 dataset. It integrates **FastAPI**, **Docker**, **Prometheus**, and **Grafana** to provide a complete deployment and monitoring solution.

---

## Setup Instructions

1. **Extract all files** into a single folder.
2. **Navigate to the root folder** in your terminal.
3. Run the following command to build and start the application:
   ```bash
   docker compose up --build

## Application Endpoints

- **FastAPI (Image Denoising)**  
  Access the API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)  
  > **Note:** This project only supports denoising images from the **CIFAR-10** dataset.  
  > Sample noisy and clean images are provided in the `noisy_test` and `clean_test` folders, respectively.

- **Prometheus Monitoring Interface**  
  [http://localhost:9090](http://localhost:9090)

- **Grafana Dashboard Interface**  
  [http://localhost:3000](http://localhost:3000)

  To configure Prometheus as a data source in Grafana:
  1. Go to **Settings > Data Sources**.
  2. Add a new Prometheus data source with the URL:
     ```
     http://prometheus:9090
     ```
  3. You can now use Prometheus metrics to create custom dashboards.
