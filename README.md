# MLOps-AI-Application
AI application project for the course DA5402.
You can find the full github repo at: https://github.com/mohammednawfal/MLOps-AI-Application

# Instructions
1. Extract all files into a single folder.
2. From the root folder, run on terminal, "docker compose up --build".
3. The FastAPI endpoint for image denoising can be accessed at "localhost:8000/docs". This project is only limited to denoising images of the CIFFAR-10 dataset. Sample test images and corresponding clean images are present in "noisy_test" and "clean_test".
4. The Prometheus Instrumentation Interface can be accessed at "localhost:9090".
5. The Grafana Interface can be accessed at "localhost:3000". To use Prometheus as a data source, configure the prometheus connection using "http://prometheus:9090". You can then use data siphoned from promtheus in building your dashboard.
