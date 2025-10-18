

# SMS Spam Detection System

A real-time SMS spam detection system featuring a complete NLP pipeline powered by PySpark, a RESTful API built with FastAPI, and a dynamic frontend developed with vanilla JavaScript.

## Project Overview

This project is a full-stack machine learning application designed to classify SMS messages as either "Spam" or "Not Spam" (Ham) in real-time. The core of the project is a Naive Bayes classifier trained on a robust NLP pipeline. The model is exposed via a high-performance FastAPI backend, and the interactive user interface is built with modern HTML, CSS, and JavaScript, ensuring a seamless user experience.

The goal was not just to build a model, but to demonstrate an end-to-end understanding of a production-ready ML system—from data processing and model training to API development and frontend integration.

-----

## Technical Architecture

The system is designed with a decoupled frontend and backend, which is a standard practice for scalable web applications. The data flows from the user, through the API, to the ML model, and back.

1.  **Frontend (Client-Side)**:

      * Built with **HTML5, CSS3, and Vanilla JavaScript**.
      * Uses the **`fetch` API** for asynchronous communication with the backend, providing a non-blocking, responsive user experience.
      * Features a clean, modern UI with dynamic animations and visual feedback (loading spinners, color-coded results) to clearly communicate the system's status to the user.

2.  **Backend (API Server)**:

      * Developed using **FastAPI**, a modern, high-performance Python web framework chosen for its speed, automatic interactive documentation (via Swagger UI), and data validation powered by Pydantic.
      * Provides a single RESTful endpoint (`/predict`) that accepts an SMS message as a query parameter.
      * Handles all the business logic, including receiving requests, invoking the ML pipeline, and returning the prediction in a structured JSON format.

3.  **Machine Learning Pipeline**:

      * **Data Processing**: Implemented using **PySpark** to handle the NLP preprocessing steps. This choice demonstrates the ability to work with scalable data processing frameworks, even if the initial dataset is small. The pipeline includes:
          * **Tokenization**: Splitting text into individual words.
          * **Stop Word Removal**: Filtering out common words (e.g., "the", "a", "is").
          * **TF-IDF Vectorization**: Converting the processed text into numerical feature vectors that the model can understand.
      * **Model**: A **Multinomial Naive Bayes** classifier was chosen for its excellent performance and efficiency in text classification tasks. The model was trained on the popular [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

-----

## Challenges & Learnings

This section highlights key technical hurdles and the solutions implemented, demonstrating problem-solving skills.

#### **Challenge 1: Real-Time Prediction with PySpark**

  * **Problem**: PySpark has a significant startup overhead, which could introduce latency on the first API call, making real-time prediction slow.
  * **Solution**: To mitigate this, the trained PySpark ML pipeline and model are **loaded into memory once** when the FastAPI application starts. This ensures that subsequent prediction requests are handled instantly, as the Spark context and model are already initialized and ready. This is a common pattern for productionizing ML models.

#### **Challenge 2: Graceful Frontend-Backend Communication**

  * **Problem**: Network failures or server-side errors could cause the application to crash or hang without informing the user.
  * **Solution**: The frontend JavaScript includes **robust error handling** within the `fetch` call's `.catch()` block. It checks the HTTP status of the response and can parse error messages from the API. This ensures that if the backend sends an error (e.g., 500 Internal Server Error), the user is shown a clear, user-friendly error message instead of a broken interface.

-----

## How to Run Locally

Clear, concise instructions are crucial for demonstrating professionalism and ensuring others can replicate your work.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/SmsSpamDetector.git
    cd SmsSpamDetector
    ```

2.  **Set up the Backend:**

    ```bash
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate 

    # Install Python dependencies
    pip install -r requirements.txt

    # Run the FastAPI server
    uvicorn main:app --host 0.0.0.0 --port 5001
    ```

    The API will now be running at `http://localhost:5001`.

3.  **Launch the Frontend:**

      * Navigate to the `frontend` directory.
      * Open the `index.html` file in your web browser.

-----

## Future Improvements

This shows forward-thinking and an awareness of industry-standard practices.

  * **Containerization**: Dockerize the frontend and backend services using `docker-compose` to streamline the setup process and ensure environment consistency.
  * **Deployment**: Deploy the application to a cloud platform like AWS (using EC2 for the backend and S3 for the frontend) or Heroku for public access.
  * **Model Enhancement**: Experiment with more advanced NLP models like LSTMs or Transformers (using Hugging Face) to potentially improve classification accuracy.
  * **CI/CD Pipeline**: Implement a CI/CD pipeline using GitHub Actions to automate testing and deployment whenever new code is pushed to the main branch.
