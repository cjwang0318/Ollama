from flask import Flask, jsonify, request
from langchain_query import do_query
from threading import Thread
import logging
import time

app = Flask(__name__)

# Status dictionary to store the status of each request
status = {"state": "idle", "result": None}

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def generate_results(model_type, query):
    global status
    status["state"] = "processing"
    # Simulate image generation process
    # time.sleep(10)  # Replace this with the actual image generation logic
    result = do_query(model_type, query)
    status["state"] = "complete"
    status["result"] = result
    # Log the result of the query
    logging.info(f"Query result: {result}")


@app.route('/text_generation', methods=['POST'])
def forecasting_route():
    global status
    if status["state"] == "processing":
        return jsonify({"message": "Forecasting is already in progress"}), 409

    # Reset result to None before starting new generation
    status["result"] = None

    # Parse JSON data from the request
    json_request = request.get_json()
    # print(json_request)
    # Log the incoming JSON request
    logging.info(f"Received JSON request: {json_request}")
    # Extract metadata if needed
    model_type = json_request.get("model_type")
    query = json_request.get("query")
    thread = Thread(target=generate_results, args=(model_type, query))
    thread.start()

    return jsonify({"message": "Generating started"}), 202


@app.route('/status', methods=['GET'])
def status_route():
    global status
    return jsonify(status), 200


if __name__ == '__main__':
    app.run(host='192.168.50.26', port=8000, debug=True)
