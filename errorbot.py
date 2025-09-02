import requests
import json
import time
from datetime import datetime

class ApiErrorReader:
    def __init__(self, api_url, poll_interval_seconds=60, auth=None, headers=None):
        """
        Initializes the ApiErrorReader.

        Args:
            api_url (str): The base URL of the API to monitor.
            poll_interval_seconds (int): How often to check the API (in seconds).
            auth (tuple, optional): Authentication credentials (e.g., ('username', 'password')). Defaults to None.
            headers (dict, optional): Custom headers to include in the requests. Defaults to None.
        """
        self.api_url = api_url
        self.poll_interval = poll_interval_seconds
        self.auth = auth
        self.headers = headers if headers else {}

    def fetch_data(self, endpoint):
        """
        Fetches data from a specific API endpoint.

        Args:
            endpoint (str): The relative path of the endpoint.

        Returns:
            dict or None: The JSON response if successful, None otherwise.
        """
        url = f"{self.api_url}/{endpoint}"
        try:
            response = requests.get(url, auth=self.auth, headers=self.headers)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {url}")
            return None

    def check_for_errors(self, data):
        """
        Analyzes the fetched data for errors. This is highly API-specific.

        Args:
            data (dict): The JSON data fetched from the API.

        Returns:
            list: A list of error dictionaries, each containing relevant error information.
                  Returns an empty list if no errors are found.
        """
        errors = []
        # **Important:** Implement your API's error detection logic here.
        # This is a placeholder and will vary greatly depending on the API's response structure.

        # Example 1: Errors might be in a specific "errors" list
        if isinstance(data, dict) and "errors" in data and isinstance(data["errors"], list):
            for error in data["errors"]:
                errors.append({
                    "timestamp": datetime.now().isoformat(),
                    "api_endpoint": self.current_endpoint,
                    "error_details": error
                })

        # Example 2: An "error" field might be present when an error occurs
        elif isinstance(data, dict) and "error" in data:
            errors.append({
                "timestamp": datetime.now().isoformat(),
                "api_endpoint": self.current_endpoint,
                "error_details": data["error"]
            })
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "status" in item and item["status"] != "success":
                    errors.append({
                        "timestamp": datetime.now().isoformat(),
                        "api_endpoint": self.current_endpoint,
                        "error_details": item
                    })

        return errors

    def process_error(self, error_data):
        """
        Processes a single error by enriching the context and sending it to the RAG application.

        Args:
            error_data (dict): The dictionary containing error information.
        """
        enriched_error_data = {
            "timestamp": error_data.get("timestamp"),
            "api_endpoint": error_data.get("api_endpoint"),
            "error_message": str(error_data.get("error_details")), # Ensure it's a string
            # Add more contextual information here if available in your API response
        }
        print(f"Found error: {enriched_error_data}")
        self.send_to_rag(enriched_error_data)

    def send_to_rag(self, error_info):
        """
        Sends the error information to your Ollama RAG application.

        Args:
            error_info (dict): The enriched error information.
        """
        rag_url = "YOUR_OLLAMA_RAG_ENDPOINT"  # Replace with your RAG API endpoint
        try:
            response = requests.post(rag_url, json={"query": f"Analyze this API error and suggest solutions: {error_info}"})
            response.raise_for_status()
            rag_response = response.json()
            print(f"RAG Response: {rag_response.get('result')}") # Adjust based on your RAG response structure
        except requests.exceptions.RequestException as e:
            print(f"Error sending error to RAG: {e}")
        except json.JSONDecodeError:
            print("Error decoding JSON from RAG response.")

    def monitor_endpoint(self, endpoint):
        """
        Monitors a specific API endpoint for errors in a loop.

        Args:
            endpoint (str): The relative path of the endpoint to monitor.
        """
        self.current_endpoint = endpoint
        while True:
            data = self.fetch_data(endpoint)
            if data:
                errors = self.check_for_errors(data)
                for error in errors:
                    self.process_error(error)
            time.sleep(self.poll_interval)

    def run(self, endpoints_to_monitor):
        """
        Runs the error monitoring for the specified endpoints.

        Args:
            endpoints_to_monitor (list): A list of API endpoint paths to monitor.
        """
        for endpoint in endpoints_to_monitor:
            print(f"Monitoring endpoint: {self.api_url}/{endpoint}")
            self.monitor_endpoint(endpoint) # Consider running these in separate threads or asynchronously

if __name__ == "__main__":
    api_base_url = "https://api.example.com/v1"  # Replace with your API base URL
    endpoints = ["users", "products/errors", "orders"] # Replace with the endpoints you want to monitor
    polling_frequency = 30  # Check every 30 seconds
    api_auth = ('your_username', 'your_password') # Replace with your API credentials if needed
    custom_headers = {'X-API-Key': 'your_api_key'} # Add any custom headers

    error_reader = ApiErrorReader(
        api_base_url,
        poll_interval_seconds=polling_frequency,
        auth=api_auth,
        headers=custom_headers
    )
    error_reader.run(endpoints)