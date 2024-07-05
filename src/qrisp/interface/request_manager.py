import requests
from requests.auth import HTTPBasicAuth


class RequestManager:
    def __init__(self, base_url, username=None, password=None):
        self.base_url = base_url
        self.session = requests.Session()
        if username is not None and password is not None:
            self.set_credentials(username, password)

    def set_credentials(self, username, password):
        """Set HTTP basic authentication credentials."""
        self.session.auth = HTTPBasicAuth(username, password)

    def get(self, endpoint, **kwargs):
        """Execute GET request. """
        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url, **kwargs)
        return response

    def post(self, endpoint, data=None, json=None, **kwargs):
        url = f"{self.base_url}{endpoint}"
        response = self.session.post(url, data=data, json=json, **kwargs)
        return response

    def delete(self, endpoint, data=None, json=None, **kwargs):
        url = f"{self.base_url}{endpoint}"
        response = self.session.post(url, data=data, json=json, **kwargs)
        return response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the session."""
        self.session.close()
