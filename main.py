# main.py
from app import create_app # Import the application factory function

if __name__ == '__main__':
    app = create_app() # Call the factory function to create the app instance
    app.run(debug=True, host='0.0.0.0', port=5000)
