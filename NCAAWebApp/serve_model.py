import sys
import os

# Ensure Flask finds templates regardless of bundling
base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
template_folder = os.path.join(base_path, "NCAAWebApp", "templates")

from flask import Flask
import NCAAWebApp.app as app_module

app_module.app.template_folder = template_folder  # overwrite default

if __name__ == "__main__":
    app_module.app.run(debug=True, port=5000)
