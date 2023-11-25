# Create an environment and install the required packages
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Build the project
cd binaries/linux/x86_64
python3 ../../../python/setup.py build_ext --inplace -v
cd ../../../samples/python/recognizer
# The build.py script will download tensorflow and set up needed paths
python3 ../../../build.py
