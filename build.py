import os
import requests
import tarfile


def download_file(save_path: str, file_url: str):
    response = requests.get(file_url)

    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Tensorflow downloaded successfully to: {save_path}")
    else:
        print("Failed to download Tensorflow")

def unpack_file(file_path: str):
    try:
        tar = tarfile.open(file_path, "r:gz")
    except:
        mode = 'r'
        tar = tarfile.open(file_path, mode)
    tar.extractall()
    tar.close()    
        
        
def prepare_tensorflow():
    file_url = "https://doubango.org/deep_learning/libtensorflow_r1.14_cpu+gpu_linux_x86-64.tar.gz"
    save_directory = '../../../binaries/linux/x86_64/'
    save_path = os.path.join(save_directory, os.path.basename(file_url))
    
    download_file(save_path, file_url)
    unpack_file(save_path)
    

def set_environment_paths():
    python_path = os.pathsep.join([
        os.path.abspath("../../../binaries/linux/x86_64"),
        os.path.abspath("../../../python")
    ])
    ld_library_path = f"../../../binaries/linux/x86_64:{os.environ.get('LD_LIBRARY_PATH', '')}"

    os.environ['PYTHONPATH'] = python_path
    os.environ['LD_LIBRARY_PATH'] = ld_library_path
    

def prepare_everything():
    prepare_tensorflow()
    set_environment_paths()


if __name__ == "__main__":
    prepare_everything()
