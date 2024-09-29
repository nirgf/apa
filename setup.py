# setup.py
import os
import subprocess
import sys

venv_dir = 'venv_apa'


def create_virtualenv():
    # Create virtual environment
    if not os.path.exists(venv_dir):
        subprocess.check_call([sys.executable, '-m', 'venv', venv_dir])
        print(f"Virtual environment created at {venv_dir}")
    else:
        print("Virtual environment already exists.")


def install_requirements():
    # Activate virtual environment and install requirements
    if sys.platform == "win32":
        activate_script = os.path.join(venv_dir, 'Scripts', 'activate')
    else:
        activate_script = f"chmod +x {venv_dir}/bin/activate;" + os.path.join(venv_dir, 'bin', 'activate')


    subprocess.call([activate_script], shell=True)

    subprocess.check_call([os.path.join(venv_dir, 'bin', 'pip'), 'install', '-r', 'requirements.txt'])
    print("Requirements installed.")


if __name__ == "__main__":
    create_virtualenv()
    install_requirements()
