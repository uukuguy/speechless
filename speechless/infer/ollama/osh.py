'''
Copyright (c) 2024 Yannis Charalambidis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
# https://github.com/charyan/osh/blob/master/osh.py
import requests
import json
import argparse
import sys
import os

import platform
import subprocess

import time
from multiprocessing import Process

# Ollama
ollama_server_url = "http://localhost:11434/api/generate"
# ollama_model_name = "mistral"
ollama_model_name = "uukuguy/mistral:7b-instruct-v0.2.Q8_0"
ollama_prompt_cmd = """
Write shell commands for a unix-like shell without comments. Do not write comments. Be concise. Do not write alternative suggestions. Adapt the command to the system information.
"""
ollama_prompt_explanation = """
Explain what the shell command and its parameters do. Be concise.
"""


class Colors:
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"


class Spinner():
    """
    A class representing a spinner animation.

    Attributes:
        _instance (Spinner): The singleton instance of the Spinner class.
        _color (Colors): The color of the spinner.
        _counter (int): The current counter for the spinner animation.
        _text (str): The text to be displayed alongside the spinner.
        _symbols (list[str]): The list of symbols used for the spinner animation.
        _break (bool): A flag indicating whether the spinner animation should stop.
        _process (Process): The process running the spinner animation.

    Methods:
        get(): Returns the singleton instance of the Spinner class.
        text(text: str): Sets the text to be displayed alongside the spinner.
        color(color: Colors): Sets the color of the spinner.
        _clear(): Clears the current line in the console.
        start(): Starts the spinner animation.
        stop(): Stops the spinner animation.
    """
    _instance = None
    _color: Colors = None
    _counter: int = 0
    _text: str = ""
    _symbols: list[str] = [x for x in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"]
    _break: bool = False
    _process: Process = None

    def get():
        """
        Returns the instance of the Spinner class.
        If the instance doesn't exist, it creates a new one and returns it.
        """
        if not Spinner._instance:
            Spinner._instance = Spinner()

        return Spinner._instance

    def text(self, text: str):
        """
        Set the text for the spinner.

        Args:
            text (str): The text to be set.
        """
        self._text = text

    def color(self, color: Colors):
        """
        Set the color of the object.

        Args:
            color (Colors): The color to set.
        """
        self._color = color

    def _clear(self):
        print("\r", end="")
        for _ in range(len(self._text) + 2):  # + braille char and space
            print(" ", end="")

    def start(self):
        """
        Start the animation process.
        """
        def _start():
            while True:
                self._clear()
                print(
                    f"\r{self._color}{self._symbols[self._counter]}{Colors.ENDC} {self._text}  ", end="")
                self._counter = (self._counter + 1) % len(self._symbols)
                time.sleep(0.1)

        # self._process = Process(target=_start)
        import dill
        self._process = Process(target=dill.dumps(_start), args=())

        self._process.start()

    def stop(self):
        """
        Stops the process and clears the output.

        This method clears the output, terminates the running process, and moves the cursor to the beginning of the line.
        """
        self._clear()
        print("\r", end="")
        self._process.kill()


def get_linux_distribution():
    """
    Retrieves the Linux distribution name.

    This function tries to determine the Linux distribution name by checking various sources,
    such as the '/etc/os-release' file, the 'lsb_release' command, and the '/etc/issue' file.
    If the distribution name cannot be determined, it returns 'Unknown'.

    Returns:
        str: The Linux distribution name.
    """
    try:
        with open("/etc/os-release") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('NAME='):
                    return line.split('=')[1].strip().strip('"')
    except IOError:
        pass

    try:
        return subprocess.check_output(["lsb_release", "-d"]).split(b":")[1].strip().decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        with open("/etc/issue") as f:
            return f.readline().strip()
    except IOError:
        pass

    return "Unknown"


def get_system_info():
    """
    Get system information including OS type and distribution.

    Returns:
        str: System information.
    """
    system_info = subprocess.check_output(["uname", "-a"]).decode().strip()
    result = f"System Information: {system_info}\n"

    os_type = platform.system()

    if os_type == "Linux":
        result += f"Distribution: {get_linux_distribution()}"
    elif os_type == "Darwin":
        result += f"OS: macOS {platform.mac_ver()[0]}"
    elif os_type in ["FreeBSD", "NetBSD", "OpenBSD"]:
        result += f"OS: {os_type} {platform.release()}"
    else:
        result += "OS: Unknown"

    return result


def generate_cmd(prompt: str):
    """
    Generate a shell command using the Ollama API.

    Args:
        prompt (str): The prompt to generate the command.

    Returns:
        str: The generated shell command.

    MIT License
    Source: https://github.com/jmorganca/ollama/blob/main/api/client.py
    """

    spinner = Spinner.get()
    spinner.color(Colors.CYAN)
    spinner.text("Generating shell command")
    spinner.start()

    try:

        payload = {
            "model": ollama_model_name,
            "prompt": f"{ollama_prompt_cmd}\n\n{get_system_info()}\n\nUser request:{prompt}",
            "system": None,
            "template": None,
            "context": None,
            "options": None,
            "format": None,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        with requests.post(ollama_server_url, json=payload, stream=True) as response:
            response.raise_for_status()

            # Variable to hold concatenated response strings if no callback is provided
            full_response = ""

            # Iterating over the response line by line and displaying the details
            for line in response.iter_lines():
                if line:
                    # Parsing each line (JSON chunk) and extracting the details
                    chunk = json.loads(line)

                    # If this is not the last chunk, add the "response" field value to full_response and print it
                    if not chunk.get("done"):
                        response_piece = chunk.get("response", "")
                        full_response += response_piece

            fs = full_response.splitlines()

            # Remove markdown code blocks
            response = "\n".join([s for s in fs if "```" not in s])

            # Remove backticks
            response = "\n".join(
                [e for e in response.split("`") if e.strip() != ""])

            response = response.strip()

            spinner.stop()

            if response == "":
                print("Error parsing response")
                exit(-1)

            return response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        spinner.stop()
        exit(1)


def generate_explanation(cmd: str):
    """
    Generate an explanation a shell command using the Ollama API and prints it.

    Args:
        cmd (str): The cmd to generate the command.

    MIT License
    Source: https://github.com/jmorganca/ollama/blob/main/api/client.py
    """

    spinner = Spinner.get()
    spinner.color(Colors.YELLOW)
    spinner.text("Generating explanation")
    spinner.start()

    try:

        payload = {
            "model": ollama_model_name,
            "prompt": f"{ollama_prompt_explanation}\n\n{get_system_info()}\n\nCommand:{cmd}",
            "system": None,
            "template": None,
            "context": None,
            "options": None,
            "format": None,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        with requests.post(ollama_server_url, json=payload, stream=True) as response:
            spinner.stop()
            print("\r", end="")
            print()

            response.raise_for_status()

            # Variable to hold concatenated response strings if no callback is provided
            full_response = ""

            # Iterating over the response line by line and displaying the details
            for line in response.iter_lines():
                if line:
                    # Parsing each line (JSON chunk) and extracting the details
                    chunk = json.loads(line)

                    # If this is not the last chunk, add the "response" field value to full_response and print it
                    if not chunk.get("done"):
                        response_piece = chunk.get("response", "")
                        full_response += response_piece
                        print(response_piece, end="", flush=True)

            if response == "":
                print("Error parsing response")
                exit(-1)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        spinner.stop()
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get a command for a unix-like shell from a model running with Ollama and execute it")
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Execute the command without asking for confirmation')
    parser.add_argument('-s', '--system-info', action='store_true',
                        help='Display system info')
    parser.add_argument('prompt', metavar='PROMPT',
                        type=str, nargs='?', default=sys.stdin)

    args = parser.parse_args()
    yes = args.yes
    prompt = args.prompt

    if args.system_info is True:
        print(get_system_info())
        exit(0)

    try:
        cmd = generate_cmd(prompt)
    except KeyboardInterrupt:
        print("\n\033[91m" + "Interrupted" + "\033[0m")
        exit(0)

    print("\033[91m" + cmd + "\033[0m", end="")

    if not yes:
        print("\033[92m\t Execute ? (y/N/[e]xplain)\033[0m ", end="")

    answer = input().lower()

    if answer == "explain" or answer == "e":
        generate_explanation(cmd)
        print()

        print("\033[91m" + cmd + "\033[0m", end="")
        print("\033[92m\t Execute ? (y/N)\033[0m ", end="")
        answer = input().lower()

    if yes or answer == "y":
        os.system(cmd)
