import pyautogui

import time
import subprocess
import argparse
from pathlib import Path

first_performance_button = (461, 325)
device_button = (358, 419)
next_button = (622, 537)
test_button = next_button
performance_button = (106, 737)

def remove_substring_from_end(s, suffix):
    """
    Removes the specified suffix from the end of a string if it exists.

    Parameters:
    s (str): The original string.
    suffix (str): The suffix to be removed from the end of the string.

    Returns:
    str: The string without the suffix at the end, or the original string if the suffix wasn't present.
    """
    if s.endswith(suffix):
        return s[:-len(suffix)]
    else:
        return s


def main(args):
    models = sorted(Path(args.path).rglob("*.mlmodel"))

    # Determine the locations of the buttons on the screen.
    try:
        while args.print_locations:
            print(pyautogui.position())
            time.sleep(args.duration)
    except KeyboardInterrupt:
        print("Program terminated.")

    for model in models:
        model = remove_substring_from_end(str(model), suffix="/Data/com.apple.CoreML/model.mlmodel")
        print(model)
        subprocess.run(["open", model])
        x, y = first_performance_button
        pyautogui.moveTo(x, y, duration=args.duration)
        pyautogui.click()

        for i in range(args.num_tests):
            for x, y in [performance_button, device_button, next_button, test_button]:
                pyautogui.moveTo(x, y, duration=args.duration)
                pyautogui.click()
            time.sleep(args.sleep_seconds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser( 'Automatic test latency with CoreML Tools.')
    parser.add_argument('--path', type=str, default='.', help='Path to the directory containing the models.')
    parser.add_argument('--duration', type=float, default=1, help='Duration of the mouse movement.')
    parser.add_argument('--sleep_seconds', type=float, default=10, help='Duration to wait between tests.')
    parser.add_argument('--num_tests', type=int, default=20, help='Number of tests to run.')
    parser.add_argument('--print_locations', action='store_true', help='Print the locations of the mouse.')
    args = parser.parse_args()
    main(args)