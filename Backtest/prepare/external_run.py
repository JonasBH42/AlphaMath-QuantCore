import subprocess
import os


path = os.path.dirname(os.path.abspath(__file__))

def run_external_program(input_values):
    # Prepare the input string (each input on a new line)
    input_str = "\n".join(input_values) + "\n"

    # Full path to the external program.
    # Using a raw string (r"") helps avoid issues with backslashes.
    program_path = os.path.join(path, "SVR_new.py")

    # Execute the external program.
    result = subprocess.run(
        ["python", program_path],  # Use "python3" if needed in your environment.
        input=input_str,
        text=True,  # Handle input/output as text.
        capture_output=True,  # Capture stdout and stderr.
    )

    # Check for errors
    if result.returncode != 0:
        print("Error running the external program:")
        print(result.stderr)

    return result.stdout


text_path = os.path.join(path, "results.txt")

if __name__ == "__main__":
    # First set of inputs (adjust these as needed)
    print("start")
    # inputs_set1 = ["9", "14", "10"]
    # output1 = run_external_program(inputs_set1)
    # print(output1)
    # with open(text_path, "w") as file:
    #     file.write(output1)

    inputs_set2 = ["8", "13", "10"]
    output2 = run_external_program(inputs_set2)
    print(output2)
    with open(text_path, "a") as file:
        file.write(f"\n\n{output2}")
    
    # inputs_set3 = ["7", "12", "10"]
    # output3 = run_external_program(inputs_set3)
    # print(output3)
    # with open(text_path, "a") as file:
    #     file.write(f"\n\n{output3}")
    
    inputs_set5 = ["4", "8", "10"]
    output5 = run_external_program(inputs_set5)
    print(output5)
    with open(text_path, "a") as file:
        file.write(f"\n\n{output5}")

    inputs_set6 = ["28", "35", "10"]
    output6 = run_external_program(inputs_set6)
    print(output6)
    with open(text_path, "a") as file:
        file.write(f"\n\n{output6}")
    
    inputs_set7 = ["15", "28", "10"]
    output7 = run_external_program(inputs_set7)
    print(output7)
    with open(text_path, "a") as file:
        file.write(f"\n\n{output7}")
    