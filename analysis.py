import json
import numpy as np
from colorama import Fore, Back, Style

# Function to calculate mean values and print differences
def print_mean_differences(original, modified, label):
    original_mean = np.mean(np.array(original), axis=0)
    modified_mean = np.mean(np.array(modified), axis=0)
    differences = np.subtract(modified_mean, original_mean)
    
    print(Fore.BLUE + f"Mean differences for {label} (Original vs. Modified):" + Style.RESET_ALL)
    print("-" * 50)
    for i in range(5):
        for j in range(5):
            index = i * 5 + j
            diff = differences[index]
            color = Fore.RED if diff < 0 else Fore.GREEN
            print(color + f"{diff:+.2f}" + Style.RESET_ALL, end='   ')
        print()
    print("-" * 50)
    overall_original_mean = np.mean(original_mean)
    overall_modified_mean = np.mean(modified_mean)
    print(Fore.BLUE + f"Overall Mean - Original: {overall_original_mean:.2f}, Modified: {overall_modified_mean:.2f}" + Style.RESET_ALL)
    print()  # Extra newline for spacing

# Load JSON data
with open('tables.json', 'r') as f:
    tables = json.load(f)

with open('delltables.json', 'r') as f:
    delltables = json.load(f)

# Initialize lists
knightstables, bishopstables, queentables, kingstables = [], [], [], []
dell_knightstables, dell_bishopstables, dell_queentables, dell_kingstables = [], [], [], []
knightsweights, bishopsweights, queensweights, kingsweights = [], [], [], []
dell_knightsweights, dell_bishopsweights, dell_queensweights, dell_kingsweights = [], [], [], []

# Populate lists
for obj in tables:
    knightstables.append(obj['knightstable'])
    bishopstables.append(obj['bishopstable'])
    queentables.append(obj['queenstable'])
    kingstables.append(obj['kingstable'])
    knightsweights.append(obj['knightweight'])
    bishopsweights.append(obj['bishopweight'])
    queensweights.append(obj['queenweight'])
    kingsweights.append(obj['kingweight'])

for obj in delltables:
    dell_knightstables.append(obj['knightstable'])
    dell_bishopstables.append(obj['bishopstable'])
    dell_queentables.append(obj['queenstable'])
    dell_kingstables.append(obj['kingstable'])
    dell_knightsweights.append(obj['knightweight'])
    dell_bishopsweights.append(obj['bishopweight'])
    dell_queensweights.append(obj['queenweight'])
    dell_kingsweights.append(obj['kingweight'])

# Print mean differences
print_mean_differences(knightstables, dell_knightstables, "knightstables")
print_mean_differences(bishopstables, dell_bishopstables, "bishopstables")
print_mean_differences(queentables, dell_queentables, "queentables")
print_mean_differences(kingstables, dell_kingstables, "kingstables")

# For weights, since they are single values per object, we handle them slightly differently
def print_weight_differences(original, modified, label):
    original_mean = np.mean(original)
    modified_mean = np.mean(modified)
    difference = modified_mean - original_mean
    color = Fore.RED if difference < 0 else Fore.GREEN
    print(Fore.BLUE + f"Mean difference for {label} weight:" + Style.RESET_ALL)
    print(color + f"{difference:+.2f}" + Style.RESET_ALL)
    print(Fore.BLUE + f"Overall Mean - Original: {original_mean:.2f}, Modified: {modified_mean:.2f}" + Style.RESET_ALL)
    print()  # Extra newline for spacing
    
print_weight_differences(knightsweights, dell_knightsweights, "Knight")
print_weight_differences(bishopsweights, dell_bishopsweights, "Bishop")
print_weight_differences(queensweights, dell_queensweights, "Queen")
print_weight_differences(kingsweights, dell_kingsweights, "King")