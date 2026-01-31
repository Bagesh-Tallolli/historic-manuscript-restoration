#!/usr/bin/env python3
"""
Convert kaggle_complete_training.py to a complete Jupyter notebook
"""
import json
import re

def py_to_notebook(py_file, notebook_file):
    """Convert Python file with cell markers to Jupyter notebook"""

    with open(py_file, 'r') as f:
        content = f.read()

    # Split by CELL markers
    cell_pattern = r'# CELL \d+:(.*?)(?=# CELL \d+:|$)'
    matches = list(re.finditer(cell_pattern, content, re.DOTALL))

    cells = []

    for match in matches:
        cell_content = match.group(1).strip()

        # Check if it's markdown or code
        if cell_content.startswith('# %%'):
            # Extract markdown
            lines = cell_content.split('\n')
            if lines[0].strip() == '# %%':
                # Code cell
                code_lines = lines[1:]
                code = '\n'.join(code_lines)
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [code + '\n']
                })
        else:
            # Parse markdown and code cells
            parts = cell_content.split('# %%')
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Check if starts with markdown marker
                if part.startswith('# md\n') or part.startswith('#%% md\n'):
                    # Markdown cell
                    md_lines = part.split('\n')[1:]  # Skip marker
                    md_text = '\n'.join(line[2:] if line.startswith('# ') else line for line in md_lines)
                    cells.append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [md_text.strip() + '\n']
                    })
                else:
                    # Code cell
                    cells.append({
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [part + '\n']
                    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(notebook_file, 'w') as f:
        json.dump(notebook, f, indent=1)

    print(f"✓ Created notebook with {len(cells)} cells")
    print(f"  Output: {notebook_file}")

if __name__ == '__main__':
    py_to_notebook(
        'kaggle_complete_training.py',
        'kaggle_training_notebook_complete.ipynb'
    )

