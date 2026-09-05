"""
Update notebook cells in 01_data_preprocessing.ipynb to include full 36 state mappings.
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
NB_DIR = ROOT_DIR / "notebooks"


def update_notebook_01():
    nb_path = NB_DIR / "01_data_preprocessing.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 11: state_mapping with Delhi and Andaman
    nb["cells"][11]["source"] = [
        "state_mapping = {\n",
        "    'MP': 'Madhya Pradesh',\n",
        "    'UP': 'Uttar Pradesh',\n",
        "    'HP': 'Himachal Pradesh',\n",
        "    'Pondy': 'Puducherry',\n",
        "    'J&K(UT) & Ladakh(UT)': 'Jammu and Kashmir',\n",
        "    'J&K(UT) & \\nLadakh(UT)': 'Jammu and Kashmir',\n",
        "    'J&K(UT) and Ladakh(UT)': 'Jammu and Kashmir',\n",
        "    'J&K': 'Jammu and Kashmir',\n",
        "    'J&K (UT)': 'Jammu and Kashmir',\n",
        "    'J&K(UT) &': 'Jammu and Kashmir',\n",
        "    'Jammu & Kashmir': 'Jammu and Kashmir',\n",
        "    'DD': 'Daman and Diu',\n",
        "    'DNH': 'Dadra and Nagar Haveli',\n",
        "    'DNHDDPDCL': 'Dadra and Nagar Haveli',\n",
        "    'Daman & Diu': 'Daman and Diu',\n",
        "    'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli',\n",
        "    'Delhi': 'NCT of Delhi',\n",
        "    'Andaman and Nicobar Islands': 'Andaman and Nicobar'\n",
        "}\n"
    ]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("✓ Updated state_mapping in 01_data_preprocessing.ipynb to cover all 36 states")


if __name__ == "__main__":
    update_notebook_01()
