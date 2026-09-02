# From Materials Project
from mp_api.client import MPRester
import torch
import random
import pandas as pd
import os
from dotenv import load_dotenv 

def materials_pipeline():
    """ Pulls a random, finite set of materials from the Materials Project.
    Parameter(s):
        None
    Returns:
        
    """

    random_page = random.randint(0, 150)

    load_dotenv(dotenv_path="/home/kiri/materials_project/API_keys.env")
    with MPRester() as mpr:
        materials = mpr.materials.summary.search(
            _page=random_page,
            chunk_size=500,
            num_chunks=1,
            fields=["material_id","formation_energy_per_atom", "band_gap", "volume", "density", "nelements", "efermi"],
        )

    candidate_df = pd.DataFrame([
        {
            "material_id": m.material_id,
            "band_gap": m.band_gap,
            "volume": m.volume,
            "density": m.density,
            "nelements": m.nelements,
            "efermi": m.efermi,
            "formation_energy": m.formation_energy_per_atom,
        }
        for m in materials
    ])

    #test:train split
    input_df = candidate_df.sample(n=200, random_state=42)

    init_df = input_df.iloc[:100]
    # drop ID, drop target variable
    init_X = init_df.drop(columns=["material_id", "formation_energy"])
    init_X = torch.tensor(init_X.to_numpy(), dtype=torch.float32)
    init_Y = init_df["formation_energy"]
    init_Y = torch.tensor(init_Y.to_numpy(), dtype=torch.float32)

    candidate_pool_df = input_df.iloc[100:150]
    held_out_test_df = input_df.iloc[150:200]

    return init_X, init_Y
if __name__=="__main__":
    materials_pipeline()