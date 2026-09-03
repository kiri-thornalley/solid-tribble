# From Materials Project
from mp_api.client import MPRester
import torch
import random
import pandas as pd
from dotenv import load_dotenv 

def materials_pipeline(seed):
    """ Pulls a random, finite set of materials from the Materials Project.
    Parameter(s):
        None
    Returns:
        
    """
    dtype = torch.double
    random.seed(seed)
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
    candidate_df = candidate_df.dropna().reset_index(drop=True)
    #test:train split
    input_df = candidate_df.sample(n=200, random_state=seed)

    init_df = input_df.iloc[:10]
    # drop ID, drop target variable
    init_X = init_df.drop(columns=["material_id", "formation_energy"])
    init_X = torch.tensor(init_X.to_numpy(), dtype=dtype)
    # Calculate scaling parameters from initial training set 
    mean = init_X.mean(dim=0) 
    std = init_X.std(dim=0).clamp_min(1e-8)
    # Standardise initial data 
    init_X = (init_X - mean) / std

    init_Y = init_df["formation_energy"]
    init_Y = -torch.tensor(init_Y.to_numpy(), dtype=dtype).unsqueeze(-1) 
    # .unsqueeze(-1), otherwise BoTorch sulks about it, something something explit output dimension

    candidate_pool_df = input_df.iloc[10:150]
    candidate_X = candidate_pool_df.drop(
        columns=["material_id", "formation_energy"]
    )
    candidate_X = torch.tensor(
        candidate_X.to_numpy(),
        dtype=dtype
    )
    candidate_X = (candidate_X - mean) / std
    #held_out_test_df = input_df.iloc[150:200]

    print("Initial best formation energy:",
            init_df["formation_energy"].min())
    print(
    "Best candidate pool material:",
    candidate_pool_df["formation_energy"].min()
)
    return init_X, init_Y, candidate_X, candidate_pool_df
if __name__=="__main__":
    materials_pipeline()