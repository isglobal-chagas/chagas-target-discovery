#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:38:53 2024

@author: isglobal
"""
from config import *

# Standard library imports
import argparse
import json
import pickle
import re
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import ast
import os
from os.path import expanduser
import math
import warnings
import psutil
warnings.filterwarnings("ignore", category=UserWarning, module="rdkit")

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from tqdm import tqdm

# Biopython imports
from Bio import PDB
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

# Biotite
import biotite.database.rcsb as rcsb

# RDKit imports
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdFMCS, rdDepictor, MACCSkeys, rdReducedGraphs
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem.Scaffolds import MurckoScaffold

# RDKit setup
rdDepictor.SetPreferCoordGen(True)


def parse_args():
    parser = argparse.ArgumentParser(description='NTDetect — target deconvolution pipeline')
    parser.add_argument('--smiles',
                        type=str,
                        required=True,
                        help='SMILES string of the reference compound')
    parser.add_argument('--compound',
                        type=str,
                        required=True,
                        help='Name of the reference compound')
    return parser.parse_args()


def calculate_centroid(atoms_coords):
    total_x = total_y = total_z = 0.0
    atom_count = len(atoms_coords)

    if atom_count == 0:
        return None  # Return None if no atoms are provided

    for atom_coord in atoms_coords:
        x, y, z = atom_coord
        total_x += x
        total_y += y
        total_z += z

    # Calculate the average for each coordinate
    center_x = np.round(total_x / atom_count, 3)
    center_y = np.round(total_y / atom_count, 3)
    center_z = np.round(total_z / atom_count, 3)

    return center_x, center_y, center_z


def get_plddts(pdb_structure):
    plddt_values = {}

    for model in pdb_structure:
        for chain in model:
            for residue in chain:
                residue_id = residue.get_id()[1]
                for atom in residue:
                    if atom.get_name() == 'CA':  # Only consider C-alpha atoms for pLDDT
                        plddt_values[residue_id] = atom.bfactor
                        break  # No need to check other atoms in the residue
    return plddt_values


def get_pae(af_file):
    pdb_prefix = af_file.stem.rsplit("-", 1)[0]
    error_fname = f"{pdb_prefix}-predicted_aligned_error_v6.json"
    error_url = f"https://alphafold.ebi.ac.uk/files/{error_fname}"

    try:
        response = requests.get(error_url)
        error_json = response.text
    except Exception as e:
        print(e)
        time.sleep(10)
        response = requests.get(error_url)
        error_json = response.text        
    
    if error_json:
        return error_json
    else:
        return None


def calculate_pae(residue_ids, pae_matrix):
    residue_ids = set(residue_ids)  
    pae_means = {}
    for base_residue, pae_values in enumerate(pae_matrix, 1):
        
        if base_residue not in residue_ids:
            continue
    
        aligned_pae = [pae_value
                       for ix, pae_value in enumerate(pae_values, 1)
                       if ix in residue_ids and ix != base_residue
                       ]
        residue_pae_mean = np.array(aligned_pae).mean(dtype=np.float64)
        pae_means[base_residue] = residue_pae_mean

    return pae_means


def parse_pqr_file(file_path):
    pocket_data = {}
    atoms = []

    # Regular expressions for header and atom lines
    header_regex = re.compile(r'HEADER\s+(\d+)\s*-\s*(.*?):\s*(.+)')

    with open(file_path, 'r') as file:
        for line in file:
            
            if line.startswith('HEADER'):
                # Match header information
                header_match = header_regex.match(line)
                if header_match:
                    key = header_match.group(2).strip()
                    value = float(header_match.group(3).strip())
                    pocket_data[key] = value
            
            elif line.startswith('ATOM'):
                atom_number = int(line[6:11].strip())
                atom_type = line[12:16].strip()
                residue_name = line[17:20].strip()
                residue_id = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                charge = float(line[54:60].strip())
                radius = float(line[60:66].strip())                
                    
                atoms.append({
                    'atom_number': atom_number,
                    'atom_type': atom_type,
                    'residue_name': residue_name,
                    'residue_id': residue_id,
                    'coord': (x, y, z),
                    'charge': charge,
                    'radius': radius
                })
    if not atoms:
        print(f"No voronoi vertices found in {file_path.stem}!")
        return pocket_data, None
    return pocket_data, atoms


def predict_pockets(pdb_file, fpocket_path, p2rank_path):

    # Clean up any existing output for this file
    fpocket_folder = pdb_file.parent / f"{pdb_file.stem}_out"
    if fpocket_folder.exists():
        shutil.rmtree(fpocket_folder)
    
    # Run fpocket
    cmd = [fpocket_path, '-f', str(pdb_file)]
    subprocess.run(cmd, env=ENV, check=True)
    
    # Create ds file for P2Rank
    prediction_file = fpocket_folder / f"{fpocket_folder.name}.pdb"
    fpocket_list_file = Path(f'{pdb_file.stem}_fpocket_hit.ds') # needs to be in current dir...
    with open(fpocket_list_file, 'w+') as f:
        f.write("PARAM.PREDICTION_METHOD=fpocket\n\n")
        f.write("HEADER: prediction protein\n\n")
        f.write(f"{prediction_file}  {pdb_file}\n")

    # Clean up any existing p2rank folder
    p2rank_output = pdb_file.parent / f"{pdb_file.stem}_p2rank"
    for f in p2rank_output.glob('./*'):
        if f.is_file():
            f.unlink()
        else:
            shutil.rmtree(f)
    p2rank_output.mkdir(parents=True, exist_ok=True)

    # Run P2Rank
    cmd = [p2rank_path, 'rescore', fpocket_list_file, '-c', 'alphafold', '-o', p2rank_output]
    subprocess.run(cmd, env=ENV, check=True)
    fpocket_list_file.unlink()
    p2rank_result = p2rank_output / f'{prediction_file.name}_rescored.csv'
    p2rank_df = pd.read_csv(p2rank_result, index_col=0)
    p2rank_dict = p2rank_df.to_dict('index')

    # Get pLDDT values from the PDB file
    try:
        pdb_parser = PDBParser(QUIET=True)
        full_structure = pdb_parser.get_structure('whole', pdb_file)
        plddts = get_plddts(full_structure)
    except Exception as e:
        print(f"Error reading PDB file {pdb_file}: {e}")
        return None

    # Get the PAE matrix
    pae_data = get_pae(pdb_file)
    pae_data = json.loads(pae_data)[0]
    pae_matrix = np.array(pae_data["predicted_aligned_error"])

    # Iterate through all pockets
    results = []
    pockets_dir = fpocket_folder / 'pockets'
    for pqr_file in pockets_dir.glob('*.pqr'):

        pocket_num = pqr_file.stem.split('_')[0]
        pocket_res_file = pockets_dir / f'{pocket_num}_atm.pdb'

        # Get pocket information
        pocket_info, points = parse_pqr_file(pqr_file)
        if points is None:
            print(f"Error reading Fpocket file {pqr_file} skipping...")
            continue  # Skip if no points found

        points_coords = [p['coord'] for p in points]
        p_x, p_y, p_z = calculate_centroid(points_coords)
        
        # Get nearby residues
        try:
            pocket_resis = [r for r in pdb_parser.get_structure('pocket_resis', pocket_res_file).get_residues()]
            resis_ids = [r.id[1] for r in pocket_resis]
        except Exception as e:
            print(f"Error reading pocket residues for {pdb_file.stem}, skipping pocket {pocket_num}: {e}")
            continue

        # Calculate pLDDT and PAE
        pocket_plddts = [plddts.get(resid, 0) for resid in resis_ids]  # Default to 0 if residue not found
        mean_plddt = np.mean(pocket_plddts)
        median_plddt = np.median(pocket_plddts)

        pocket_paes = calculate_pae(resis_ids, pae_matrix)
        mean_pae = np.mean(list(pocket_paes.values()))
        median_pae = np.median(list(pocket_paes.values()))
    
        # Get P2Rank scores
        p2rank_key = f'pocket.{pocket_num.split("pocket")[1]}'
        if p2rank_key not in p2rank_dict:
            print(f"Error: P2Rank key not found for pocket {pocket_num} in {pdb_file.stem}, skipping...")
            continue

        p2rank_rescore = p2rank_dict[p2rank_key].get('score', 0)
        p2rank_rank = p2rank_dict[p2rank_key].get('rank', 0)

        entry = {
            'protein_id': pdb_file.stem.split('-')[1],
            'pocket_id': pocket_num,
            'centroid_x': p_x,
            'centroid_y': p_y,
            'centroid_z': p_z,
            'fpocket_rank': int(pocket_num.split("pocket")[1]),
            'p2rank_rank': p2rank_rank,
            'mean_plddt': np.round(mean_plddt, 2),
            'median_plddt': np.round(median_plddt, 2),
            'mean_pae': np.round(mean_pae, 2),
            'median_pae': np.round(median_pae, 2),
            'p2rank_score': p2rank_rescore,
            'fpocket_score': float(pocket_info.get('Pocket Score', 0)),
            'fpocket_drug': float(pocket_info.get('Drug Score', 0)),
        }

        results.append(entry)    
    return results


def parse_tmalign_output(tmalign_output):
    # Define regular expressions for the required values
    rmsd_regex = r'RMSD=\s+([\d\.]+)'
    tm_score_chain1_regex = r'TM-score=\s+([\d\.]+) \(if normalized by length of Chain_1'
    tm_score_chain2_regex = r'TM-score=\s+([\d\.]+) \(if normalized by length of Chain_2'

    # Search for the patterns in the output
    rmsd_match = re.search(rmsd_regex, tmalign_output)
    tm_score_chain1_match = re.search(tm_score_chain1_regex, tmalign_output)
    tm_score_chain2_match = re.search(tm_score_chain2_regex, tmalign_output)
    
    # Extract the values from the matches
    rmsd = float(rmsd_match.group(1)) if rmsd_match else None
    tm_score_chain1 = float(tm_score_chain1_match.group(1)) if tm_score_chain1_match else None
    tm_score_chain2 = float(tm_score_chain2_match.group(1)) if tm_score_chain2_match else None

    return rmsd, tm_score_chain1, tm_score_chain2


def parse_us_align_output(output):
    result = []
    alignments = output.split('********************************************************************\n')
    
    for alignment in alignments:
        if 'Name of Structure_1' in alignment and 'Name of Structure_2' in alignment:
            data = {}
            structure_1_match = re.search(r'Name of Structure_1:\s+(.*):(\w+)', alignment)
            structure_2_match = re.search(r'Name of Structure_2:\s+(.*):(\w+)', alignment)
            length_1_match = re.search(r'Length of Structure_1:\s+(\d+)', alignment)
            length_2_match = re.search(r'Length of Structure_2:\s+(\d+)', alignment)
            aligned_length_match = re.search(r'Aligned length=\s+(\d+)', alignment)
            rmsd_match = re.search(r'RMSD=\s+([\d.]+)', alignment)
            seq_id_match = re.search(r'Seq_ID=n_identical/n_aligned=\s+([\d.]+)', alignment)
            tm_score_1_match = re.search(r'TM-score=\s+([\d.]+) \(normalized by length of Structure_1', alignment)
            tm_score_2_match = re.search(r'TM-score=\s+([\d.]+) \(normalized by length of Structure_2', alignment)
            alignment_pairs_match = re.search(r'\n\(":" denotes residue pairs of d < 5.0 Angstrom, "." denotes other aligned residues\)\n(.*)\n([A-Z:\.]+)\n(-+)', alignment, re.DOTALL)

            if structure_1_match:
                data['Structure_1'] = structure_1_match.group(1)
                data['Chain_1'] = structure_1_match.group(2)
            if structure_2_match:
                data['Structure_2'] = structure_2_match.group(1)
                data['Chain_2'] = structure_2_match.group(2)
            if length_1_match:
                data['Length_1'] = int(length_1_match.group(1))
            if length_2_match:
                data['Length_2'] = int(length_2_match.group(1))
            if aligned_length_match:
                data['Aligned_length'] = int(aligned_length_match.group(1))
            if rmsd_match:
                data['RMSD'] = float(rmsd_match.group(1))
            if seq_id_match:
                data['Seq_ID'] = float(seq_id_match.group(1))
            if tm_score_1_match:
                data['TM-score_1'] = float(tm_score_1_match.group(1))
            if tm_score_2_match:
                data['TM-score_2'] = float(tm_score_2_match.group(1))
            if alignment_pairs_match:
                data['Aligned_Residues_1'] = alignment_pairs_match.group(1).strip()
                data['Aligned_Residues_2'] = alignment_pairs_match.group(2).strip()

            if data:
                result.append(data)
    
    return result


def residues_near_ligand(structure,
                         ligand_resname,
                         distance_threshold):

    
    rna = {'A', 'T', 'C', 'G', 'U'}
    dna = {'DA', 'DT', 'DC', 'DG', 'DU'}
    
    models = list(structure)
    if len(models) != 1:
        print(f"Expected single model, found {len(models)}")
    model = models[0]
    
    # Get ligand atoms
    ligand_dict = {}
    for chain in model:
        for residue in chain:
            if residue.resname == ligand_resname:
                ligand_atoms = list(residue.get_atoms())
                ligand_coords = [a.get_coord() for a in ligand_atoms]
                ligand_centroid = np.array(calculate_centroid(ligand_coords))
                ligand_dict[chain] = (ligand_atoms, ligand_centroid)

    all_atoms = list(structure.get_atoms())
    if not all_atoms:
        return None

    # Find all atoms near the ligand centroid
    ns = PDB.NeighborSearch(all_atoms)
    nearby_residues = {}
    for ligand_chain, ligand_entry in ligand_dict.items():
        ligand_atom_list, ligand_centroid = ligand_entry
        nearby_residues[ligand_chain] = []
        done_residues = set()
        nearby_atoms = ns.search(ligand_centroid, distance_threshold)
        for atom in nearby_atoms:
            residue = atom.get_parent()
            resname = residue.get_resname().strip()
            
            if residue in done_residues:
                continue
            
            if residue.get_resname() == ligand_resname:
                continue
            
            hetatm, _, _ = residue.get_full_id()[-1]
            if hetatm != ' ':
                continue
            
            residue_coords = [a.get_coord() for a in residue.get_atoms()]
            residue_centroid = np.array(calculate_centroid(residue_coords))
            distance = np.linalg.norm(ligand_centroid - residue_centroid)
                           
            if is_aa(residue):  # normal case, residue is amino acid
               nearby_residues[ligand_chain].append((residue, distance, 'PROTEIN'))
            elif resname.upper() in rna:  # res is RNA
                nearby_residues[ligand_chain].append((residue, distance, 'RNA'))
            elif resname.upper() in dna:  # res is DNA
                nearby_residues[ligand_chain].append((residue, distance, 'DNA'))
            else:
                print(f"Unknown residue {residue} not added")

            done_residues.add(residue)
    
        nearby_residues[ligand_chain].sort(key=lambda x: x[1])


    # No residues found within distance
    if not nearby_residues:
        return None

    return nearby_residues

def residues_near_point(structure, point, distance_threshold):

    # Find nearby residues
    atoms = [atom for atom in structure.get_atoms() if is_aa(atom.get_parent())]
    ns = PDB.NeighborSearch(atoms)
    nearby_residues = defaultdict(list)
    
    done_residues = set()
    nearby_atoms = ns.search(point, distance_threshold)
    for atom in nearby_atoms:
        residue = atom.get_parent()
        
        if residue in done_residues:
            continue

        if is_aa(residue):
            residue_coords = [a.get_coord() for a in residue.get_atoms()]
            residue_centroid = np.array(calculate_centroid(residue_coords))
            residue_chain = residue.get_full_id()[2]
            distance = np.linalg.norm(point - residue_centroid)
            nearby_residues[residue_chain].append((residue, distance, 'PROTEIN'))
            done_residues.add(residue)
    
    # No residues found within distance
    if not nearby_residues:
        return None
    
    return dict(nearby_residues)


def get_chains_file(structure,
                    nearby_residues,
                    output_file):
      
    # Filter chains by residues
    chains_to_keep = set([residue.get_parent().id
                          for ligand_chain, residue_list in nearby_residues.items()
                          for residue, distance, chain_type in residue_list
                          if chain_type == 'PROTEIN'
                          ]
                         )
    if not chains_to_keep:
        return None
    
    filtered_structure = []
    for model in structure:
        for chain in model:
            if chain.id in chains_to_keep:
                filtered_structure.append(chain)
    
    # Initialize MMCIFIO object
    mmcif_io = PDB.MMCIFIO()
    
    # Set structure to be saved
    mmcif_io.set_structure(structure)
    
    # Define custom selection class
    class ChainSelect(PDB.Select):
        def accept_chain(self, chain):
            return chain in filtered_structure
        
    # Save structure with custom selection
    mmcif_io.save(output_file, ChainSelect())
    
    del mmcif_io
    
    return output_file


def get_pocket_file(structure, nearby_residues, output_file):
    
    residue_ids = set([(residue.get_parent().id, str(residue.get_id()[1]))
                       for ligand_chain, residue_list in nearby_residues.items()
                       for residue, distance, chain_type in residue_list
                       if chain_type == 'PROTEIN'
                       ]
                      )
    
    # Create a new structure for the pocket
    pocket_structure = PDB.Structure.Structure('pocket')

    for model in structure:
        pocket_model = PDB.Model.Model(model.id)
        for chain in model:
            pocket_chain = PDB.Chain.Chain(chain.id)
            for residue in chain:
                res_id = str(residue.get_id()[1])
                if (chain.id, res_id) in residue_ids:
                    pocket_chain.add(residue.copy())
            if len(pocket_chain):
                pocket_model.add(pocket_chain)
        if len(pocket_model):
            pocket_structure.add(pocket_model)
    
    # Initialize MMCIFIO object
    mmcif_io = PDB.MMCIFIO()
    
    # Save the new structure
    mmcif_io.set_structure(pocket_structure)
    mmcif_io.save(str(output_file))
    
    del mmcif_io
    
    return output_file


def process_pdb_hit(chem_hit_id,
                    pdb_hit_id,
                    ligand_distance=10
                    ):

    # Fetch the PDB file corresponding to the hit ID
    print(f"\nFetching PDB {pdb_hit_id}...", end='')
    pdb_hit_file = rcsb.fetch(pdb_hit_id, "cif", 'tmp')
    # Load structure
    parser = PDB.MMCIFParser(QUIET=True)
    pdb_structure = parser.get_structure('structure', pdb_hit_file)
    pdb_name = pdb_structure.header['head'] + ' | ' + pdb_structure.header['name']
    print("done!")
    
    # Get residues near ligand. This is to help select which chains to align
    print(f"\nFinding residues near ligand {chem_hit_id}...", end='')
    nearby_residues = residues_near_ligand(structure=pdb_structure,
                                           ligand_resname=chem_hit_id,
                                           distance_threshold=ligand_distance
                                           )
    if not nearby_residues:
        print("\nNo nearby residues found. Skipping...")
        return pdb_name, None, None
    print('done!')


    # We want to check if the ligand primarly interacts with proteins or RNA
    # Flatten all residue entries from all chains
    flat_residues = []
    for residue_list in nearby_residues.values():
        flat_residues.extend(residue_list)
    flat_residues.sort(key=lambda x: x[1])  # (residue, distance, type)
    
    # Count types
    rna_dna_count = sum(1 for _, _, t in flat_residues if t in {'RNA', 'DNA'})
    rna_dna_percent = rna_dna_count / len(flat_residues)
    protein_count = sum(1 for _, _, t in flat_residues if t == 'PROTEIN')

    # If more than 75% of residues are nucleic, skip this hit
    if protein_count == 0 or (rna_dna_count > protein_count and rna_dna_percent >= 0.75):
        print(f"\nLigand likely interacts primarily with nucleic acids ({round(rna_dna_percent, 4)*100}%). Skipping...")
        return pdb_name, 'NUCLEIC ACIDS', None
    
    
    # Get the chains file with the chains nearest the ligand
    outfile = str(TMP_DIR / f'{pdb_hit_id}_chains.cif')
    chains_file = get_chains_file(structure=pdb_structure,
                                  nearby_residues=nearby_residues,
                                  output_file=outfile
                                  )
    if not chains_file:
        return pdb_name, None, None, 
    
    # Get the pocket file with just the residues near the ligand
    outfile = str(TMP_DIR / f'{pdb_hit_id}_pocket_d{ligand_distance}.cif')
    hit_pocket_file = get_pocket_file(structure=pdb_structure,
                                      nearby_residues=nearby_residues,
                                      output_file=outfile,
                                      )

    del pdb_structure

    return pdb_name, chains_file, hit_pocket_file


def download_af_file(uniprot_id):
    # Fetch the PDB file corresponding to the hit ID
    # The AFDB is constantly updating, in the end it will prob be necessary
    # to have a local DB of structures. Aprox 20 TB.
    url = f'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb'
    response = requests.get(url)

    if response.status_code != 200:
        print (f"Failed to download AlphaFold model for {uniprot_id}\n{url}")
        return None

    output_path = f'tmp/AF-{uniprot_id}-F1-model_v6.pdb'
    with open(output_path, 'wb') as f:
        f.write(response.content)

    return Path(output_path)

    
def process_af_hit(hit_id):
    
    # Fetch the AF file corresponding to the hit ID
    print(f"\nFetching AlphaFold {hit_id} file...", end='')
    hit_pdb_file = download_af_file(uniprot_id=hit_id)
    
    if not hit_pdb_file:
        return None, None, None
    
    # Load structure
    parser = PDBParser(QUIET=True)
    pdb_structure = parser.get_structure('structure', hit_pdb_file)
    structure_org = pdb_structure.header['source']['1']['organism_scientific']
    structure_name = pdb_structure.header['compound']['1']['molecule']
    pdb_name = structure_org + ' | ' + structure_name
    print("done!")
    
    # First, we need to predict the pockets of this hit
    hit_pockets = predict_pockets(pdb_file=hit_pdb_file,
                                  fpocket_path=FPOCKET_PATH,
                                  p2rank_path=P2RANK_PATH
                                  )
    filtered_pockets = []
    for pocket in hit_pockets:
        if pocket['p2rank_score'] < P2RANK_SCORE_CUTOFF:
            continue
        
        if pocket['mean_plddt'] < POCKET_PLDDT_CUTOFF:
            continue
        
        if pocket['median_plddt'] < POCKET_PLDDT_CUTOFF:
            continue

        if pocket['mean_pae'] > POCKET_PAE_CUTOFF:
            continue
        
        if pocket['median_pae'] > POCKET_PAE_CUTOFF:
            continue        
        
        filtered_pockets.append(pocket)
    
    pocket_distance = int(LIGAND_DISTANCE)
    pocket_files = []
    for pocket in filtered_pockets:
        pocket_center = (pocket['centroid_x'],
                         pocket['centroid_y'],
                         pocket['centroid_z']
                         )
        pocket_id = pocket["pocket_id"]
        pocket_residues = residues_near_point(structure=pdb_structure,
                                              point=pocket_center,
                                              distance_threshold=pocket_distance
                                              )
        pocket_file = get_pocket_file(structure=pdb_structure,
                                      nearby_residues=pocket_residues,
                                      output_file=f'tmp/{hit_id}_{pocket_id}_d{pocket_distance}.cif'
                                      )
        pocket_files.append((pocket, pocket_file))
  
    del pdb_structure
    return pdb_name, hit_pdb_file, pocket_files


def run_usalign(protein_1, protein_2, usalign_path='USalign'):
    cmd = [usalign_path, str(protein_1), str(protein_2), '-mol', 'prot', '-ter', '1']
    proc = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    if not proc.stdout.strip() or 'TM-score' not in proc.stdout:
        print(f"USalign produced no output for {protein_1} vs {protein_2}: {proc.stderr.strip()}")
        raise RuntimeError(f"USalign produced no output for {protein_1} vs {protein_2}")
    return parse_us_align_output(proc.stdout)



def global_alignment(protein_id, hit_protein_file):
    global_results = defaultdict(list)
    protein_file = PDB_DIR / f'AF-{protein_id}-F1-model_v6.pdb'
    outputs = run_usalign(protein_1=protein_file,
                          protein_2=hit_protein_file,
                          usalign_path=USALIGN_PATH
                          )
    for output in outputs:
        current_chain = output['Chain_2']
        parsed_dict = {'global_rmsd': output['RMSD'],
                       'global_seq_id': output['Seq_ID'],
                       'global_score': output['TM-score_2'],
                       'local_rmsd': np.nan,
                       'local_identity': np.nan,
                       'local_score': np.nan
                       }
        output_dict = {current_chain: parsed_dict}
        global_results[protein_id].append(output_dict)

    return global_results


def compute_global_alignments(protein_ids, hit_protein_file, workers):
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(global_alignment, pid, hit_protein_file): pid for pid in protein_ids}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def local_alignment(pocket_entry_dict, hit_pocket_file, ligand_distance=10):
    results = defaultdict(list)
    
    protein_id = pocket_entry_dict['protein_id']
    pocket_id = pocket_entry_dict['pocket_id'].strip()

    pocket_distance = int(ligand_distance)
    pocket_file = POCKETS_DIR / f'{protein_id}_{pocket_id}_d{pocket_distance}.cif'
    assert pocket_file.is_file()
    
    outputs = run_usalign(protein_1=pocket_file,
                          protein_2=hit_pocket_file,
                          usalign_path=USALIGN_PATH
                          )
    if not outputs:
        return results
    
    for output in outputs:
        current_chain = output['Chain_2']
        parsed_dict = {'global_rmsd': np.nan,
                       'global_seq_id': np.nan,
                       'global_score': np.nan,
                       'local_rmsd': output['RMSD'],
                       'local_identity': output['Seq_ID'],
                       'local_score': output['TM-score_2'],
                       }
        output_dict = {current_chain: parsed_dict}
        results[protein_id].append((pocket_entry_dict, output_dict))
    return results


def compute_local_alignments(pockets_list, hit_pocket_file, workers):
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(local_alignment, pocket_entry, hit_pocket_file, LIGAND_DISTANCE): pocket_entry for pocket_entry in pockets_list}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]
        

def list_to_bv(l):
    bv = ExplicitBitVect(len(l))
    bv.SetBitsFromList(np.where(l)[0].tolist())
    return bv


def compute_similarity(ref_fp_vect,
                       list_fp_vect,
                       SIMILARITY,
                       alpha=0.9
                       ):
    similarity_type = SIMILARITY.lower().replace("-", "")
    similarity_functions = {'tanimoto': DataStructs.TanimotoSimilarity,
                            "dice": DataStructs.DiceSimilarity,
                            "sokal": DataStructs.SokalSimilarity,
                            "russel": DataStructs.RusselSimilarity,
                            "rogotgoldberg": DataStructs.RogotGoldbergSimilarity,
                            "allbit": DataStructs.AllBitSimilarity,
                            "kulczynski": DataStructs.KulczynskiSimilarity,
                            "mcconnaughey": DataStructs.McConnaugheySimilarity,
                            "asymmetric": DataStructs.AsymmetricSimilarity,
                            "braunblanquet": DataStructs.BraunBlanquetSimilarity
                            }
        
    
    if similarity_type in similarity_functions:
        metric = similarity_functions[similarity_type]
        sims = [DataStructs.FingerprintSimilarity(ref_fp_vect,
                                                  fp,
                                                  metric=metric
                                                  )
                for fp in list_fp_vect
                ]
        return sims
    
    elif similarity_type == 'cosine':
        
        def _available_ram_gb() -> float:
            return psutil.virtual_memory().available / (1024 ** 3)      
        
        n = len(list_fp_vect)
        fp0 = list_fp_vect[0]
        size = len(fp0)
        is_numpy = isinstance(fp0, np.ndarray)

        # Prepare the reference vector once
        if not isinstance(ref_fp_vect, np.ndarray):
            ref = np.zeros(size, dtype=np.float32)
            ConvertToNumpyArray(ref_fp_vect, ref)
        else:
            ref = ref_fp_vect.astype(np.float32)
        ref_norm = np.linalg.norm(ref)

        #Normal path with enough RAM
        if _available_ram_gb() >= 64:
            mat = np.zeros((n, size), dtype=np.float32)
            if is_numpy:
                for i, fp in enumerate(tqdm(list_fp_vect, desc="Loading vectors")):
                    mat[i] = fp
            else:
                for i, fp in enumerate(tqdm(list_fp_vect, desc="Converting fingerprints")):
                    ConvertToNumpyArray(fp, mat[i])

            mat_norms = np.linalg.norm(mat, axis=1)
            dots = mat @ ref
            with np.errstate(invalid='ignore', divide='ignore'):
                return np.where(
                    (mat_norms == 0) | (ref_norm == 0),
                    0.0,
                    dots / (mat_norms * ref_norm),
                )

        # If low memory: stream in chunks
        sims = np.empty(n, dtype=np.float32)
        chunk_buf = np.zeros((4096, size), dtype=np.float32)

        with tqdm(total=n, desc="Computing similarities (chunked)") as pbar:
            for start in range(0, n, 4096):
                end = min(start + 4096, n)
                actual = end - start
                chunk = chunk_buf[:actual]
                chunk[:] = 0.0  # reuse buffer cleanly

                if is_numpy:
                    for j, fp in enumerate(list_fp_vect[start:end]):
                        chunk[j] = fp
                else:
                    for j, fp in enumerate(list_fp_vect[start:end]):
                        ConvertToNumpyArray(fp, chunk[j])

                norms = np.linalg.norm(chunk, axis=1)
                dots = chunk @ ref
                with np.errstate(invalid='ignore', divide='ignore'):
                    sims[start:end] = np.where(
                        (norms == 0) | (ref_norm == 0),
                        0.0,
                        dots / (norms * ref_norm),
                    )
                pbar.update(actual)

        return sims
    
    
    elif similarity_type == 'tversky':
        beta = 1.0 - alpha
        return [DataStructs.TverskySimilarity(ref_fp_vect, fp, alpha, beta)
                for fp in list_fp_vect
                ]


    elif similarity_type == 'tsi':
        """
        Compute the Tripartite Similarity Index (TSI)
        Tulloss R. E. Assessment of Similarity Indices for Undesirable Properties and a New Tripartite Similarity Index Based on Cost Functions.
        """

        sims = []
        for fp in list_fp_vect:
            a = (ref_fp_vect & fp).GetNumOnBits()
            b = ref_fp_vect.GetNumOnBits() - a
            c = fp.GetNumOnBits() - a 
            
            # Avoid invalid values
            if (max(b, c) + a) == 0 or (a + 1) == 0 or (a + b) == 0 or (a + c) == 0:
                sims.append(0.0)
                continue
    
            # Cost Functions
            U = math.log2(1 + (min(b, c) + a) / (max(b, c) + a))
            S = 1 / math.sqrt(math.log2(2 + min(b, c) / (a + 1)))
            R = math.log2(1 + a / (a + b)) * math.log2(1 + a / (a + c))
        
            # Tripartite Similarity Index T
            T = math.sqrt(U * S * R)
    
            sims.append(T)
        
        return sims

    else:
        raise ValueError(f"Unknown similarity metric: {SIMILARITY}")



#%% Match reference molecule to database to find best matches
if __name__ == '__main__':
    args = parse_args()
    ref_smiles = args.smiles
    COMPOUND_NAME = args.compound
    Path('results').mkdir(exist_ok=True)

    pocket_df = pd.read_pickle('db/proteins/all_pockets.pkl')
    print(f"Started with {pocket_df['protein_id'].nunique()} proteins")
    # Apply filters to the pocket data based on input parameters
    pocket_df = pocket_df[
        (pocket_df['p2rank_score'] >= P2RANK_SCORE_CUTOFF) &
        (pocket_df['mean_plddt'] >= POCKET_PLDDT_CUTOFF) &
        (pocket_df['median_plddt'] >= POCKET_PLDDT_CUTOFF) &
        (pocket_df['mean_pae'] <= POCKET_PAE_CUTOFF) &
        (pocket_df['median_pae'] <= POCKET_PAE_CUTOFF) &
        (pocket_df['max_domain_prop'] >= DOMAIN_CUTOFF)
        ]
    pocket_len = len(pocket_df)
    main_pocket_dict = pocket_df.to_dict(orient='records')
    pocket_prots = list(set([x['protein_id'] for x in main_pocket_dict]))
    print(f"After applying filters: {pocket_df['protein_id'].nunique()} proteins")

    # Make sure we have all the pocket files needed
    pocket_distance = int(LIGAND_DISTANCE)
    rows_to_drop = []
    for idx, row in tqdm(pocket_df.iterrows(), total=pocket_len, desc='Checking pocket files'):

        pocket_file = POCKETS_DIR / f'{row["protein_id"]}_{row["pocket_id"]}_d{pocket_distance}.cif'
        if not pocket_file.is_file() or pocket_file.stat().st_size == 0:  # has this pocket been already created?
            parser = PDBParser(QUIET=True)
            pocket_parent_structure = parser.get_structure('structure', PDB_DIR / f'AF-{row["protein_id"]}-F1-model_v6.pdb')
            pocket_nearby_residues = residues_near_point(structure=pocket_parent_structure,
                                                         point=(row['centroid_x'],
                                                                row['centroid_y'],
                                                                row['centroid_z']
                                                                ),
                                                         distance_threshold=pocket_distance
                                                         )
            
            if pocket_nearby_residues is None:
                print(f"No residues found for {row['protein_id']} pocket {row['pocket_id']}")
                rows_to_drop.append(idx)
                continue  # or raise, or log — but don't try to save an empty structure
            
            pocket_file = get_pocket_file(structure=pocket_parent_structure,
                                          nearby_residues=pocket_nearby_residues,
                                          output_file=pocket_file
                                          )
            if not pocket_file.is_file():
                print(row)
                print(pocket_nearby_residues)
                raise Exception
            if pocket_file.stat().st_size == 0:
                print("Created a file with size 0!")
                print(row)
                print(pocket_nearby_residues)
                raise Exception
            
            
    if rows_to_drop:
        print(f"Dropping {len(rows_to_drop)} rows from pocket_df")
        pocket_df = pocket_df.drop(index=rows_to_drop).reset_index(drop=True)
        pocket_len = len(pocket_df)

    # Compute fingerprints for our compound    
    has_scaffold = True  # Check if our molecule has a scaffold. Otherwise, we will skip scaffold in the weights
    ref_mol = Chem.MolFromSmiles(ref_smiles)
    ref_scaffold = MurckoScaffold.GetScaffoldForMol(ref_mol)
    if ref_scaffold.GetNumHeavyAtoms() == 0:
        ref_scaffold = Chem.MolFromSmiles(ref_smiles)
        has_scaffold = False
        print(f"\nNo scaffold for {COMPOUND_NAME}! Weights will be redistributed")
    
    morgan_fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    rdkit_fpgen = AllChem.GetRDKitFPGenerator(fpSize=2048)
    ref_mol_fp_dict = {'MORGAN': morgan_fpgen.GetFingerprint(ref_mol),
                       'RDKIT': rdkit_fpgen.GetFingerprint(ref_mol),
                       'MACCS': MACCSkeys.GenMACCSKeys(ref_mol),
                       'ERG': rdReducedGraphs.GetErGFingerprint(ref_mol),
                       'PATTERN': Chem.PatternFingerprint(ref_mol, fpSize=2048)
                       }
    ref_mol_sc_fp_dict = {'MORGAN': morgan_fpgen.GetFingerprint(ref_scaffold),
                          'RDKIT': rdkit_fpgen.GetFingerprint(ref_scaffold),
                          'MACCS': MACCSkeys.GenMACCSKeys(ref_scaffold),
                          'ERG': rdReducedGraphs.GetErGFingerprint(ref_scaffold),
                          'PATTERN': Chem.PatternFingerprint(ref_scaffold, fpSize=2048)
                          }
    
    # Select the desired FPs
    ref_fps_dict = {}
    for FP, _ in STRUCTURAL_FINGERPRINTS:
        ref_fps_dict[FP] = ref_mol_fp_dict[FP]
    
    ref_sc_fps_dict = {}
    for FP, _ in SCAFFOLD_FINGERPRINTS:
        ref_sc_fps_dict[FP] = ref_mol_sc_fp_dict[FP]
    
    
    # Load database
    db_fps_dict = {}
    for FP, _ in STRUCTURAL_FINGERPRINTS:
        with open(f'db/fps/COMBINED_{FP}.pkl', 'rb') as f:   
                db_fps_dict[FP] = pickle.load(f)         
    db_sc_fps_dict = {}
    for FP, _ in SCAFFOLD_FINGERPRINTS:
        with open(f'db/fps/COMBINED_SCAFFOLD_{FP}.pkl', 'rb') as f:
            db_sc_fps_dict[FP] = pickle.load(f)
    
    
    # Get DB data
    # Just use the first
    id_to_smiles = {x[0]: x[1] for x in db_fps_dict[STRUCTURAL_FINGERPRINTS[0][0]]}
    mol_names = [x[0] for x in db_fps_dict[STRUCTURAL_FINGERPRINTS[0][0]]]
    affinities = [x[2] for x in db_fps_dict[STRUCTURAL_FINGERPRINTS[0][0]]]
    
    # Calculate whole similarities
    all_similarities = {}
    structural_columns = []
    for FP, SIMILARITY_MEASURE in STRUCTURAL_FINGERPRINTS:
        print(f"\nComputing similarities for {FP}|{SIMILARITY_MEASURE}")
        comb_name = f'{FP}|{SIMILARITY_MEASURE}'
        ref_fp = ref_fps_dict[FP]
        db_fps = [x[3] for x in db_fps_dict[FP]]
        fp_similarity = np.array(compute_similarity(ref_fp, db_fps, SIMILARITY_MEASURE))
        all_similarities[comb_name] = fp_similarity
        structural_columns.append(comb_name)
    
    # Calculate scaffold similarities
    scaffold_columns = []
    for FP, SIMILARITY_MEASURE in SCAFFOLD_FINGERPRINTS:
        print(f"\nComputing similarities for scaffold {FP}|{SIMILARITY_MEASURE}")
        comb_name = f'scaffold_{FP}|{SIMILARITY_MEASURE}'
        ref_fp = ref_sc_fps_dict[FP]
        db_fps = [x[3] for x in db_sc_fps_dict[FP]]
        fp_similarity = np.array(compute_similarity(ref_fp, db_fps, SIMILARITY_MEASURE))
        all_similarities[comb_name] = fp_similarity
        scaffold_columns.append(comb_name)
    
    
    # Put it all together
    hits_df = pd.DataFrame({'id': mol_names,
                            'smiles': [id_to_smiles[n] for n in mol_names],
                            'affinity':  affinities,
                            **all_similarities
                            })
    hits_df.set_index('id', inplace=True)

    if has_scaffold:
        sim_cols = structural_columns + scaffold_columns
    else:
        sim_cols = structural_columns
    sim_records = hits_df[sim_cols].to_dict(orient='records')
    
    # Expand by affinities
    expanded_rows = []
    for i, (chem_id, row) in enumerate(tqdm(hits_df[['smiles', 'affinity']].iterrows(),
                                            total=len(hits_df),
                                            desc='Parsing hits'
                                            )
                                       ):
        sims = sim_records[i]
        for affinity_key, affinity_data in row['affinity'].items():
            uniprot_id, pdb_id = affinity_key
            assay_type, operator, value, unit, database = affinity_data
            r = {
                'chem_id':           chem_id,
                'smiles':            row['smiles'],
                'uniprot_target_id': uniprot_id,
                'pdb_target_id':     pdb_id,
                'assay_type':        assay_type,
                'operator':          operator,
                'value':             value,
                'unit':              unit,
                'database':          database,
                **sims,
            }
            expanded_rows.append(r)

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df = expanded_df[(expanded_df['value'].isna()) | (expanded_df['value'] <= AFFINITY_THRESHOLD)]
    
    # Calculate simil. metric
    if has_scaffold:
        expanded_df['similarity_score'] = sum(expanded_df[col] * w
                                            for col, w in WEIGHTS.items()
                                            if col in expanded_df.columns
                                            )
    else:
        scaff_weights = 0
        new_weights = {}
        drop_cols = []
        for col, w in WEIGHTS.items():
            if 'scaffold' in col:
                scaff_weights += w
                drop_cols.append(col)
            else:
                new_weights[col] = w
        add_w = scaff_weights / len(new_weights)
        for col, w in new_weights.items():
            new_weights[col] += add_w
        
        expanded_df['similarity_score'] = sum(expanded_df[col] * w
                                            for col, w in new_weights.items()
                                            if col in expanded_df.columns
                                            )

    
    expanded_df = expanded_df.sort_values('similarity_score', ascending=False)
    
    # Define database priority: PDB highest, then UniProt/BDB
    source_priority = {'PDB': 0, 'CHEMBL': 1}  # smaller number = higher priority
    
    # Add temporary column for sorting
    expanded_df['db_priority'] = expanded_df['database'].map(source_priority).fillna(99)
    expanded_df = expanded_df.sort_values(by=['similarity_score', 'db_priority', 'value', 'chem_id'],
                                              ascending=[False, True, True, True],
                                              )
    
    # For a given target, keep the highest similarity hit
    with_target = expanded_df[expanded_df['uniprot_target_id'].notna()]
    no_target   = expanded_df[expanded_df['uniprot_target_id'].isna()]
    with_target = with_target.drop_duplicates(subset='uniprot_target_id', keep='first')
    hits_df = pd.concat([with_target, no_target], ignore_index=True)
    hits_df = hits_df.sort_values(by=['similarity_score', 'db_priority', 'value', 'chem_id'],
                                  ascending=[False, True, True, True]
                                  )
    hits_df = hits_df.drop(columns=['db_priority']).reset_index(drop=True)
    hits_df.to_csv(f'results/{COMPOUND_NAME}_all_hits.csv')


    # Heatmap
    agg_df = hits_df.groupby('chem_id', as_index=False).agg({'similarity_score':'max'}).sort_values(by='similarity_score', ascending=False).head(NUM_HITS)
    heatmap_data = agg_df.set_index('chem_id').T
    annot_labels = heatmap_data.round(2).astype(str)

    plt.figure(figsize=(12, 2), dpi=300)
    ax = sns.heatmap(
        heatmap_data,
        annot=annot_labels,
        fmt="",
        cmap=plt.cm.viridis_r,
        cbar=False,
        xticklabels=True,
        yticklabels=True,
        vmin=0,
        vmax=1
    )
    plt.title(f'Top {NUM_HITS} Matches to {COMPOUND_NAME}')
    plt.xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'results/{COMPOUND_NAME}_best_hits.png', bbox_inches='tight', dpi=300)
    plt.show()
    

    # Draw molecule
    ref_smiles_std = Chem.MolToSmiles(ref_mol, isomericSmiles=False)
    ref_mol_std = Chem.MolFromSmiles(ref_smiles_std)
    AllChem.Compute2DCoords(ref_mol_std)

    # Create the dict keys to be used
    dict_keys = ['chem_id',
                 'smiles',
                 *sim_cols,
                 'similarity_score',
                 'database',
                 'hit_protein_id',
                 'hit_protein_name',
                 'hit_protein_chain',
                 'hit_protein_pocket_id',
                 'hit_protein_pocket_score',
                 'hit_protein_affinity',
                 'target_protein_id',
                 'target_protein_name',
                 'target_protein_pocket_id',
                 'target_protein_pocket_score',
                 'global_rmsd',
                 'global_seq_id',
                 'global_score',
                 'local_rmsd',
                 'local_seq_id',
                 'local_score'
                 ]


    top_chem_ids = (hits_df.groupby('chem_id')['similarity_score']
                    .max()
                    .sort_values(ascending=False)
                    .head(NUM_HITS)
                    .index
                    )
    top_hits_df = hits_df[hits_df['chem_id'].isin(top_chem_ids)].copy()
    top_hits_df[sim_cols + ['similarity_score']] = top_hits_df[sim_cols + ['similarity_score']].round(4)
    print(f'Searching through {len(top_chem_ids)} hits and {len(top_hits_df)} proteins')

    # MAIN LOOP HERE
    all_results = []
    for _, row in tqdm(top_hits_df.iterrows(), total=len(top_hits_df)):
        chem_id = row['chem_id']
        protein_hit = row['pdb_target_id'] if pd.notna(row['pdb_target_id']) else row['uniprot_target_id']
        
        # If it has PDB, we will use PDB entry instead of AF
        is_pdb = True if pd.notna(row['pdb_target_id']) else False

        database = row['database']
        similarity_score = row['similarity_score']
        hit_smiles = row['smiles']
        
        # Check affinities for later
        if pd.isna(row['value']):
            affinity = pd.NA
        else:
            affinity = row['assay_type'] + row['operator'] + str(row['value']) + row['unit']

        # Two main forks, either PDB or AlphaFold proteins
        if is_pdb:
            pdb_chem_id = chem_id.split('PDB', 1)[1]
            pdb_name, hit_chains_file, hit_pocket_file = process_pdb_hit(chem_hit_id=pdb_chem_id,
                                                                         pdb_hit_id=protein_hit,
                                                                         ligand_distance=LIGAND_DISTANCE
                                                                         )
        
            # If more than 75% of residues near the ligand are nucleic acids
            # we will assume that the ligand is not interacting with a protein
            # and thus we won't carry out alignments.
            if hit_chains_file == 'NUCLEIC ACIDS':
                new_result = {k: pd.NA for k in dict_keys}
                new_result.update({k: row[k] for k in dict_keys if k in row.index})
                new_result.update({'hit_protein_id': protein_hit,
                                   'hit_protein_name': pdb_name,
                                   'hit_protein_affinity': affinity,
                                   'target_protein_id' : "INTERACTING WITH NUCLEIC ACIDS",
                                   }
                                  )
                all_results.append(new_result)
                continue
            
            # If the chains file was not generated, skip
            if not hit_chains_file:
                new_result = {k: pd.NA for k in dict_keys}
                new_result.update({k: row[k] for k in dict_keys if k in row.index})
                new_result.update({'hit_protein_id': protein_hit,
                                   'hit_protein_name': pdb_name,
                                   'hit_protein_affinity': affinity,
                                   'target_protein_id' : "CHAIN FILE NOT GENERATED",
                                   }
                                  )
                all_results.append(new_result)
                continue           

            # First, compute all global alignments (full protein-protein)
            chunks = list(chunk_list(pocket_prots, CHUNK_SIZE)) 
            with ProcessPoolExecutor(max_workers=EXT_WORK) as executor:
                futures = [executor.submit(compute_global_alignments, chunk, hit_chains_file, INT_WORK) for chunk in chunks]
                global_results = []
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   position=0,
                                   leave=True,
                                   desc="Computing global alignments..."):
                    for res in future.result():
                        global_results.append(res)

            # Then, compute al local alignments (pocket to pocket)
            chunks = list(chunk_list(main_pocket_dict, CHUNK_SIZE)) 
            local_results = []
            with ProcessPoolExecutor(max_workers=EXT_WORK) as executor:
                futures = [executor.submit(compute_local_alignments, chunk, hit_pocket_file, INT_WORK) for chunk in chunks]
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   position=0,
                                   leave=True,
                                   desc="Computing local alignments..."):
                    for res in future.result():
                        local_results.append(res)
        
            # Parse all global results into a dict
            parsed_global_res = {}
            for d in global_results:
                for protein_id, results_list in d.items():
                    for result_dict in results_list:
                        for chain, result in result_dict.items():
                            new_k = (protein_id, chain)
                            parsed_global_res[new_k] = result
            
            # Go through the local results and create new final results
            for d in local_results:
                for protein_id, results_list in d.items():
                    for pocket_data, result_dict in results_list:
                        for chain, local_res in result_dict.items():
                            
                            # Get results from the global alignment
                            global_key = (protein_id, chain)
                            global_res = parsed_global_res[global_key]
                            
                            new_result = {k: pd.NA for k in dict_keys}
                            new_result.update({k: row[k] for k in dict_keys if k in row.index})
                            new_result.update({'hit_protein_id': protein_hit,
                                               'hit_protein_name': pdb_name,
                                               'hit_protein_affinity': affinity,
                                               'hit_protein_chain' : chain,
                                               'target_protein_id': protein_id,
                                               'target_protein_name': pocket_data['Product Description'],
                                               'target_protein_pocket_id': pocket_data['pocket_id'],
                                               'target_protein_pocket_score': pocket_data['p2rank_score'],
                                               'global_rmsd': global_res['global_rmsd'],
                                               'global_seq_id': global_res['global_seq_id'],
                                               'global_score': global_res['global_score'],
                                               'local_rmsd': local_res['local_rmsd'],
                                               'local_seq_id': local_res['local_identity'],
                                               'local_score': local_res['local_score']
                                               }
                                              )
                            if new_result['global_score'] >= TM_SCORE_CUTOFF or new_result['local_score'] >= TM_SCORE_CUTOFF:
                                all_results.append(new_result)

                    
        else:
            
            pdb_name, hit_chains_file, hit_pocket_list = process_af_hit(hit_id=protein_hit)
            
            if not pdb_name:
                new_result = {k: pd.NA for k in dict_keys}
                new_result.update({k: row[k] for k in dict_keys if k in row.index})
                new_result.update({'hit_protein_id': protein_hit,
                                   'hit_protein_name': pdb_name,
                                   'hit_protein_affinity': affinity,
                                   'target_protein_id' : "AF PDB NOT GENERATED, VIRAL PROTEIN?",
                                   }
                                  )
                all_results.append(new_result)
                continue                     
            
            # If the chains file was not generated, skip
            if not hit_chains_file:
                new_result = {k: pd.NA for k in dict_keys}
                new_result.update({k: row[k] for k in dict_keys if k in row.index})
                new_result.update({'hit_protein_id': protein_hit,
                                   'hit_protein_name': pdb_name,
                                   'hit_protein_affinity': affinity,
                                   'target_protein_id' : "CHAIN FILE NOT GENERATED",
                                   }
                                  )
                all_results.append(new_result)
                continue           

            # First, compute all global alignments (full protein-protein)
            chunks = list(chunk_list(pocket_prots, CHUNK_SIZE)) 
            with ProcessPoolExecutor(max_workers=EXT_WORK) as executor:
                futures = [executor.submit(compute_global_alignments, chunk, hit_chains_file, INT_WORK) for chunk in chunks]
                global_results = []
                for future in tqdm(as_completed(futures),
                                   total=len(futures),
                                   position=0,
                                   leave=True,
                                   desc="Computing global alignments..."
                                   ):
                    for res in future.result():
                        global_results.append(res)
            
            # Then, compute al local alignments (pocket to pocket)
            chunks = list(chunk_list(main_pocket_dict, CHUNK_SIZE)) 
            local_results = []
            for hit_pocket, hit_pocket_file in hit_pocket_list:    
                with ProcessPoolExecutor(max_workers=EXT_WORK) as executor:
                    futures = [executor.submit(compute_local_alignments, chunk, hit_pocket_file, INT_WORK) for chunk in chunks]
                    for future in tqdm(as_completed(futures),
                                       total=len(futures),
                                       position=0,
                                       leave=True,
                                       desc=f"Aligning {hit_pocket['pocket_id']}"
                                       ):
                        for res in future.result():
                            local_results.append((hit_pocket, res))
            
            # Parse all global results into a dict
            parsed_global_res = {}
            for d in global_results:
                for protein_id, results_list in d.items():
                    for result_dict in results_list:
                        for chain, result in result_dict.items():
                            new_k = (protein_id, chain)
                            parsed_global_res[new_k] = result
            
            # Go through the local results and create new final results
            for hit_pocket, d in local_results:
                for protein_id, results_list in d.items():
                    for pocket_data, result_dict in results_list:
                        for chain, local_res in result_dict.items():
                            
                            # Get results from the global alignment
                            global_key = (protein_id, chain)
                            global_res = parsed_global_res[global_key]
                            
                            new_result = {k: pd.NA for k in dict_keys}
                            new_result.update({k: row[k] for k in dict_keys if k in row.index})
                            
                            new_result.update({'hit_protein_id': protein_hit,
                                               'hit_protein_name': pdb_name,
                                               'hit_protein_affinity': affinity,
                                               'hit_protein_chain' : chain,
                                               'hit_protein_pocket_id' : hit_pocket['pocket_id'].strip(),
                                               'hit_protein_pocket_score': hit_pocket['p2rank_score'],
                                               'target_protein_id': protein_id,
                                               'target_protein_name': pocket_data['Product Description'],
                                               'target_protein_pocket_id': pocket_data['pocket_id'],
                                               'target_protein_pocket_score': pocket_data['p2rank_score'],
                                               'global_rmsd': global_res['global_rmsd'],
                                               'global_seq_id': global_res['global_seq_id'],
                                               'global_score': global_res['global_score'],
                                               'local_rmsd': local_res['local_rmsd'],
                                               'local_seq_id': local_res['local_identity'],
                                               'local_score': local_res['local_score']
                                               }
                                              )
                            if new_result['global_score'] >= TM_SCORE_CUTOFF or new_result['local_score'] >= TM_SCORE_CUTOFF:
                                all_results.append(new_result)
                

    results_df = pd.DataFrame(all_results, columns=dict_keys)
    results_df.to_csv(f'results/{COMPOUND_NAME}_results.csv')
