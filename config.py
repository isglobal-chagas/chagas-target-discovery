import os
from pathlib import Path

# Paths
FPOCKET_PATH = '/home/isglobal.lan/aros/software/fpocket/bin/fpocket'
P2RANK_PATH = '/home/isglobal.lan/aros/software/p2rank_2.4/prank'
USALIGN_PATH = '/home/isglobal.lan/aros/software/us-align/USalign'
PDB_DIR = Path('db/proteins/pdb')
POCKETS_DIR = Path('db/proteins/pocket_pdbs')
TMP_DIR = Path('tmp')
TMP_DIR.mkdir(exist_ok=True, parents=True)

# Environment
ENV = os.environ.copy()
ENV["TMPDIR"] = str(TMP_DIR.resolve())

# Workers
EXT_WORK = 10
INT_WORK = 20
CHUNK_SIZE = 20

# Thresholds
NUM_HITS = 20
P2RANK_SCORE_CUTOFF = 4.8
POCKET_PLDDT_CUTOFF = 70
POCKET_PAE_CUTOFF = 5
DOMAIN_CUTOFF = 0.5
TM_SCORE_CUTOFF = 0.5
LIGAND_DISTANCE = 12
AFFINITY_THRESHOLD = 1e3  # 1uM

STRUCTURAL_FINGERPRINTS = (('PATTERN', 'cosine'),
                           ('ERG', 'cosine'),
                           )

SCAFFOLD_FINGERPRINTS = (('PATTERN', 'cosine'),
                         )

WEIGHTS = {'PATTERN|cosine': 0.5,
           'ERG|cosine': 0.25,
           'scaffold_PATTERN|cosine': 0.25,
}
