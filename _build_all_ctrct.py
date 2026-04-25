"""Build all_ctrct.pkl: 11-digit tract IDs for CBGs in top-N CBSAs by population.

Replaces the (now-removed) per-MSA accumulator at the bottom of 1-main: that one
only emitted tracts for CBSAs that actually got simulated, so all_ctrct shrunk
to whatever subset had been processed. This builds the full top-N up front so
0-OD_from_Individual.py can pre-filter Veraset visits to the right scope.
"""
import pandas as pd

from paths import SHP_DIR

UN_ST = ['02', '15', '60', '66', '69', '72', '78']  # AK, HI + territories (matches 1-main)
TOP_N = 100

smart_loc = pd.read_pickle(SHP_DIR / 'SmartLocationDatabase.pkl')
smart_loc['BGFIPS'] = smart_loc['BGFIPS'].astype(str).str.zfill(12)
smart_loc = smart_loc[~smart_loc['BGFIPS'].str[:2].isin(UN_ST)].reset_index(drop=True)
smart_loc['CBSA_Name'] = smart_loc['CBSA_Name'].str.replace('/', '-')

msa_pop = (smart_loc.drop_duplicates(subset=['CBSA_Name', 'CBSA'])
           [['CBSA_Name', 'CBSA', 'CBSA_POP']]
           .sort_values(by='CBSA_POP', ascending=False).reset_index(drop=True))

top_cbsas = msa_pop.head(TOP_N)
print(f'Top {TOP_N} CBSAs by population:')
print(f'  smallest CBSA_POP: {top_cbsas["CBSA_POP"].min():,.0f}')
print(f'  largest  CBSA_POP: {top_cbsas["CBSA_POP"].max():,.0f}')

in_top = smart_loc[smart_loc['CBSA'].isin(top_cbsas['CBSA'])]
tracts = sorted(in_top['BGFIPS'].str[:11].unique())

state_counts = pd.Series([t[:2] for t in tracts]).value_counts()
print(f'\nTotal CBGs in top {TOP_N} CBSAs: {len(in_top):,}')
print(f'Unique 11-digit tracts:        {len(tracts):,}')
print(f'States covered ({len(state_counts)}):')
print(state_counts.to_string())

out = SHP_DIR / 'all_ctrct.pkl'
pd.DataFrame(tracts).to_pickle(out)
print(f'\nSaved → {out}')
