import pandas as pd

# --- Paths ---
checklists_file = "data_processing/data-cleaning/complete-checklists_Jan2021.txt"
checklist_counts_file = "data_processing/data-cleaning/n_checklists_Apr2021.csv"
output_file = "data_processing/data-cleaning/hotspotlist_with_50_complete_checklists_michigan.csv"
michigan_hotspots_file = "data_processing/ebird/michigan_hotspots.csv"

michigan_hotspots = pd.read_csv(michigan_hotspots_file)['hotspot_id'].to_list()

# Read complete checklists
df = pd.read_csv(checklists_file, delimiter="\t", keep_default_na=False)
df_nc_hotspot = pd.read_csv(checklist_counts_file, keep_default_na=False)

# Sort and merge
df_new = df_nc_hotspot.rename({'locality_id': 'LOCALITY ID', 'locality': 'LOCALITY'}, axis=1)
df_checklist_count = df_new.merge(df, on='LOCALITY', how='inner')

# Filter continental USA
df_usa = df_checklist_count[(df_checklist_count['STATE'] != "Alaska") &
                            (df_checklist_count['STATE'] != "District of Columbia") &
                            (df_checklist_count['STATE'] != "Hawaii")]

# Filter Michigan hotspots only
df_usa = df_usa[df_usa['LOCALITY ID'].isin(michigan_hotspots)]

# Apply threshold
threshold = 49
df_usa_threshold = df_usa[df_usa['n'] > threshold]

# Save Michigan hotspot list
df_usa_threshold[['LOCALITY ID']].to_csv(output_file, index=False, header=['LOCALITY_ID'])
print(f"Saved {len(df_usa_threshold)} Michigan hotspots with >=50 checklists")
