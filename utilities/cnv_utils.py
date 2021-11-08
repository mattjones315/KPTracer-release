import pandas as pd

def count_cnvs_per_cell(df_cells, df_regions):
    df_merged = pd.merge(
        df_regions, df_cells, on='cell_group_name'
    )[['cell', 'cnv_name']]
    return dict(df_merged.groupby('cell').size())
