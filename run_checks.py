
import pandas as pd, numpy as np, re, datetime as pydt
from pathlib import Path

DATA = Path('data')
excel_path = DATA / 'DENZA 31 POINTS DATA 2020-2024 PHD.xlsx'
assert excel_path.exists(), f"Missing: {excel_path}"

def find_header_row(df, max_scan=60):
    for i in range(min(max_scan, len(df))):
        row_vals = df.iloc[i].astype(str).str.strip().str.lower().tolist()
        keys = ['sr. no', 'parameters', 'turbidity', 'coliform', 'ph', 'bod']
        if any(any(k in v for k in keys) for v in row_vals):
            return i
    return None

xls = pd.ExcelFile(excel_path)
YEARS = [s for s in xls.sheet_names if s.isdigit()]
header_rows = {s: find_header_row(pd.read_excel(excel_path, sheet_name=s, header=None)) for s in YEARS}

def get_date_cols(cols):
    dcs = [c for c in cols if isinstance(c, (pd.Timestamp, pydt.datetime))]
    return sorted(dcs, key=lambda x: (x.month if hasattr(x, 'month') else pd.Timestamp(x).month))

def parse_sheet_to_long(year, df):
    date_cols = get_date_cols(df.columns)
    records = []; current_station = None
    for _, row in df.iterrows():
        sr = row.get('Sr. No'); param_label = row.get('Parameters')
        if pd.isna(param_label) and isinstance(sr, str) and len(sr.strip())>0:
            current_station = sr.strip(); continue
        if pd.isna(param_label) or current_station is None:
            continue
        p = str(param_label).strip().lower()
        if p.startswith('fecal coliform') or p.startswith('faecal coliform'): key = 'Fecal_Coliform'
        elif p.startswith('turbidity'): key = 'Turbidity'
        else: continue
        for dcol in date_cols:
            val = row[dcol]
            if isinstance(val, str):
                v = val.strip().lower()
                if v == 'bdl': num = 0.0
                else:
                    try: num = float(re.sub(r'[^0-9.\-eE]', '', v))
                    except: num = np.nan
            else:
                num = pd.to_numeric(val, errors='coerce')
            m = dcol.month if hasattr(dcol, 'month') else pd.Timestamp(dcol).month
            records.append({'year': int(year),'station': current_station,'parameter': key,'month': m,'date': pd.Timestamp(year=int(year), month=m, day=1),'value': num})
    return pd.DataFrame(records)

tidy = {s: pd.read_excel(excel_path, sheet_name=s, header=header_rows[s]) for s in YEARS}
long_df = pd.concat([parse_sheet_to_long(s, tidy[s]) for s in YEARS], ignore_index=True)
long_df['is_stp'] = long_df['station'].str.contains('STP', case=False, na=False)

THRESH = {'Fecal_Coliform': 200.0, 'Turbidity': 5.0}
def exceedance_sample(df):
    return (df.groupby(['parameter','year']).apply(lambda g: (g['value']>THRESH[g.name[0]]).mean()*100).reset_index(name='exceedance_rate_samples_pct'))

exc_all = exceedance_sample(long_df)
exc_no_stp = exceedance_sample(long_df[~long_df['is_stp']])

def pick(param, year, table):
    v = table[(table['parameter']==param) & (table['year']==year)]['exceedance_rate_samples_pct']
    return float(v.iloc[0]) if len(v) else float('nan')

print("ALL 2024 FC exceed %:", round(pick('Fecal_Coliform', 2024, exc_all), 2))
print("ALL 2024 Turbidity exceed %:", round(pick('Turbidity', 2024, exc_all), 2))
print("NO_STP 2024 FC exceed %:", round(pick('Fecal_Coliform', 2024, exc_no_stp), 2))
print("NO_STP 2024 Turbidity exceed %:", round(pick('Turbidity', 2024, exc_no_stp), 2))
