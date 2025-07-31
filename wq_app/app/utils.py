
import re, numpy as np, pandas as pd
from pathlib import Path
import datetime as pydt
import requests

THRESH = {'Fecal_Coliform': 200.0, 'Turbidity': 5.0}

def find_header_row(df, max_scan=60):
    for i in range(min(max_scan, len(df))):
        row_vals = df.iloc[i].astype(str).str.strip().str.lower().tolist()
        keys = ['sr. no', 'parameters', 'dissolved oxygen', 'turbidity', 'coliform', 'ph', 'bod', 'ammonia']
        if any(any(k in v for k in keys) for v in row_vals):
            return i
    return None

def get_date_cols(cols):
    dcs = [c for c in cols if isinstance(c, (pd.Timestamp, pydt.datetime))]
    dcs = sorted(dcs, key=lambda x: (x.month if hasattr(x, 'month') else pd.Timestamp(x).month))
    return dcs

def parse_sheet_to_long(year, df):
    date_cols = get_date_cols(df.columns)

    def pick_station_from_row(row):
        for col_name in ['Sr. No', 'Sr.No', 'Sr.No.', 'Station', 'Location', 'Unnamed: 0']:
            if col_name in row.index and isinstance(row[col_name], str) and row[col_name].strip():
                return row[col_name].strip()
        for v in row.values:
            if isinstance(v, str) and len(v.strip()) >= 3:
                return v.strip()
        return None

    records = []
    current_station = None

    for _, row in df.iterrows():
        param_label = row.get('Parameters')
        if pd.isna(param_label):
            maybe_station = pick_station_from_row(row)
            if maybe_station:
                current_station = maybe_station
            continue

        if current_station is None or pd.isna(param_label):
            continue

        p = str(param_label).strip().lower()
        if p.startswith('fecal coliform') or p.startswith('faecal coliform'):
            key = 'Fecal_Coliform'
        elif p.startswith('turbidity'):
            key = 'Turbidity'
        else:
            continue

        for dcol in date_cols:
            val = row[dcol]
            if isinstance(val, str):
                v = val.strip().lower()
                if v == 'bdl':
                    num = 0.0
                else:
                    try:
                        num = float(re.sub(r'[^0-9.\-eE]', '', v))
                    except Exception:
                        num = np.nan
            else:
                num = pd.to_numeric(val, errors='coerce')

            m = dcol.month if hasattr(dcol, 'month') else pd.Timestamp(dcol).month
            records.append({
                'year': int(year),
                'station': current_station,
                'parameter': key,
                'month': m,
                'date': pd.Timestamp(year=int(year), month=m, day=1),
                'value': num
            })

    out = pd.DataFrame(records)
    out.loc[out['value'] < 0, 'value'] = np.nan

    st = out['station'].astype(str).str.lower()
    out['is_stp'] = (
        st.str.contains(r'\bstp\b', regex=True, na=False) |
        st.str.contains(' inlet', na=False) |
        st.str.contains('inlet ', na=False) |
        st.str.contains('sewage', na=False) |
        st.str.contains(' stp', na=False) |
        st.str.contains('stp ', na=False)
    )
    return out

def load_long_df(excel_path: Path):
    parquet_path = excel_path.parent / "long_df.parquet"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass

    xls = pd.ExcelFile(excel_path)
    years = [s for s in xls.sheet_names if s.isdigit()]
    header_rows = {s: find_header_row(pd.read_excel(excel_path, sheet_name=s, header=None)) for s in years}
    tidy = {s: pd.read_excel(excel_path, sheet_name=s, header=header_rows[s]) for s in years}
    long_df = pd.concat([parse_sheet_to_long(s, tidy[s]) for s in years], ignore_index=True)

    try:
        long_df.to_parquet(parquet_path)
    except Exception:
        pass
    return long_df

def annual_median(df):
    return (df.groupby(['parameter','year'], as_index=False)['value']
              .median()
              .rename(columns={'value':'annual_median'}))

def yoy_change(ann):
    out = ann.copy()
    out['yoy_change_pct'] = out.groupby('parameter')['annual_median'].pct_change()*100
    return out

def exceedance_sample(df):
    thr_map = df['parameter'].map(THRESH)
    exceed = (df['value'] > thr_map).astype(float)
    out = (df.assign(exceed=exceed)
             .groupby(['parameter','year'], as_index=False)['exceed'].mean())
    out = out.rename(columns={'exceed':'exceedance_rate_samples_pct'})
    out['exceedance_rate_samples_pct'] *= 100
    return out

def monthly_quarterly(df):
    monthly = (df.groupby(['parameter','year','month'], as_index=False)['value']
                 .median().rename(columns={'value':'monthly_median'}))
    monthly['quarter'] = ((monthly['month']-1)//3)+1
    monthly['yoy_change'] = (monthly.sort_values(['parameter','month','year'])
                             .groupby(['parameter','month'])['monthly_median']
                             .pct_change()*100)
    quarterly = (monthly.groupby(['parameter','year','quarter'], as_index=False)['monthly_median']
                 .median().rename(columns={'monthly_median':'quarterly_median'}))
    quarterly['yoy_change'] = (quarterly.sort_values(['parameter','quarter','year'])
                               .groupby(['parameter','quarter'])['quarterly_median']
                               .pct_change()*100)
    return monthly, quarterly

def monthly_quality(df, param: str):
    m = (df[df["parameter"]==param]
         .groupby(["year","month"], as_index=False)["value"].median()
         .rename(columns={"value":"q_median"}))
    m["date"] = pd.to_datetime(m["year"].astype(str)+"-"+m["month"].astype(str)+"-01")
    return m[["year","month","date","q_median"]]

def monthly_series(df, param):
    s = monthly_quality(df, param).rename(columns={"q_median":"val"})
    ts = s.set_index('date')["val"].asfreq("MS")
    return ts

def corr_and_leadlag(df):
    fc = monthly_series(df, 'Fecal_Coliform')
    tb = monthly_series(df, 'Turbidity')
    corr = fc.corr(tb)
    shifts = range(-3,4)
    rows = []
    for k in shifts:
        if k<0: c = fc.shift(-k).corr(tb)
        else:   c = fc.corr(tb.shift(k))
        rows.append({'lag_months': k, 'corr': c})
    return corr, pd.DataFrame(rows)

def robust_z_log(series):
    x = np.log1p(series.dropna())
    med = x.median()
    mad = (x-med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(index=series.index, dtype=float)
    return ((x-med)/(1.4826*mad)).reindex(series.index)

def anomaly_flags(df, threshold=3.5):
    rows = []
    for param in df['parameter'].unique():
        g = df[df['parameter']==param]
        for st, sub in g.groupby('station'):
            m = (sub.groupby(['year','month'])['value'].median().rename('val').reset_index())
            m['date'] = pd.to_datetime(m['year'].astype(str)+'-'+m['month'].astype(str)+'-01')
            m = m.set_index('date').sort_index()
            z = robust_z_log(m['val'])
            m['robust_z_log'] = z
            m['anomaly'] = (z.abs()>=threshold)
            m['parameter'] = param
            m['station'] = st
            rows.append(m.reset_index())
    return pd.concat(rows, ignore_index=True)

def segment_stations(df):
    res = []
    for param in df['parameter'].unique():
        g = df[df['parameter']==param].copy()
        g['exceed'] = (g['value'] > THRESH[param]).astype(float)
        ex = g.groupby('station', as_index=False)['exceed'].mean().rename(columns={'exceed':'exceed_pct'})
        ex['exceed_pct'] *= 100
        med = g.groupby('station', as_index=False)['value'].median().rename(columns={'value':'median'})
        seg = ex.merge(med, on='station', how='outer')
        def tier(row):
            if row['exceed_pct']>=60 or row['median']>=THRESH[param]*1.5: return 'High-Risk'
            if row['exceed_pct']>=30 or row['median']>=THRESH[param]:   return 'Moderate'
            return 'Lower'
        seg['tier'] = seg.apply(tier, axis=1)
        seg['parameter'] = param
        res.append(seg)
    return pd.concat(res, ignore_index=True)

def forecast_preview(df, param, year_next=2025):
    ts = (df[df['parameter']==param]
            .groupby(['year','month'])['value'].median().rename('y').reset_index())
    ts = ts.sort_values(['year','month']).reset_index(drop=True)
    ts['t'] = np.arange(len(ts))+1
    for m in range(1,12):
        ts[f'm{m}'] = (ts['month']==m).astype(int)
    X = ts[['t'] + [f'm{m}' for m in range(1,12)]].values
    y = ts['y'].values.reshape(-1,1)
    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
    yhat = (X @ beta).ravel()
    resid = (y.ravel() - yhat)
    future = []
    last_t = int(ts['t'].iloc[-1])
    for i, m in enumerate(range(1,13), start=1):
        row = {'year': year_next, 'month': m, 't': last_t + i}
        row.update({f'm{k}': 1 if m==k else 0 for k in range(1,12)})
        future.append(row)
    F = pd.DataFrame(future)
    Xf = F[['t'] + [f'm{k}' for k in range(1,12)]].values
    yhat_f = (Xf @ beta).ravel()
    s = resid.std(ddof=X.shape[1]) if len(resid)>X.shape[1] else resid.std()
    F['forecast']=yhat_f
    F['lo80']=yhat_f-1.28*s; F['hi80']=yhat_f+1.28*s
    F['lo95']=yhat_f-1.96*s; F['hi95']=yhat_f+1.96*s
    return F[['year','month','forecast','lo80','hi80','lo95','hi95']]

def scope_filter(df, scope):
    if scope == 'ALL': return df
    if scope == 'NO_STP': return df[~df['is_stp']].copy()
    return df

# -------------------- External data & ML helpers --------------------

def fetch_rainfall_monthly(lat: float, lon: float, start_date: str, end_date: str):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "auto"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    dates = js.get("daily", {}).get("time", [])
    precip = js.get("daily", {}).get("precipitation_sum", [])
    df = pd.DataFrame({"date": pd.to_datetime(dates), "rain_mm": precip})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    m = df.groupby(["year","month"], as_index=False)["rain_mm"].sum()
    m["date"] = pd.to_datetime(m["year"].astype(str) + "-" + m["month"].astype(str) + "-01")
    return m[["year","month","date","rain_mm"]]

def monthly_quality(df, param: str):
    m = (df[df["parameter"]==param]
         .groupby(["year","month"], as_index=False)["value"].median()
         .rename(columns={"value":"q_median"}))
    m["date"] = pd.to_datetime(m["year"].astype(str)+"-"+m["month"].astype(str)+"-01")
    return m[["year","month","date","q_median"]]

def correlate_quality_rain(df, param: str, rain_df: pd.DataFrame, lags=(0,1,2)):
    q = monthly_quality(df, param)
    out_rows = []
    for L in lags:
        jj = q.merge(rain_df, on=["year","month"], how="inner")
        jj["date"] = pd.to_datetime(jj["year"].astype(str)+"-"+jj["month"].astype(str)+"-01")
        jj = jj.sort_values(["year","month"]).reset_index(drop=True)
        jj["rain_lag"] = jj["rain_mm"].shift(L)
        c = jj["q_median"].corr(jj["rain_lag"])
        out_rows.append({"lag_months": int(L), "corr": float(c) if pd.notna(c) else None})
    return pd.DataFrame(out_rows)

def build_predictor_rain(df, param: str, rain_df: pd.DataFrame):
    from sklearn.linear_model import RidgeCV
    q = monthly_quality(df, param)
    jj = q.merge(rain_df, on=["year","month"], how="inner")
    jj["date"] = pd.to_datetime(jj["year"].astype(str)+"-"+jj["month"].astype(str)+"-01")
    jj = jj.sort_values(["year","month"]).reset_index(drop=True)
    for k in [0,1,2]:
        jj[f"rain_lag{k}"] = jj["rain_mm"].shift(k)
    jj["t"] = range(1, len(jj)+1)
    for m in range(1,12):
        jj[f"m{m}"] = (jj["month"]==m).astype(int)
    jj = jj.dropna().reset_index(drop=True)
    if len(jj) < 12:
        return {"error": "Not enough data after alignment for modeling."}
    y = np.log1p(jj["q_median"]).values
    X_cols = ["t","rain_lag0","rain_lag1","rain_lag2"] + [f"m{m}" for m in range(1,12)]
    X = jj[X_cols].values
    model = RidgeCV(alphas=[0.1,1.0,10.0]).fit(X, y)
    yhat = model.predict(X)
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))
    r2 = float(1 - np.sum((y-yhat)**2)/np.sum((y - y.mean())**2))
    coefs = dict(zip(X_cols, model.coef_.tolist()))
    intercept = float(model.intercept_)
    clim = rain_df.groupby("month", as_index=False)["rain_mm"].mean().rename(columns={"rain_mm":"rain_clim"})
    F = pd.DataFrame({"year":[2025]*12,"month":list(range(1,13))})
    F = F.merge(clim, on="month", how="left")
    F["date"] = pd.to_datetime(F["year"].astype(str)+"-"+F["month"].astype(str)+"-01")
    F["rain_lag0"] = F["rain_clim"]
    F["rain_lag1"] = F["rain_clim"]
    F["rain_lag2"] = F["rain_clim"]
    last_t = jj["t"].iloc[-1]
    F["t"] = range(last_t+1, last_t+1+len(F))
    for m in range(1,12):
        F[f"m{m}"] = (F["month"]==m).astype(int)
    Xf = F[["t","rain_lag0","rain_lag1","rain_lag2"] + [f"m{m}" for m in range(1,12)]].values
    yhat_f = model.predict(Xf)
    F["forecast_q"] = np.expm1(yhat_f)
    return {"rmse": rmse, "r2": r2, "alpha": float(model.alpha_), "intercept": intercept, "coefs": coefs, "fitted": jj.assign(pred=np.expm1(yhat)), "forecast": F[["year","month","date","forecast_q"]]}
