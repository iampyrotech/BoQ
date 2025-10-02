# %%
import sf_quant.data as sfd
import sf_quant.optimizer as sfo
import sf_quant.backtester as sfb
import sf_quant.performance as sfp
import polars as pl
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

N_CPUS = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))


# %% [markdown]
# Dataloader requests:
# * Ask for dlst codes and rets in crsp data loader
# * Ask for compustat in data loader
# * Ask for someone to fix crsp data loader bc it is not on grpquant

# %% [markdown]
# Problems
# - We do not have the same data as CHS
# - Our target is not exactly same measure as theirs
# - Our hi-prob of failure stocks perform too well

# %% [markdown]
# ### Loading in data

# %% [markdown]
# CRSP

# %%
crsp_pricing = pl.read_csv(
    r"/home/porter77/sf_fall_2025/sf-quant-labs/crsp_pricing.csv",
    infer_schema_length=10000,
    schema_overrides={
        "CUSIP": pl.Utf8,
        "NCUSIP": pl.Utf8
    }
)

crsp_delisting = pl.read_csv(
    r"/home/porter77/sf_fall_2025/sf-quant-labs/crsp_delisting.csv",
    infer_schema_length=10000,
    schema_overrides={
        "CUSIP": pl.Utf8,
        "NCUSIP": pl.Utf8
    }
)

crsp=crsp_pricing.join(crsp_delisting,on=['date','PERMNO','CUSIP'])

# %% [markdown]
# COMPUSTAT

# %%
compustat = pl.read_csv(
    r'/home/porter77/sf_fall_2025/sf-quant-labs/compustat_updated.csv',
    infer_schema_length=10000,
    schema_overrides={
        "GVKEY": pl.Utf8,
        "CUSIP": pl.Utf8,
        "TIC": pl.Utf8,
        "CONM": pl.Utf8
    }
)
#get the compustat to crsp link
link=pl.read_csv(r'/home/porter77/sf_fall_2025/sf-quant-labs/link.csv')

link=link.rename({'LPERMNO':'PERMNO','GVKEY':'gvkey'})
compustat=compustat.join(link,on=['gvkey','cusip'])
compustat=compustat.rename({'datadate':'date'})


# %%
#merge compustat and crsp
data=crsp.join(compustat,on=['PERMNO','date'],how='left')

# %% [markdown]
# Get SP500 data for Size for exret factors in logit
# 
# - SP500 RET
# - SP MKT CAP

# %%
#get 
sp500=pl.read_csv(r'/home/porter77/sf_fall_2025/sf-quant-labs/sp500.csv')

sp500 = sp500.with_columns(
    pl.col("caldt").str.strptime(pl.Date, format="%Y-%m-%d").alias("caldt_date")
)

# 2. Create a "date" column set to the month-end
sp500 = sp500.with_columns(
    pl.col("caldt_date").dt.month_end().alias("date")
)
sp500

# %%
#make date a dt.date so it can match with sp500
data = data.with_columns(
    pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d").alias("date")
)

data=data.join(sp500,on='date')

# %% [markdown]
# ### Signal Construction

# %% [markdown]
# #### Cleaning and Defining Vars

# %% [markdown]
# ##### Adding Delisted return

# %% [markdown]
# "Because we are studying the returns to distressed stocks, it is important to handle carefully the returns to stocks that are delisted and thus disappear from the CRSP database. In many cases, CRSP reports a delisting return for the final month of the firm’s life; we have 6,481 such delisting returns in our sample and we use them where they are available. Otherwise, we use the last available full-month return in CRSP. In some cases, this effectively assumes that our portfolios sell distressed stocks at the end of the month before delisting, which imparts an upward bias to the returns on distressed-stock portfolios (Shumway (1997) and Shumway and Warther (1999)).13 We assume that the proceeds from sales of delisted stocks are reinvested in each portfolio in proportion to the weights of the remaining stocks in the portfolio. In a few cases, stocks are delisted and then re-enter the database, but we do not include these stocks in the sample after the first delisting. We treat firms that fail as equivalent to delisted firms, even if CRSP continues to report returns for these firms. That is,our portfolios sell stocks of companies that fail and we use the latest available CRSP data to calculate a final return on such stocks."

# %% [markdown]
# We do not restrict the crosssection of firms to include only share codes 10 and 11, as Hong, Lim, and Stein (2000) do, but we
# have checked that all of our results are robust to such a restriction

# %%
# #filter to shrcd 10,11
# data = data.filter(pl.col("SHRCD").is_in([10, 11])).sort(["PERMNO", "date"])

#cast dlret to float
data = data.with_columns(
    (pl.col('DLRET').cast(pl.Float32,strict=False)).alias('DLRET')
)
#cast ret o float
data=data.with_columns(
    (pl.col('RET').cast(pl.Float32,strict=False)).alias('RET')
)

#use dlret for ret when no ret exists
data=data.with_columns(
    pl.when((pl.col('RET').is_null()) & (pl.col('DLRET').is_not_null()))
    .then(pl.col('DLRET'))
    .otherwise(pl.col('RET'))
    .alias('RET')
)

#multiply them to get the total return for that month
data=data.with_columns(
    pl.when((pl.col('RET').is_not_null()) & (pl.col('DLRET').is_not_null()))
    .then(
        ((pl.col('RET')+1)* (1+pl.col('DLRET')))-1

    )
    .otherwise(pl.col('RET'))
    .alias('RET')
)

# %% [markdown]
# ##### Defining Vars

# %% [markdown]
# $$
# \text{Total Assets (adjusted)}_{i,t} = TA_{i,t} + 0.1 \big( ME_{i,t} - BE_{i,t} \big)
# 
# $$
# $$
# \text{Market-to-Book}_{i,t} \;=\; \frac{\text{Market Equity}_{i,t}}{\text{Book Equity}_{i,t}}
# 
# 
# $$
# 

# %%
#we lag any accounting data by 2 months

#sort before shifting
data = data.sort(["PERMNO", "date"])


#make price always postiive
data = data.with_columns(
    pl.col("PRC").abs().alias("PRC")
)

data = data.with_columns(
    (((pl.col("seqq") + pl.col("txditcq").fill_null(0) - pl.col("pstkrq").fill_null(0))).shift(2).over('PERMNO'))
    .alias("beq")
)
data=data.with_columns(
    (pl.when(pl.col('beq')<0)
    .then(pl.lit(1/1_000_000))
    .otherwise(pl.col('beq')))
    .alias('beq')
)

#now make the market equity
data = data.with_columns(
    (pl.col("PRC") * (pl.col("SHROUT")/1000))
    .alias("meq")
)

#make M/B

data=data.with_columns(
    (pl.col('meq')/pl.col('beq')).alias('mbq')
)
#make total assets adjusted
data = data.with_columns(
    (((pl.col("atq")).shift(2).over('PERMNO')) + (0.1 * (pl.col("meq") - pl.col("beq"))))
    .alias("taq_adjusted")
)



# %%
# data = data.with_columns([
#     pl.col("beq").fill_null(
#         pl.col("beq").mean().over("date")
#     ).alias("beq")
# ])

# #filter out the infiinte mbq
# data = data.filter(~pl.col("mbq").is_infinite())


# %% [markdown]
# Make the rest of the Factors

# %% [markdown]
# $$
# RSIZE_{i,t} = \log \left( \frac{\text{Firm Market Equity}_{i,t}}{\text{Total S\&P500 Market Value}_{t}} \right)
# $$
# 
# $$
# EXRET_{i,t} = \log(1+R_{i,t}) - \log(1+R_{S\&P500,t})
# $$
# 
# $$
# NITA_{i,t} = \frac{\text{Net Income}_{i,t}}{\text{Total Assets (adjusted)}_{i,t}}
# $$
# 
# $$
# TLTA_{i,t} = \frac{\text{Total Liabilities}_{i,t}}{\text{Total Assets (adjusted)}_{i,t}}
# $$
# 
# $$
# NIMTA_{i,t} = \frac{\text{Net Income}_{i,t}}{\text{Firm Market Equity}_{i,t} + \text{Total Liabilities}_{i,t}}
# $$
# 
# $$
# TLMTA_{i,t} = \frac{\text{Total Liabilities}_{i,t}}{\text{Firm Market Equity}_{i,t} + \text{Total Liabilities}_{i,t}}
# $$
# 
# $$
# CASHMTA_{i,t} = \frac{\text{Cash and Short Term Investments}_{i,t}}{\text{Firm Market Equity}_{i,t} + \text{Total Liabilities}_{i,t}}
# $$
# 
# 

# %%

data=data.with_columns(
    ((pl.col('meq')/(pl.col('totval')/1000)).log()).alias('rsize')
)

#exret
data = data.with_columns(
    ( ((1 + pl.col("RET")).log()) - ((1 + pl.col("sprtrn")).log())).alias("exret")
)

data = data.sort(["PERMNO", "date"])
#NIMTA
data=data.with_columns(
    (((pl.col('niq')).shift(2).over('PERMNO'))/(pl.col('meq')+((pl.col('ltq')).shift(2).over('PERMNO')))).alias('nimta')
)

# i had chat write these two bc i didnt want to write it out myself
data = data.sort(["PERMNO", "date"])
#TLMTA
data = data.with_columns(
    (
        (pl.col("ltq") ).shift(2).over("PERMNO")
        / (pl.col("meq") + (pl.col("ltq")).shift(2).over("PERMNO"))
    ).alias("tlmta")
)
data = data.sort(["PERMNO", "date"])
#CSHMTA
data = data.with_columns(
    (
        (pl.col("cheq") ).shift(2).over("PERMNO")
        / (pl.col("meq") + (pl.col("ltq")).shift(2).over("PERMNO"))
    ).alias("cshmta")
)

data = data.sort(["PERMNO", "date"])
#NITA
data=data.with_columns(
   ( ((pl.col('niq')).shift(2).over("PERMNO"))
    / ((pl.col('taq_adjusted')))).alias('nita')
)

data = data.sort(["PERMNO", "date"])
#TLTA
data=data.with_columns(
    (((pl.col('ltq')).shift(2).over("PERMNO"))
    / ((pl.col('taq_adjusted')))).alias('tlta')
)

# %%
#fill in nita and tlta

data = data.with_columns([
    pl.col("nita").fill_null(
        pl.col("nita").mean().over("date")
    ).alias("nita"),

    pl.col("tlta").fill_null(
        pl.col("tlta").mean().over("date")
    ).alias("tlta")
])

# #fill in cshmta
# data = data.with_columns([
#     pl.col("cshmta").fill_null(
#         pl.col("cshmta").mean().over("date")
#     ).alias("cshmta")
# ])



# %% [markdown]
# $$
# SIGMA_{i,t-1,t-3} \;=\;
# \left( 252 \cdot \frac{1}{N-1} 
# \sum_{k \in \{t-1, t-2, t-3\}} r_{i,k}^2 \right)^{\tfrac{1}{2}}.
# $$
# 

# %%
#getting sigma
crsp_daily=pl.read_csv(r'/home/porter77/sf_fall_2025/sf-quant-labs/crsp_daily.csv')


#do some cleaning
crsp_daily=crsp_daily.with_columns(
    (pl.col('PRC').abs()).alias('PRC')
)
crsp_daily=crsp_daily.with_columns(
    pl.col('RET').cast(pl.Float32,strict=False)
)

crsp_daily = crsp_daily.with_columns(
    pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d").alias("date")
)

crsp_daily=crsp_daily.sort(['PERMNO','date'])
crsp_daily=crsp_daily.with_columns(
    (pl.col('RET').rolling_std(63).over('PERMNO')).alias('std_3mo')  #63 trading days in 3 month
)
crsp_daily = crsp_daily.with_columns(
    (pl.col("std_3mo") * (252**0.5)).alias('sigma_ann')
)


#i think I prolly could have just used monthly data for this bc now i have to align it back to my other data
#so here is chats code for doing that

# 5) align to month end (keep as-is)
crsp_daily = crsp_daily.with_columns(pl.col("date").dt.month_end().alias("date"))

# 6) collapse: take the last NON-NULL sigma in each month
sigma_m = (
    crsp_daily
    .group_by(["PERMNO", "date"])
    .agg(pl.col("sigma_ann").drop_nulls().last().alias("sigma"))
)

#fill remaining nulls with same-month cross sectional mean
sigma_cs = sigma_m.group_by("date").agg(pl.col("sigma").mean().alias("sigma_cs_mean"))
sigma_m = (
    sigma_m.join(sigma_cs, on="date", how="left")
           .with_columns(
               pl.coalesce([pl.col("sigma"), pl.col("sigma_cs_mean")]).alias("sigma")
           )
           .drop("sigma_cs_mean")
)

# # then join sigma_m back to your monthly panel `data` on [PERMNO, date]
# data = data.join(sigma_m, on=["PERMNO", "date"], how="left")

#merge sigma_m to my data
data=data.join(sigma_m,on=['PERMNO','date'],how='left')

# %% [markdown]
# $$
# NIMTAAVG_{t-1,t-12} = \frac{1 - \phi^3}{1 - \phi^{12}} 
# \left( NIMTA_{t-1,t-3} + \cdots + \phi^9 NIMTA_{t-10,t-12} \right),
# $$
# 
# $$
# EXRETAVG_{t-1,t-12} = \frac{1 - \phi}{1 - \phi^{12}}
# \left( EXRET_{t-1} + \cdots + \phi^{11} EXRET_{t-12} \right),
# $$
# 
# where $\phi = 2^{-\tfrac{1}{3}}$
# 
# When lagged excess returns or profitability are missing, we replace them with their crosssectional means in order to avoid losing observations. 
# 

# %%
#fill in nimta and tlmta

data = data.with_columns([
    pl.col("nimta").fill_null(
        pl.col("nimta").mean().over("date")
    ).alias("nimta"),

    pl.col("tlmta").fill_null(
        pl.col("tlmta").mean().over("date")
    ).alias("tlmta")
])

# %%
import polars as pl

# ---------- config ----------
phi = 0.5 ** (1 / 3)          

# Make sure data are sorted and monthly, with columns:
# PERMNO, date (monthly), nimta, exret
data = data.sort(["PERMNO", "date"])

# ===== EXRETAVG: EWMA over t-1..t-12 (monthly) =====
# EXRETAVG
ex_terms = [
    pl.col("exret").shift(i).over("PERMNO") * (phi ** (i - 1))
    for i in range(1, 13)
]

data = data.with_columns(
    (((1 - phi) / (1 - phi**12)) * sum(ex_terms)).alias("exretavg")
)

# ===== NIMTAAVG: 3-month block averages, then quarterly decay =====
# First build the 3-month trailing mean aligned at t using t-1..t-3:
# 3-month trailing average aligned at t (covering t-1, t-2, t-3)
nimta_3m = (
    pl.col("nimta")
    .shift(1).rolling_mean(window_size=3, min_periods=3)
    .over("PERMNO")
)

# now just shift the block-averages in multiples of 3 months
q1 = nimta_3m
q2 = nimta_3m.shift(3)
q3 = nimta_3m.shift(6)
q4 = nimta_3m.shift(9)

data = data.with_columns(
    (
        ((1 - phi**3) / (1 - phi**12))
        * (q1 + (phi**3)*q2 + (phi**6)*q3 + (phi**9)*q4)
    ).alias("nimtaavg")
)

# Optional: if you want strictly finite outputs (drop early rows without history)
data = data.with_columns([
    pl.when(pl.col("exretavg").is_finite()).then(pl.col("exretavg")).otherwise(None).alias("exretavg"),
    pl.when(pl.col("nimtaavg").is_finite()).then(pl.col("nimtaavg")).otherwise(None).alias("nimtaavg"),
])


# %%
#winsorize the price
data = data.with_columns(
    pl.when(pl.col("PRC") > 15)
      .then(15)
      .otherwise(pl.col("PRC"))
      .alias("prc_winsor")
)

#log it
data = data.with_columns(
    pl.col("prc_winsor").log().alias("log_prc")
)


# %% [markdown]
# ##### Winsorizing

# %%
# Variables to winsorize
vars_to_winsor = [
    "nita",        
    "tlta",
    "exret",
    "sigma",
    "rsize",
    'log_prc',
    'nimtaavg',
    'exretavg',
    'mbq',
    'tlmta',
    'cshmta'
]

# 1. Clean infinities -> null so they don't pollute quantiles
data = data.with_columns([
    pl.when(pl.col(c).is_infinite())
      .then(None)
      .otherwise(pl.col(c))
      .alias(c)
    for c in vars_to_winsor
])

# 2. Compute pooled 5% / 95% cutoffs
qdf = data.select(
    *[pl.col(c).quantile(0.05, interpolation="lower").alias(f"{c}_lo") for c in vars_to_winsor],
    *[pl.col(c).quantile(0.95, interpolation="higher").alias(f"{c}_hi") for c in vars_to_winsor],
).to_dicts()[0]


# 3. Apply winsorization; keep originals, add *_w columns
for c in vars_to_winsor:
    lo = qdf[f"{c}_lo"]
    hi = qdf[f"{c}_hi"]
    data = data.with_columns(
        pl.when(pl.col(c) < lo).then(lo)
         .when(pl.col(c) > hi).then(hi)
         .otherwise(pl.col(c))
         .alias(f"{c}_w")
    )


start = pl.date(1965, 1, 1)   # Jan 1965
end   = pl.date(2003, 12, 31) # Dec 2003


# filter in Polars
in_sample = data.filter(
    (pl.col("date") >= start) & (pl.col("date") <= end)
)


# %% [markdown]
# #### Model 2 From CHS [EXACTLY] 

# %% [markdown]
# 1 month lookahead

# %%
# 1) grab coefficients from statsmodels
betas_exact = {'nimtaavg_w': -29.67,
 'tlmta_w': 3.36,
 'rsize_w': 0.082,
 'exretavg_w': -7.35,
 'sigma_w': 1.48,
 'cshmta_w': -2.40,
 'log_prc_w':-0.937,
 'mbq_w':.054
 }

b0_exact = -9.08

# 2) build the linear predictor in Polars
linpred = pl.lit(b0_exact)
for name, coef in betas_exact.items():
    # skip any coef whose column isn't present (defensive)
    if name in in_sample.columns:
        linpred = linpred + pl.col(name) * float(coef)

# 3) logistic transform to get probability for every row
in_sample = in_sample.with_columns(
    ((pl.lit(1.0) / (pl.lit(1.0) + (-linpred).exp())).shift(1)).alias("p_failure")
)


start=dt.date(2004,1,1)
end=dt.date(2024,12,31)

out_of_sample=data.filter(pl.col('date').is_between(start,end))



##############################################################################################
# 1) grab coefficients from statsmodels
betas_exact = {'nimtaavg_w': -29.67,
 'tlmta_w': 3.36,
 'rsize_w': 0.082,
 'exretavg_w': -7.35,
 'sigma_w': 1.48,
 'cshmta_w': -2.40,
 'log_prc_w':-0.937,
 'mbq_w':.054
 }

b0_exact = -9.08

# 2) build the linear predictor in Polars
linpred = pl.lit(b0_exact)
for name, coef in betas_exact.items():
    # skip any coef whose column isn't present (defensive)
    if name in out_of_sample.columns:
        linpred = linpred + pl.col(name) * float(coef)

# 3) logistic transform to get probability for every row
out_of_sample = out_of_sample.with_columns(
    ((pl.lit(1.0) / (pl.lit(1.0) + (-linpred).exp())).shift(1)).alias("p_failure")
)


# 1) grab coefficients from statsmodels
betas_exact = {'nimtaavg_w': -20.26,
 'tlmta_w': 1.42,
 'rsize_w': -0.045,
 'exretavg_w': -7.13,
 'sigma_w': 1.41,
 'cshmta_w': -2.13,
 'log_prc_w':-0.058,
 'mbq_w':.075
 }

b0_exact = -9.16

# 2) build the linear predictor in Polars
linpred = pl.lit(b0_exact)
for name, coef in betas_exact.items():
    # skip any coef whose column isn't present (defensive)
    if name in out_of_sample.columns:
        linpred = linpred + pl.col(name) * float(coef)

# 3) logistic transform to get probability for every row
out_of_sample = out_of_sample.with_columns(
    ((pl.lit(1.0) / (pl.lit(1.0) + (-linpred).exp())).shift(1)).alias("p_failure_12")
)



# %% [markdown]
# 12 month lookahead

# %%
# 1) grab coefficients from statsmodels
betas_exact = {'nimtaavg_w': -20.26,
 'tlmta_w': 1.42,
 'rsize_w': -0.045,
 'exretavg_w': -7.13,
 'sigma_w': 1.41,
 'cshmta_w': -2.13,
 'log_prc_w':-0.058,
 'mbq_w':.075
 }

b0_exact = -9.16

# 2) build the linear predictor in Polars
linpred = pl.lit(b0_exact)
for name, coef in betas_exact.items():
    # skip any coef whose column isn't present (defensive)
    if name in in_sample.columns:
        linpred = linpred + pl.col(name) * float(coef)

# 3) logistic transform to get probability for every row
in_sample = in_sample.with_columns(
    ((pl.lit(1.0) / (pl.lit(1.0) + (-linpred).exp())).shift(1)).alias("p_failure_12")
)



# %% [markdown]
# ### Portfolio Formation

# %%
ff5=pl.read_csv(r'/home/porter77/sf_fall_2025/sf-quant-labs/ff5.csv')

# ff5["date"] is like 196307 → make it the last calendar day of that month
ff5 = ff5.with_columns(
    (
        pl.concat_str([pl.col("date").cast(pl.Utf8), pl.lit("01")])  # "19630701"
        .str.strptime(pl.Date, format="%Y%m%d")                        # 1963-07-01
        .dt.offset_by("1mo")                                        # 1963-08-01
        .dt.replace(day=1)                                          # 1963-08-01 (idempotent)
        - pl.duration(days=1)                                       # 1963-07-31
    ).alias("date")
)


# %% [markdown]
# ### MVO backtest

# %%
start=dt.date(2004,1,1)
end=dt.date(2024,12,31)

barra_data=sfd.assets.load_assets(start=start,end=end,in_universe=True,columns=['date','historical_beta','ticker','price','return','barrid','cusip','specific_return','specific_risk','market_cap','total_risk','predicted_beta'])
barra_data=barra_data.rename({'cusip':'CUSIP'})
barra_data = barra_data.with_columns(
    pl.col("CUSIP").str.slice(0, 8).alias("CUSIP")
)


# %% [markdown]
# Merge out_of_sample with barra_data


# %%
out=out_of_sample.join(barra_data,on=['date','CUSIP'],how='right')

# %% [markdown]
# Make the alphas

# %%
#score the signal

out=out.with_columns(
    (((pl.col('p_failure'))-(pl.col('p_failure').mean()))/(pl.col('p_failure').std()*-1))
    .alias('p_failure_score')
)

out=out.with_columns(
    (((pl.col('p_failure_12'))-(pl.col('p_failure_12').mean()))/(pl.col('p_failure_12').std()*-1))
    .alias('p_failure_12_score')
)

# %%
IC=.05

out=out.with_columns(
    ((pl.col('p_failure_score'))*IC*(pl.col('specific_risk')/100)).alias('1_y_alpha')
)

out=out.with_columns(
    ((pl.col('p_failure_12_score'))*IC*(pl.col('specific_risk')/100)).alias('1_mo_alpha')
)




# %%
#get rid of out duplicates
out_dedup = out.unique(subset=["barrid", "date"], keep="first")


# %%
alpha1m=out_dedup
alpha1y=out_dedup

alpha1m=alpha1m.select(['date','barrid','1_mo_alpha','predicted_beta'])
alpha1y=alpha1y.select(['date','barrid','1_y_alpha','predicted_beta'])

alpha1m=alpha1m.rename({'1_mo_alpha':'alpha'})
alpha1y=alpha1y.rename({'1_y_alpha':'alpha'})

alpha1m=alpha1m.with_columns(
    (pl.col('alpha').fill_null(0)).alias('alpha')
)

alpha1y=alpha1y.with_columns(
    (pl.col('alpha').fill_null(0)).alias('alpha')
)

# %%
mkt_cap=out.select(['market_cap','barrid','date','rsize','meq','totval'])
mkt_cap=mkt_cap.with_columns(
    (((pl.col('market_cap')/1000)*13)/pl.col('totval')).alias('signal_weight')
)



# %%
alpha1m_vw=alpha1m.join(mkt_cap,on=['barrid','date'])
alpha1m_vw=alpha1m_vw.with_columns(
    ((pl.col('alpha'))*pl.col('signal_weight')).alias('alpha')
)

alpha1m_vw = alpha1m_vw.unique(subset=["barrid", "date"], keep="first")
alpha1m_vw=alpha1m_vw.select(['date','barrid','alpha','predicted_beta'])

# %%
alpha1y_vw=alpha1y.join(mkt_cap,on=['barrid','date'])
alpha1y_vw=alpha1y_vw.with_columns(
    ((pl.col('alpha'))*pl.col('signal_weight')).alias('alpha')
)

alpha1y_vw = alpha1y_vw.unique(subset=["barrid", "date"], keep="first")
alpha1y_vw=alpha1y_vw.select(['date','barrid','alpha','predicted_beta'])

# %% [markdown]
# **Do the backests**

# %%
out.filter(pl.col('1_y_alpha').is_not_null())

# %% [markdown]
# 1 month alpha

# %%
constraints=[sfo.NoBuyingOnMargin(),sfo.FullInvestment(),sfo.LongOnly()]
# 


month_alpha_bt=sfb.backtest_parallel(alpha1y,constraints=constraints,gamma=2,n_cpus=N_CPUS)

# %%
month_port_rets=sfp.generate_returns_from_weights(month_alpha_bt)

sfp.generate_returns_chart(month_port_rets,title='Distress Signals Backtest (1mo alpha)',file_name='bt.png')
sum_stats=sfp.generate_summary_table(month_port_rets)
sum_stats.write_csv('sum_stats.csv')

# %%
month_ret_ff=month_port_rets.pivot(index='date',values='return',on='portfolio')
bin_cols=['active','benchmark','total']

# --- to pandas ---
ff = ff5.to_pandas().copy()
rets_pd = month_ret_ff.to_pandas().copy()

# --- dates & indexing ---
ff["date"] = pd.to_datetime(ff["date"])          # your FF is month-end already
rets_pd["date"] = pd.to_datetime(rets_pd["date"])
ff = ff.set_index("date").sort_index()
rets_pd = rets_pd.set_index("date").sort_index()

# --- factor cleaning/scaling ---
factor_cols = ["Mkt-RF","SMB","HML","RMW","CMA","RF"]
ff[factor_cols] = ff[factor_cols].apply(pd.to_numeric, errors="coerce")
# Scale % → decimals if needed
if ff[factor_cols].abs().median().max() > 0.5:
    ff[factor_cols] = ff[factor_cols] / 100.0

# --- align on date ---
M = rets_pd.join(ff[factor_cols], how="inner")

# portfolio columns (or keep your existing bin_cols if defined)
try:
    bin_cols = [c for c in bin_cols if c in M.columns]
except NameError:
    bin_cols = [c for c in rets_pd.columns if c not in factor_cols]

# ensure numeric
M[bin_cols] = M[bin_cols].apply(pd.to_numeric, errors="coerce")
factors = M[["Mkt-RF","SMB","HML","RMW","CMA"]]

# --- regressions ---
coefs, tstats, skipped = {}, {}, {}
for b in bin_cols:
    tmp = pd.concat([M[[b, "RF"]], factors], axis=1)
    tmp.columns = ["port", "RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    tmp["y"] = tmp["port"] - tmp["RF"]
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    if tmp.shape[0] <= 6:   # need > #params (const + 5 factors = 6)
        skipped[b] = f"not enough obs (n={tmp.shape[0]})"
        continue

    X = sm.add_constant(tmp[["Mkt-RF","SMB","HML","RMW","CMA"]], has_constant="add")
    y = tmp["y"]
    res = sm.OLS(y, X).fit()
    coefs[b], tstats[b] = res.params, res.tvalues

# --- results table ---
row_order = ["const","Mkt-RF","SMB","HML","RMW","CMA"]
coef_df  = pd.DataFrame(coefs).reindex(row_order)
tstat_df = pd.DataFrame(tstats).reindex(row_order)
table = coef_df.round(4).astype(str) + " (" + tstat_df.round(2).astype(str) + ")"

table.to_csv('table.csv')
